import logging
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING, Dict, Any
from ...image.tracing import trace_surface
from ...core.geo import contours, Geometry
from ...core.matrix import Matrix
from ...core.ops import Ops, SectionType
from ...core.vectorization_spec import TraceSpec
from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem
from .base import OpsProducer, CutSide


if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser

logger = logging.getLogger(__name__)


class CutOrder(Enum):
    """Defines the processing order for nested paths."""

    INSIDE_OUTSIDE = auto()
    OUTSIDE_INSIDE = auto()


class ContourProducer(OpsProducer):
    """
    Uses the tracer to find all paths in a shape. Can optionally trace
    only the outermost paths, ignoring any holes.
    """

    def __init__(
        self,
        remove_inner_paths: bool = False,
        path_offset_mm: float = 0.0,
        cut_side: CutSide = CutSide.CENTERLINE,
        cut_order: CutOrder = CutOrder.INSIDE_OUTSIDE,
        override_threshold: bool = False,
        threshold: float = 0.5,
    ):
        """
        Initializes the ContourProducer.

        Args:
            remove_inner_paths: If True, only the outermost paths (outlines)
                                are traced, and inner holes are ignored.
            path_offset_mm: An absolute distance to offset the generated path.
            cut_side: The rule for determining the final cut side.
            cut_order: The processing order for nested paths.
            override_threshold: If True, ignores source vectors and re-traces
                                the rendered surface.
            threshold: The brightness threshold (0.0-1.0) for re-tracing.
        """
        super().__init__()
        self.remove_inner_paths = remove_inner_paths
        self.path_offset_mm = path_offset_mm
        self.cut_side = cut_side
        self.cut_order = cut_order
        self.override_threshold = override_threshold
        self.threshold = threshold

    @property
    def requires_full_render(self) -> bool:
        # If we are overriding the threshold, we rely on the tracer, which
        # operates on a raster surface. We must force the pipeline to
        # render one.
        return self.override_threshold

    def run(
        self,
        laser: "Laser",
        surface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
    ) -> WorkPieceArtifact:
        if workpiece is None:
            raise ValueError("ContourProducer requires a workpiece context.")

        # 1. Calculate total offset from producer and step settings
        kerf_mm = (settings or {}).get("kerf_mm", laser.spot_size_mm[0])
        kerf_compensation = kerf_mm / 2.0
        total_offset = 0.0
        if self.cut_side == CutSide.CENTERLINE:
            total_offset = 0.0  # Centerline ignores path offset
        elif self.cut_side == CutSide.OUTSIDE:
            total_offset = self.path_offset_mm + kerf_compensation
        elif self.cut_side == CutSide.INSIDE:
            total_offset = -self.path_offset_mm - kerf_compensation

        # 2. Get base contours and determine the correct scaling matrix
        base_contours = []

        # Check if we have source vectors.
        has_vector_source = (
            workpiece
            and workpiece.boundaries
            and not workpiece.boundaries.is_empty()
        )

        # If override_threshold is True, we SKIP the vector source and fall
        # through to the raster tracing logic below.
        if has_vector_source and not self.override_threshold:
            assert workpiece.boundaries
            base_contours = workpiece.boundaries.split_into_contours()
            sx, sy = workpiece.matrix.get_abs_scale()
            scaling_matrix = Matrix.scale(sx, sy)
        elif surface:
            # Fall back to raster tracing if no vectors OR if override
            # is active
            spec = None
            if self.override_threshold:
                # Create a spec to force the specific threshold
                spec = TraceSpec(
                    threshold=self.threshold,
                    auto_threshold=False,
                    invert=False,
                )

            base_contours = trace_surface(surface, vectorization_spec=spec)

            width_mm, height_mm = workpiece.size
            px_width, px_height = surface.get_width(), surface.get_height()
            if px_width > 0 and px_height > 0:
                scale_x = width_mm / px_width
                scale_y = height_mm / px_height

                # Pixels are Y-down (Top-Left 0,0). World is Y-up.
                # 1. Scale Y by negative to flip axis.
                # 2. Translate Y by +height_mm to move the now-negative shape
                #    back up into the positive quadrant.
                scaling_matrix = Matrix.translation(
                    0, height_mm
                ) @ Matrix.scale(scale_x, -scale_y)
            else:
                scaling_matrix = Matrix.identity()
        else:
            # No vectors and no surface, so there is nothing to trace.
            scaling_matrix = Matrix.identity()

        # 3. Scale all contours to their final millimeter size *first*.
        mm_space_contours = []
        for geo in base_contours:
            geo.transform(scaling_matrix.to_4x4_numpy())
            mm_space_contours.append(geo)

        # 4. Normalize.
        target_contours = []
        if mm_space_contours:
            if has_vector_source and not self.override_threshold:
                # For direct vector sources, trust the input and don't
                # perform polygon cleaning, which would discard open paths.
                target_contours = contours.normalize_winding_orders(
                    mm_space_contours
                )

            else:
                # For raster-traced sources, clean up the contours.
                target_contours = contours.normalize_winding_orders(
                    mm_space_contours
                )

        # 5. Apply offsets.
        composite_geo = Geometry()
        for geo in target_contours:
            composite_geo.extend(geo)

        if abs(total_offset) > 1e-6:
            # Attempt to apply the offset (grow/shrink).
            grown_geometry = composite_geo.grow(total_offset)

            # Check if the grow operation failed (returned empty geometry).
            # This can happen with complex or malformed input shapes.
            if grown_geometry.is_empty() and not composite_geo.is_empty():
                logger.warning(
                    f"ContourProducer for '{workpiece.name}' failed to apply "
                    f"an offset of {total_offset:.3f} mm. This can be "
                    "caused by micro-gaps or self-intersections in the "
                    "source geometry. Falling back to the un-offset path."
                )
                # Fall back to the original, un-offset geometry.
                final_geometry = composite_geo
            else:
                # The grow operation was successful or input was empty.
                final_geometry = grown_geometry
        else:
            # No offset was requested, so use the composite geometry.
            final_geometry = composite_geo

        # 6. Create Ops by splitting into optimizable groups
        final_ops = Ops()
        if not final_geometry.is_empty():
            final_ops.set_laser(laser.uid)

            if self.remove_inner_paths:
                # Simple case: remove inner paths and create one optimizable
                # group
                final_geometry = final_geometry.remove_inner_edges()
                final_ops.ops_section_start(
                    SectionType.VECTOR_OUTLINE, workpiece.uid
                )
                final_ops.extend(Ops.from_geometry(final_geometry))
                final_ops.ops_section_end(SectionType.VECTOR_OUTLINE)
            else:
                # Complex case: separate inner and outer paths into two groups
                split_func = final_geometry.split_inner_and_outer_contours
                inner_contours, outer_contours = split_func()

                # Combine the lists of contours into composite Geometry objects
                outer_geo = Geometry()
                for geo in outer_contours:
                    outer_geo.extend(geo)

                inner_geo = Geometry()
                for geo in inner_contours:
                    inner_geo.extend(geo)

                group1 = (
                    inner_geo
                    if self.cut_order == CutOrder.INSIDE_OUTSIDE
                    else outer_geo
                )
                group2 = (
                    outer_geo
                    if self.cut_order == CutOrder.INSIDE_OUTSIDE
                    else inner_geo
                )

                if not group1.is_empty():
                    final_ops.ops_section_start(
                        SectionType.VECTOR_OUTLINE, workpiece.uid
                    )
                    final_ops.extend(Ops.from_geometry(group1))
                    final_ops.ops_section_end(SectionType.VECTOR_OUTLINE)

                if not group2.is_empty():
                    final_ops.ops_section_start(
                        SectionType.VECTOR_OUTLINE, workpiece.uid
                    )
                    final_ops.extend(Ops.from_geometry(group2))
                    final_ops.ops_section_end(SectionType.VECTOR_OUTLINE)

        # 7. Create the artifact.
        return WorkPieceArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=workpiece.size,
            generation_size=workpiece.size,
        )

    @property
    def supports_kerf(self) -> bool:
        return True

    def to_dict(self) -> dict:
        """Serializes the producer configuration."""
        return {
            "type": self.__class__.__name__,
            "params": {
                "remove_inner_paths": self.remove_inner_paths,
                "path_offset_mm": self.path_offset_mm,
                "cut_side": self.cut_side.name,
                "cut_order": self.cut_order.name,
                "override_threshold": self.override_threshold,
                "threshold": self.threshold,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContourProducer":
        """Deserializes a dictionary into an ContourProducer instance."""
        params = data.get("params", {})
        cut_side_str = params.get(
            "cut_side", params.get("kerf_mode", "CENTERLINE")
        )
        try:
            cut_side = CutSide[cut_side_str]
        except KeyError:
            cut_side = CutSide.CENTERLINE

        cut_order_str = params.get("cut_order", "INSIDE_OUTSIDE")
        try:
            cut_order = CutOrder[cut_order_str]
        except KeyError:
            cut_order = CutOrder.INSIDE_OUTSIDE

        return cls(
            remove_inner_paths=params.get("remove_inner_paths", False),
            path_offset_mm=params.get(
                "path_offset_mm", params.get("offset_mm", 0.0)
            ),
            cut_side=cut_side,
            cut_order=cut_order,
            override_threshold=params.get("override_threshold", False),
            threshold=params.get("threshold", 0.5),
        )

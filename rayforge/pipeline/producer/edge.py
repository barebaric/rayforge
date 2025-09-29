from typing import Optional, TYPE_CHECKING, Dict, Any
from .base import OpsProducer, PipelineArtifact, CoordinateSystem, CutSide
from ...image.tracing import trace_surface
from ...core.matrix import Matrix
from ...core.ops import (
    Ops,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
)
from ...core.geo import contours, Geometry

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


class EdgeTracer(OpsProducer):
    """
    Uses the Potrace engine to trace paths in a shape. Can optionally trace
    only the outermost paths, ignoring any holes.
    """

    def __init__(
        self,
        remove_inner_paths: bool = False,
        path_offset_mm: float = 0.0,
        cut_side: CutSide = CutSide.OUTSIDE,
    ):
        """
        Initializes the EdgeTracer.

        Args:
            remove_inner_paths: If True, only the outermost paths (outlines)
                                are traced, and inner holes are ignored.
            path_offset_mm: An absolute distance to offset the generated path.
            cut_side: The rule for determining the final cut side.
        """
        super().__init__()
        self.remove_inner_paths = remove_inner_paths
        self.path_offset_mm = path_offset_mm
        self.cut_side = cut_side

    def run(
        self,
        laser,
        surface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
    ) -> PipelineArtifact:
        if workpiece is None:
            raise ValueError("EdgeTracer requires a workpiece context.")

        final_ops = Ops()
        final_ops.add(
            OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, workpiece.uid)
        )
        final_ops.set_power((settings or {}).get("power", 0))

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
        if (
            workpiece
            and workpiece.vectors
            and not workpiece.vectors.is_empty()
        ):
            base_contours = workpiece.vectors.split_into_contours()
            sx, sy = workpiece.matrix.get_abs_scale()
            scaling_matrix = Matrix.scale(sx, sy)
        else:  # Fall back to raster tracing
            base_contours = trace_surface(surface)
            width_mm, height_mm = workpiece.size
            px_width, px_height = surface.get_width(), surface.get_height()
            if px_width > 0 and px_height > 0:
                scale_x = width_mm / px_width
                scale_y = height_mm / px_height
                scaling_matrix = Matrix.scale(scale_x, scale_y)
            else:
                scaling_matrix = Matrix.identity()

        # 3. Scale all contours to their final millimeter size *first*.
        mm_space_contours = []
        for geo in base_contours:
            geo.transform(scaling_matrix.to_4x4_numpy())
            mm_space_contours.append(geo)

        # 4. Now, process the mm-space contours: normalize, filter, and offset.
        final_geometry = Geometry()
        if mm_space_contours:
            normalized_contours = contours.normalize_winding_orders(
                mm_space_contours
            )

            if self.remove_inner_paths:
                target_contours = contours.filter_to_external_contours(
                    normalized_contours
                )
            else:
                target_contours = normalized_contours

            # Combine all contours into a single geometry object before
            # offsetting. This allows the offsetting algorithm to correctly
            # handle holes.
            composite_geo = Geometry()
            for geo in target_contours:
                composite_geo.commands.extend(geo.commands)

            if abs(total_offset) > 1e-6:
                final_geometry = composite_geo.grow(total_offset)
            else:
                final_geometry = composite_geo

        # 5. Convert to Ops. No further scaling is needed.
        final_ops.extend(Ops.from_geometry(final_geometry))
        final_ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))

        # 6. Create the artifact. The ops are pre-scaled, so they are not
        #    scalable in the pipeline cache sense.
        return PipelineArtifact(
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
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeTracer":
        """Deserializes a dictionary into an EdgeTracer instance."""
        params = data.get("params", {})
        cut_side_str = params.get(
            "cut_side", params.get("kerf_mode", "OUTSIDE")
        )
        try:
            cut_side = CutSide[cut_side_str]
        except KeyError:
            cut_side = CutSide.OUTSIDE

        return cls(
            remove_inner_paths=params.get("remove_inner_paths", False),
            path_offset_mm=params.get(
                "path_offset_mm", params.get("offset_mm", 0.0)
            ),
            cut_side=cut_side,
        )

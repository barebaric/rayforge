import logging
from enum import Enum, auto
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, Optional

from raygeo.geo import Geometry, Matrix
from raygeo.ops.part import Part
from raygeo.ops import Ops
from raygeo.ops.assembly.contour import contour
from raygeo.ops.types import SectionType

from rayforge.core.vectorization_spec import TraceSpec
from rayforge.image.tracing import trace_surface
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.producer.base import CutSide, OpsProducer
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.laser import Laser

logger = logging.getLogger(__name__)


class CutOrder(Enum):
    """Defines the processing order for nested paths."""

    INSIDE_OUTSIDE = auto()
    OUTSIDE_INSIDE = auto()

    def label(self) -> str:
        """Return a translatable label for this cut order."""
        labels = {
            self.INSIDE_OUTSIDE: _("Inside-Outside"),
            self.OUTSIDE_INSIDE: _("Outside-Inside"),
        }
        return labels[self]


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
        overcut: float = 0.0,
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
            overcut: Distance to extend closed contours past their start
                        point so that the cut overlaps itself.
        """
        super().__init__()
        self.remove_inner_paths = remove_inner_paths
        self.path_offset_mm = path_offset_mm
        self.cut_side = cut_side
        self.cut_order = cut_order
        self.override_threshold = override_threshold
        self.threshold = threshold
        self.overcut = overcut

    @staticmethod
    def _empty_artifact(workpiece, generation_id):
        return WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=workpiece.size,
            generation_size=workpiece.size,
            generation_id=generation_id,
        )

    @property
    def requires_full_render(self) -> bool:
        return self.override_threshold

    def run(
        self,
        laser: "Laser",
        surface,
        pixels_per_mm,
        *,
        generation_id: int,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        context: Optional[ProgressContext] = None,
    ) -> WorkPieceArtifact:
        if workpiece is None:
            raise ValueError("ContourProducer requires a workpiece context.")

        settings = settings or {}
        kerf_mm = settings.get("kerf_mm", laser.spot_size_mm[0])

        # 2. Build Part and use contour assembler
        has_vector_source = (
            workpiece
            and workpiece.boundaries
            and not workpiece.boundaries.is_empty()
        )
        part = None

        # If override_threshold is True, SKIP the vector source and fall
        # through to the raster tracing logic below.
        if has_vector_source and not self.override_threshold:
            part = workpiece.to_part()
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
            traced = trace_surface(surface, vectorization_spec=spec)
            if traced:
                merged = Geometry()
                width_mm, height_mm = workpiece.size
                px_w, px_h = surface.get_width(), surface.get_height()
                if px_w > 0 and px_h > 0:
                    scale_x = width_mm / px_w
                    scale_y = height_mm / px_h
                    transform = Matrix.translation(
                        0, height_mm
                    ) @ Matrix.scale(scale_x, -scale_y)
                    for g in traced:
                        g.transform(transform)
                        merged.extend(g)
                part = Part(geometry=merged, size_mm=workpiece.size)

        if part is None or not part.has_geometry():
            logger.warning(
                "ContourProducer for '%s': no geometry available — "
                "returning empty artifact",
                workpiece.name,
            )
            return self._empty_artifact(workpiece, generation_id)

        # 3. Run the contour assembler (handles kerf, offset, overcut,
        #    ordering, and curve fitting internally)
        cut_order_str = (
            "inside_outside"
            if self.cut_order == CutOrder.INSIDE_OUTSIDE
            else "outside_inside"
        )
        tolerance = settings.get("arc_tolerance", 0.03)
        allow_arcs = settings.get(
            "machine_supports_arcs", settings.get("output_arcs", True)
        )
        supports_curves = settings.get("machine_supports_curves", False)

        result = contour(
            part,
            kerf_mm=kerf_mm,
            path_offset_mm=self.path_offset_mm,
            cut_side=self.cut_side.name.lower(),
            overcut=self.overcut,
            cut_order=cut_order_str,
            remove_inner=self.remove_inner_paths,
            arc_tolerance=tolerance,
            allow_arcs=allow_arcs,
            supports_curves=supports_curves,
        )

        # 4. Create Ops — one section per contour to enable proper travel
        final_ops = Ops()
        if result.ops.len() > 0:
            final_ops.set_head(laser.uid)
            contour_geo = result.ops.to_geometry()
            contour_list = contour_geo.split_into_contours()
            for c in contour_list:
                final_ops.ops_section_start(
                    SectionType.VECTOR_OUTLINE, workpiece.uid
                )
                final_ops.extend(Ops.from_geometry(c))
                final_ops.ops_section_end(SectionType.VECTOR_OUTLINE)

        # 5. Create the artifact.
        return WorkPieceArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=workpiece.size,
            generation_size=workpiece.size,
            generation_id=generation_id,
        )

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
                "overcut": self.overcut,
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
            overcut=params.get("overcut", 0.0),
        )

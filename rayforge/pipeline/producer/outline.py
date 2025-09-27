from typing import List, Optional, TYPE_CHECKING
from .base import OpsProducer, PipelineArtifact, CoordinateSystem
from ...image.tracing import trace_surface
from ...core.geo import Geometry, contours
from ...core.ops import (
    Ops,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
)

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


class OutlineTracer(OpsProducer):
    """
    Uses the Potrace engine and filters the results to trace only the
    outermost paths of a shape, ignoring any holes.
    """

    def run(
        self,
        laser,
        surface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        y_offset_mm: float = 0.0,
    ) -> PipelineArtifact:
        if workpiece is None:
            raise ValueError("OutlineTracer requires a workpiece context.")

        final_ops = Ops()
        final_ops.add(
            OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, workpiece.uid)
        )

        source_contours: List[Geometry] = []
        source_dims = None
        coord_system = CoordinateSystem.PIXEL_SPACE

        # If the workpiece has vectors, split them into contours.
        if (
            workpiece
            and workpiece.vectors
            and not workpiece.vectors.is_empty()
        ):
            source_contours = workpiece.vectors.split_into_contours()
            coord_system = CoordinateSystem.NATIVE_VECTOR_SPACE
            _x, _y, w_mm, h_mm = workpiece.vectors.rect()
            source_dims = (w_mm, h_mm)
        # If no vectors, fall back to raster tracing the surface.
        else:
            source_contours = trace_surface(surface, pixels_per_mm)
            coord_system = CoordinateSystem.PIXEL_SPACE
            source_dims = (surface.get_width(), surface.get_height())

        # Apply the centralized, reusable filtering algorithm.
        external_contours = contours.filter_to_external_contours(
            source_contours
        )

        # Convert the final list of external contours to Ops.
        outline_ops = Ops()
        for geo in external_contours:
            outline_ops.extend(Ops.from_geometry(geo))

        final_ops.extend(outline_ops)
        final_ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))

        return PipelineArtifact(
            ops=final_ops,
            is_scalable=True,
            source_coordinate_system=coord_system,
            source_dimensions=source_dims,
        )

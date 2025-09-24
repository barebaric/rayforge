from typing import List, Optional, TYPE_CHECKING
from .base import OpsProducer
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
    Uses a tracer and filters the results to keep only the
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
    ) -> Ops:
        if workpiece is None:
            raise ValueError("OutlineTracer requires a workpiece context.")

        final_ops = Ops()
        final_ops.add(
            OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, workpiece.uid)
        )

        source_contours: List[Geometry] = []
        # If the workpiece has vectors, split them into contours.
        if (
            workpiece
            and workpiece.vectors
            and not workpiece.vectors.is_empty()
        ):
            source_contours = workpiece.vectors.split_into_contours()
        # If no vectors, fall back to raster tracing the surface.
        else:
            source_contours = trace_surface(surface, pixels_per_mm)

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
        return final_ops

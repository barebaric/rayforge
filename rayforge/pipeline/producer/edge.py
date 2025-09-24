from typing import Optional, TYPE_CHECKING
from .base import OpsProducer
from ...image.tracing import trace_surface
from ...core.ops import (
    Ops,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
)

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


class EdgeTracer(OpsProducer):
    """
    Uses the Potrace engine to trace all paths in a shape, including
    both external outlines and internal holes.
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
            raise ValueError("EdgeTracer requires a workpiece context.")

        final_ops = Ops()
        final_ops.add(
            OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, workpiece.uid)
        )

        # If the workpiece has geometry, the "Edge" strategy is to simply
        # return them unmodified.
        if workpiece and workpiece.vectors:
            vector_ops = Ops.from_geometry(workpiece.vectors)
            final_ops.extend(vector_ops)
        # If no geometry, fall back to raster tracing the surface.
        else:
            # 1. Use the centralized, robust tracing function.
            geometries = trace_surface(surface, pixels_per_mm)

            # 2. The "Edge" strategy keeps all geometries, so no filtering.
            # 3. Convert all resulting geometries to Ops.
            raster_trace_ops = Ops()
            for geo in geometries:
                raster_trace_ops.extend(Ops.from_geometry(geo))

            final_ops.extend(raster_trace_ops)

        final_ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
        return final_ops

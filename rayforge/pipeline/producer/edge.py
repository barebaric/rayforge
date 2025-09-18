from typing import List, Optional, TYPE_CHECKING
import potrace
from .potrace_base import PotraceProducer
from ...core.ops import (
    Ops,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
)

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


class EdgeTracer(PotraceProducer):
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
            raster_trace_ops = super().run(
                laser,
                surface,
                pixels_per_mm,
                workpiece=workpiece,
                y_offset_mm=y_offset_mm,
            )
            final_ops.extend(raster_trace_ops)

        final_ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
        return final_ops

    def _filter_curves(
        self, curves: List[potrace.Curve]
    ) -> List[potrace.Curve]:
        """
        The "Contour" or "Edge" strategy is to keep all paths, so this
        filter does nothing.
        """
        return curves

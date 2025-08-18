from typing import List, Optional, TYPE_CHECKING
import potrace
from .potrace_base import PotraceProducer
from ...core.ops import Ops

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
        # If the workpiece has source_ops, the "Edge" strategy is to simply
        # return them unmodified.
        if workpiece and workpiece.source_ops:
            return workpiece.source_ops

        # If no source_ops, fall back to raster tracing the surface.
        return super().run(
            laser,
            surface,
            pixels_per_mm,
            workpiece=workpiece,
            y_offset_mm=y_offset_mm,
        )

    def _filter_curves(
        self, curves: List[potrace.Curve]
    ) -> List[potrace.Curve]:
        """
        The "Contour" or "Edge" strategy is to keep all paths, so this
        filter does nothing.
        """
        return curves

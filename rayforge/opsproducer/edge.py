from typing import List
import potrace
from .potrace_base import PotraceProducer


class EdgeTracer(PotraceProducer):
    """
    Uses the Potrace engine to trace all paths in a shape, including
    both external outlines and internal holes.
    """

    def _filter_curves(
        self, curves: List[potrace.Curve]
    ) -> List[potrace.Curve]:
        """
        The "Contour" or "Edge" strategy is to keep all paths, so this
        filter does nothing.
        """
        return curves

from typing import List, Optional, TYPE_CHECKING
import potrace
from .potrace_base import PotraceProducer
from ...core.ops import Ops

if TYPE_CHECKING:
    from ...importer.base import Importer


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
        importer: "Optional[Importer]" = None,
        y_offset_mm: float = 0.0,
    ) -> Ops:
        # Vector fast path: If the importer can provide vector ops directly,
        # use them. They are already in the correct millimeter coordinate
        # system.
        if importer:
            vector_ops = importer.get_vector_ops()
            if vector_ops:
                return vector_ops

        # Fallback to standard raster tracing. The base class will handle the
        # pixel-to-millimeter conversion using the provided pixels_per_mm.
        return super().run(
            laser,
            surface,
            pixels_per_mm,
            importer=importer,
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

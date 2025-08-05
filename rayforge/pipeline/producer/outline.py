from typing import List
import numpy as np
import cv2
import potrace
from .potrace_base import PotraceProducer


class OutlineTracer(PotraceProducer):
    """
    Uses the Potrace engine and filters the results to trace only the
    outermost paths of a shape, ignoring any holes.
    """

    def _filter_curves(
        self, curves: List[potrace.Curve]
    ) -> List[potrace.Curve]:
        """
        Returns only curves that are not contained within any other curve.
        """
        if len(curves) <= 1:
            return curves

        polygons = self._curves_to_polygons(curves)

        external_curves = []
        for i, curve in enumerate(curves):
            if not self._is_contained(i, curve, polygons):
                external_curves.append(curve)
        return external_curves

    def _curves_to_polygons(
        self, curves: List[potrace.Curve]
    ) -> List[np.ndarray]:
        """
        Converts Potrace curves to OpenCV-compatible polygons for testing.
        """
        return [
            np.array([s.end_point for s in c], dtype=np.int32).reshape(
                (-1, 1, 2)
            )
            for c in curves
        ]

    def _is_contained(
        self,
        curve_index: int,
        curve_to_test: potrace.Curve,
        polygons: List[np.ndarray],
    ) -> bool:
        """Checks if a curve is inside any of the other polygons."""
        test_point = tuple(map(int, curve_to_test.start_point))
        for i, polygon in enumerate(polygons):
            if i == curve_index:
                continue
            if cv2.pointPolygonTest(polygon, test_point, False) > 0:
                return True
        return False

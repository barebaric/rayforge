from typing import List, Optional, TYPE_CHECKING
import numpy as np
import cv2
import potrace
from .potrace_base import PotraceProducer
from ...core.ops import Ops, Command

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


class OutlineTracer(PotraceProducer):
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
    ) -> Ops:
        # If the workpiece has source_ops, apply the outline-finding algorithm
        # to them.
        if (
            workpiece
            and workpiece.source_ops
            and len(workpiece.source_ops) > 0
        ):
            return self._filter_vector_ops_xor(workpiece.source_ops)

        # If no source_ops, fall back to raster tracing the surface.
        return super().run(
            laser,
            surface,
            pixels_per_mm,
            workpiece=workpiece,
            y_offset_mm=y_offset_mm,
        )

    def _filter_vector_ops_xor(self, ops: Ops) -> Ops:
        """
        Filters vector paths using a robust XOR rasterization method. This
        correctly handles all nesting, winding order, and complex geometry by
        simulating a graphics importer's even-odd fill rule and then
        extracting ONLY the external contours.
        """
        all_points = [cmd.end for cmd in ops.commands if cmd.end is not None]
        if not all_points:
            return Ops()

        # Capture the Z from the first point to re-apply later. This assumes a
        # constant Z for the purpose of this 2D outlining algorithm.
        first_z = (
            all_points[0][2]
            if len(all_points[0]) > 2 and all_points[0][2] is not None
            else 0.0
        )

        # 1. Determine the bounding box and a safe scaling factor from the XY
        # plane
        points_array = np.array([p[:2] for p in all_points], dtype=np.float32)
        min_x, min_y = points_array.min(axis=0)
        max_x, max_y = points_array.max(axis=0)

        width_mm = max_x - min_x
        height_mm = max_y - min_y

        if width_mm <= 0 or height_mm <= 0:
            return ops  # Not a shape with area, return as is.

        # Create a canvas. We can cap the size for performance.
        CANVAS_MAX_DIM = 4096
        scale = min(CANVAS_MAX_DIM / width_mm, CANVAS_MAX_DIM / height_mm)

        width_px = int(np.ceil(width_mm * scale)) + 2  # Add padding
        height_px = int(np.ceil(height_mm * scale)) + 2

        # 2. Deconstruct Ops into polygons
        paths: List[List[Command]] = []
        current_path: List[Command] = []
        for cmd in ops.commands:
            if cmd.is_travel_command():
                if current_path:
                    paths.append(current_path)
                current_path = [cmd]
            else:
                current_path.append(cmd)
        if current_path:
            paths.append(current_path)

        # 3. Create the XOR mask
        mask = np.zeros((height_px, width_px), dtype=np.uint8)

        for path in paths:
            poly_points = [cmd.end for cmd in path if cmd.end is not None]
            if len(poly_points) < 3:
                continue

            # Scale and translate XY points to fit the canvas
            scaled_poly = np.array(
                [p[:2] for p in poly_points], dtype=np.float32
            )
            scaled_poly[:, 0] = (scaled_poly[:, 0] - min_x) * scale + 1
            scaled_poly[:, 1] = (scaled_poly[:, 1] - min_y) * scale + 1

            # Create a temporary mask for the current polygon
            temp_mask = np.zeros_like(mask)
            cv2.fillPoly(temp_mask, [scaled_poly.astype(np.int32)], (255,))

            # XOR the temporary mask with the main mask
            cv2.bitwise_xor(mask, temp_mask, mask)

        # 4. Find contours on the final mask.
        #    RETR_EXTERNAL retrieves only the extreme outer edges
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 5. Reconstruct Ops from the final (outer) contours
        final_ops = Ops()
        if not contours:
            return final_ops

        for contour in contours:
            if len(contour) < 2:
                continue

            # Scale points back to original millimeter coordinates
            contour_float = contour.astype(np.float32)
            contour_float[:, 0, 0] = (
                contour_float[:, 0, 0] - 1
            ) / scale + min_x
            contour_float[:, 0, 1] = (
                contour_float[:, 0, 1] - 1
            ) / scale + min_y

            # Reshape from (N, 1, 2) to (N, 2) for easier iteration.
            final_points = contour_float.reshape(-1, 2)

            # Add commands to trace this contour
            start_x, start_y = final_points[0]
            final_ops.move_to(start_x, start_y, z=first_z)
            for x, y in final_points[1:]:
                final_ops.line_to(x, y, z=first_z)
            final_ops.close_path()  # Ensure the shape is closed

        return final_ops

    def _filter_curves(
        self, curves: List[potrace.Curve]
    ) -> List[potrace.Curve]:
        """
        (Raster path) Returns only curves that are not contained within any
        other curve.
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
        (Raster path) Converts Potrace curves to OpenCV-compatible polygons
        for testing.
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
        """
        (Raster path) Checks if a curve is inside any of the other polygons.
        """
        test_point = tuple(map(int, curve_to_test.start_point))
        for i, polygon in enumerate(polygons):
            if i == curve_index:
                continue
            if cv2.pointPolygonTest(polygon, test_point, False) > 0:
                return True
        return False

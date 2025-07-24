import math
from typing import Optional
from ..models.ops import Ops, LineToCommand, MoveToCommand
from .transformer import OpsTransformer
from ..tasker import BaseExecutionContext
from .arcwelder.points import remove_duplicates


class Smooth(OpsTransformer):
    """Smooths Ops points with a moving average, keeping sharp corners."""

    def __init__(self, smooth_window=7, corner_angle_threshold=45):
        """Initialize with window size and corner angle threshold."""
        self.smooth_window = max(1, smooth_window)
        self.corner_threshold = math.radians(corner_angle_threshold)

    def run(
        self,
        ops: Ops,
        context: Optional[BaseExecutionContext] = None,
    ):
        segments = list(ops.segments())
        ops.clear()

        total_segments = len(segments)
        for i, segment in enumerate(segments):
            if context and context.is_cancelled():
                return  # Abort if cancelled

            if self._is_line_only_segment(segment):
                points = [cmd.end for cmd in segment]
                smoothed = self._smooth_segment(points)
                ops.move_to(*smoothed[0])
                for point in smoothed[1:]:
                    ops.line_to(*point)
            else:
                for command in segment:
                    ops.add(command)

            if context and total_segments > 0:
                context.set_progress((i + 1) / total_segments)

    def _is_line_only_segment(self, segment):
        if len(segment) <= 1 or not isinstance(segment[0], MoveToCommand):
            return False
        return all(isinstance(cmd, LineToCommand) for cmd in segment[1:])

    def _smooth_segment(self, points):
        if len(points) < 3 or self.smooth_window <= 1:
            return remove_duplicates(points)
        half_window = (self.smooth_window - 1) // 2
        smoothed = []
        is_corner = [False] * len(points)
        for i in range(1, len(points) - 1):
            p0, p1, p2 = points[i - 1], points[i], points[i + 1]
            angle = self._angle_between(p0, p1, p2)
            if abs(math.pi - angle) > self.corner_threshold:
                is_corner[i] = True
        for i in range(len(points)):
            if is_corner[i]:
                smoothed.append(points[i])
            else:
                start = i
                while (
                    start > 0
                    and i - start < half_window
                    and not is_corner[start]
                ):
                    start -= 1
                if is_corner[start]:
                    start += 1
                end = i
                while (
                    end < len(points) - 1
                    and end - i < half_window
                    and not is_corner[end]
                ):
                    end += 1
                if end < len(points) and is_corner[end]:
                    end -= 1
                window = points[start:end + 1]
                avg_x = sum(p[0] for p in window) / len(window)
                avg_y = sum(p[1] for p in window) / len(window)
                smoothed.append((avg_x, avg_y))
        return smoothed

    def _angle_between(self, p0, p1, p2):
        """Calculate angle (radians) between vectors p0-p1 and p1-p2."""
        v1x, v1y = p1[0] - p0[0], p1[1] - p0[1]
        v2x, v2y = p2[0] - p1[0], p2[1] - p1[1]
        dot = v1x * v2x + v1y * v2y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 == 0 or mag2 == 0:
            return 0
        cos_theta = min(1.0, max(-1.0, dot / (mag1 * mag2)))
        return math.acos(cos_theta)

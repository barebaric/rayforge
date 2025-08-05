import math
import numpy as np
from itertools import groupby
from typing import Optional, Dict, Any, List
from scipy.optimize import least_squares
from ...shared.tasker.proxy import BaseExecutionContext
from ...core.ops import (
    Ops,
    Command,
    LineToCommand,
    ArcToCommand,
    MoveToCommand,
)
from .base import OpsTransformer


def remove_duplicates(segment):
    """
    Removes *consecutive* duplicates from a list of points.
    """
    return [k for (k, v) in groupby(segment)]


def are_colinear(points, tolerance=0.01):
    """
    Check if all points are colinear within a given tolerance.

    Args:
        points: List of (x, y) tuples.
        tolerance: Max perpendicular distance from the line (default 0.01).

    Returns:
        bool: True if all points are colinear within tolerance.
    """
    if len(points) < 2:
        return True  # Fewer than 2 points are trivially colinear
    if len(points) == 2:
        return True  # Two points define a line

    # Define line by first and last points
    p1, p2 = points[0], points[-1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    line_length = math.hypot(dx, dy)

    if line_length == 0:
        # Points are coincident, check if all are within tolerance
        return all(math.hypot(p[0] - p1[0], p[1] - p1[1]) < tolerance
                   for p in points)

    # Check perpendicular distance of each point to the line p1-p2
    for p in points[1:-1]:  # Skip endpoints as they define the line
        # Vector from p1 to p
        vx = p[0] - p1[0]
        vy = p[1] - p1[1]
        # Perpendicular distance = |ax + by + c| / sqrt(a^2 + b^2)
        # Line equation: ax + by + c = 0, where
        #                a=dy, b=-dx, c=-(dy*p1x - dx*p1y)
        dist = abs(dy * vx - dx * vy) / line_length
        if dist > tolerance:
            return False
    return True


def is_clockwise(points):
    """
    Determines direction using cross product.
    """
    if len(points) < 3:
        return False

    p1, p2, p3 = points[0], points[1], points[2]
    cross = (
        (p2[0] - p1[0]) * (p3[1] - p2[1])
        - (p2[1] - p1[1]) * (p3[0] - p2[0])
    )
    return cross < 0


def arc_direction(points, center):
    xc, yc = center
    cross_sum = 0.0
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        dx0 = x0 - xc
        dy0 = y0 - yc
        dx1 = x1 - xc
        dy1 = y1 - yc
        cross = dx0 * dy1 - dy0 * dx1
        cross_sum += cross
    return cross_sum < 0  # True for clockwise


def fit_circle(points):
    """
    Fit a circle to points, return (center, radius, error) or None.
    Error is max of point-to-arc deviation.
    """
    if len(points) < 3 or are_colinear(points):
        return None

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Initial guess: mean center and average radius
    x0, y0 = np.mean(x), np.mean(y)
    r0 = np.mean(np.sqrt((x-x0)**2 + (y-y0)**2))

    # Fit circle using least squares
    result = least_squares(
        lambda p: np.sqrt((x-p[0])**2 + (y-p[1])**2) - p[2],
        [x0, y0, r0],
        method='lm'
    )
    xc, yc, r = result.x
    center = (xc, yc)

    # Point-to-arc error: max deviation of points from circle
    distances = np.sqrt((x-xc)**2 + (y-yc)**2)
    point_error = np.max(np.abs(distances - r))

    # Total error: max of point fit and arc deviation
    return center, r, point_error


def arc_to_polyline_deviation(points, center, radius):
    """
    Compute max deviation of an arc from the original polyline.
    Args:
        points: List of (x, y) tuples forming the polyline.
        center: (xc, yc) tuple, center of the fitted circle.
        radius: Radius of the fitted circle.
    Returns:
        float: Max perpendicular distance from arc to polyline segments.
    """
    if len(points) < 2:
        return 0.0
    xc, yc = center
    max_deviation = 0.0

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        segment_length = math.hypot(dx, dy)

        if segment_length == 0:
            distance = math.hypot(x1 - xc, y1 - yc)
            deviation = abs(distance - radius)
            max_deviation = max(max_deviation, deviation)
            continue

        # Distances from center to endpoints
        d1 = math.hypot(x1 - xc, y1 - yc)
        d2 = math.hypot(x2 - xc, y2 - yc)

        # If segment exceeds diameter, use endpoint deviations
        if segment_length > 2 * radius:
            deviation = max(abs(d1 - radius), abs(d2 - radius))
        else:
            # Vectors from center to points
            v1x, v1y = x1 - xc, y1 - yc
            v2x, v2y = x2 - xc, y2 - yc

            # Dot product to find angle
            dot = v1x * v2x + v1y * v2y
            mag1 = math.hypot(v1x, v1y)
            mag2 = math.hypot(v2x, v2y)
            if mag1 < 1e-6 or mag2 < 1e-6:
                deviation = abs(d1 - radius) if mag1 < 1e-6 \
                       else abs(d2 - radius)
            else:
                cos_theta = min(1.0, max(-1.0, dot / (mag1 * mag2)))
                theta = math.acos(cos_theta)
                # Sagitta based on actual arc angle
                sagitta = radius * (1 - math.cos(theta / 2))
                # Endpoint deviations if off-arc
                endpoint_dev = max(abs(d1 - radius), abs(d2 - radius))
                deviation = max(sagitta, endpoint_dev)

        max_deviation = max(max_deviation, deviation)
    return max_deviation


def contains_command(segment, cmdcls):
    return any(isinstance(cmd, cmdcls) for cmd in segment)


def split_into_segments(commands):
    """
    Splits commands into logical segments while tracking current position.
    - Segments with arc_to are preceded by explicit or implicit move_to.
    - State commands are standalone segments.
    """
    segments = []
    current_segment: List[Command] = []
    current_pos = None  # Track current position

    for cmd in commands:
        if cmd.is_travel_command():
            # Start new segment
            if current_segment:
                segments.append(current_segment)
            current_segment = [cmd]
            current_pos = cmd.end

        elif isinstance(cmd, ArcToCommand):
            # Start new segment
            if contains_command(current_segment, LineToCommand):
                segments.append(current_segment)
                current_segment = [cmd]
            else:
                current_segment.append(cmd)
            current_pos = cmd.end

        elif isinstance(cmd, LineToCommand):
            # Add to current segment and track position
            if contains_command(current_segment, ArcToCommand):
                segments.append(current_segment)
                current_segment = []
            if not current_segment:
                if current_pos is None:
                    raise ValueError("line_to requires a starting position")
                current_segment.append(MoveToCommand(current_pos))
            current_segment.append(cmd)
            current_pos = cmd.end

        elif cmd.is_state_command():
            # All other commands are standalone
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            segments.append([cmd])

        else:
            raise ValueError(f"Unsupported command: {cmd}")

    if current_segment:
        segments.append(current_segment)

    return segments


class ArcWeld(OpsTransformer):
    """
    Converts line sequences into arcs using pre-validated geometric utilities.

    tolerance: Max allowed deviation from arc
    min_points: Minimum number of points to attempt arc fitting
    max_points: Maximum number of points to attempt arc fitting
    max_angular_step: Max angle between points on the arc
    """
    def __init__(self,
                 enabled: bool = True,
                 tolerance=0.049,
                 min_points=6,
                 max_points=15,
                 max_angular_step=75):
        super().__init__(enabled=enabled)
        self.tolerance = tolerance
        self.min_points = min_points
        self.max_points = max_points
        self.max_step = math.radians(max_angular_step)

    @property
    def label(self) -> str:
        return _("Arc Weld Path")

    @property
    def description(self) -> str:
        return _("Welds lines into arcs for smoother paths")

    def run(
        self, ops: Ops, context: Optional[BaseExecutionContext] = None
    ) -> None:
        segments = split_into_segments(ops.commands)
        ops.clear()

        for segment in segments:
            if contains_command(segment, LineToCommand):
                self.process_segment([cmd.end for cmd in segment], ops)
            else:
                for command in segment:
                    ops.add(command)

    def process_segment(self, segment, ops):
        if not segment:
            return

        # Bail out early for short segments.
        segment = remove_duplicates(segment)
        length = len(segment)
        if length < self.min_points:
            ops.move_to(*segment[0])
            for point in segment[1:]:
                ops.line_to(*point)
            return

        # Walk along the segment trying to find arcs that may fit.
        ops.move_to(*segment[0])
        index = 1
        while index < length:
            # Consume colinear points first
            colinear_points = self._count_colinear_points(segment, index-1)

            if colinear_points:
                ops.line_to(*segment[index+colinear_points-2])
                index += colinear_points
                continue

            # Try to find an arc that fits the points starting at index.
            # fit_segment already performs a fast deviation calculation,
            # but it only checks deviation from original points and not
            # from the lines that connect the points.
            arc, arc_end = self._find_longest_valid_arc(segment, index-1)
            if arc:
                # Perform better, but more expensive, deviation calculation.
                deviation = arc_to_polyline_deviation(segment[index-1:arc_end],
                                                      *arc[:2])
                if deviation <= self.tolerance:
                    self._add_arc_command(segment, index-1, arc_end, arc, ops)
                    index = arc_end  # Move to the last point of the arc
                    continue

            # Ending up here, no fitting arc was found at the current index.
            ops.line_to(*segment[index])
            index += 1

    def _count_colinear_points(self, segment, start):
        """Advance index past colinear points, returning the end index."""
        length = len(segment)
        if length-start < 3:
            return 0

        end = start+3
        found = None
        while end < length and are_colinear(segment[start:end+1]):
            end += 1
            found = end-start

        return found

    def _add_arc_command(self, segment, start, end, arc, ops):
        center, radius, _ = arc
        start_point = segment[start]
        end_point = segment[end-1]

        # Calculate I and J offsets
        i = center[0] - start_point[0]
        j = start_point[1] - center[1]  # Inverted Y-axis

        clockwise = arc_direction(segment[start:end], center)

        ops.arc_to(end_point[0], end_point[1], i, j, clockwise)

    def _find_longest_valid_arc(self, segment, start_index):
        max_search = min(len(segment)-start_index, self.max_points)

        for length in range(max_search, self.min_points-1, -1):
            end = start_index+length
            assert end - start_index >= self.min_points
            subsegment = segment[start_index:end]
            arc = fit_circle(subsegment)
            if self._is_valid_arc(subsegment, arc):
                return arc, end

        return None, start_index

    def _is_valid_arc(self, subsegment, arc):
        if arc is None:
            return False
        center, radius, error = arc
        if error > self.tolerance or radius < 1 or radius > 100:
            return False

        # Angular continuity checks
        prev_angle = None
        for x, y in subsegment:
            dx = x - center[0]
            dy = y - center[1]
            angle = math.atan2(dy, dx)
            if prev_angle is not None:
                delta = abs(angle - prev_angle)
                delta = min(delta, 2 * math.pi - delta)
                if delta > self.max_step:
                    return False
            prev_angle = angle
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the transformer's configuration to a dictionary."""
        data = super().to_dict()
        data.update({
            'tolerance': self.tolerance,
            'min_points': self.min_points,
            'max_points': self.max_points,
            'max_angular_step': math.degrees(self.max_step),
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArcWeld':
        """Creates an ArcWeld instance from a dictionary."""
        if data.get('name') != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(
            enabled=data.get('enabled', True),
            tolerance=data.get('tolerance', 0.049),
            min_points=data.get('min_points', 6),
            max_points=data.get('max_points', 15),
            max_angular_step=data.get('max_angular_step', 75),
        )

import math
from typing import Optional, List, Tuple, Dict, Any
from ...core.ops import Ops, LineToCommand, MoveToCommand
from ...shared.tasker.proxy import BaseExecutionContext
from .base import OpsTransformer


class Smooth(OpsTransformer):
    """Smooths path segments using a Gaussian filter.

    This transformer preserves sharp corners by implementing a "divide and
    conquer" strategy. It first identifies all anchor points (sharp corners
    and endpoints), then breaks the path into sub-segments between these
    anchors. Each sub-segment is smoothed independently, ensuring that
    straight lines are not affected by distant corners.
    """

    def __init__(
        self, enabled: bool = True, amount=20, corner_angle_threshold=45
    ):
        """
        Initializes the smoothing filter.

        Args:
            enabled: Whether the transformer is active.
            amount: The smoothing strength, from 0 (none) to 100 (heavy).
            corner_angle_threshold: Angles (in degrees) sharper than this
                                    are preserved as corners.
        """
        super().__init__(enabled=enabled)
        self._corner_threshold_rad = math.radians(corner_angle_threshold)
        self._kernel: Optional[List[float]] = None
        self._amount = (
            -1
        )  # Initialize to a value that guarantees the setter runs
        self.amount = amount  # Triggers the property setter and kernel setup

    @property
    def amount(self) -> int:
        """The smoothing strength, from 0 (none) to 100 (heavy)."""
        return self._amount

    @amount.setter
    def amount(self, value: int) -> None:
        new_amount = max(0, min(100, value))
        if self._amount == new_amount:
            return
        self._amount = new_amount
        self._precompute_kernel()
        self.changed.send(self)

    @property
    def corner_angle_threshold(self) -> float:
        """The corner angle threshold in degrees."""
        return math.degrees(self._corner_threshold_rad)

    @corner_angle_threshold.setter
    def corner_angle_threshold(self, value_deg: float):
        """Sets the corner angle threshold from a value in degrees."""
        new_value_rad = math.radians(value_deg)
        if math.isclose(self._corner_threshold_rad, new_value_rad):
            return
        self._corner_threshold_rad = new_value_rad
        self.changed.send(self)

    @property
    def corner_threshold(self) -> float:
        """The corner angle threshold in radians, for internal use."""
        return self._corner_threshold_rad

    def _precompute_kernel(self):
        """
        Pre-calculates the Gaussian kernel for efficient reuse.

        The kernel is derived from the `amount` property, which corresponds
        to the filter's standard deviation (sigma).
        """
        if self._amount == 0:
            self._kernel = [1.0]
            return

        # Map amount (0-100) to sigma. This is a tunable mapping.
        sigma = (self._amount / 100.0) * 5.0
        if sigma < 0.1:  # Effectively zero
            self._kernel = [1.0]
            return

        # Determine kernel size. Rule of thumb: 3*sigma on each side.
        radius = math.ceil(sigma * 3)
        size = 2 * radius + 1
        kernel = [0.0] * size
        kernel_sum = 0.0

        for i in range(size):
            x = i - radius
            val = math.exp(-0.5 * (x / sigma) ** 2)
            kernel[i] = val
            kernel_sum += val

        # Normalize the kernel so that its weights sum to 1.0.
        self._kernel = [k / kernel_sum for k in kernel]

    @property
    def label(self) -> str:
        return _("Smooth Path")

    @property
    def description(self) -> str:
        return _("Smooths the path by applying a Gaussian filter")

    def run(
        self,
        ops: Ops,
        context: Optional[BaseExecutionContext] = None,
    ):
        """Executes the smoothing transformation on the Ops object."""
        segments = list(ops.segments())
        ops.clear()

        total_segments = len(segments)
        for i, segment in enumerate(segments):
            if context and context.is_cancelled():
                return

            if self._is_line_only_segment(segment):
                points = [cmd.end for cmd in segment]
                smoothed = self._smooth_segment(points)
                if smoothed:
                    ops.move_to(*smoothed[0])
                    for point in smoothed[1:]:
                        ops.line_to(*point)
            else:
                for command in segment:
                    ops.add(command)

            if context and total_segments > 0:
                context.set_progress((i + 1) / total_segments)

    def _is_line_only_segment(self, segment: List) -> bool:
        """Checks if a segment is composed entirely of straight lines."""
        return (
            len(segment) > 1
            and isinstance(segment[0], MoveToCommand)
            and all(isinstance(cmd, LineToCommand) for cmd in segment[1:])
        )

    def _smooth_sub_segment(
        self, sub_points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Applies Gaussian smoothing to an isolated list of points.

        This method assumes the start and end points of `sub_points` are
        fixed anchors and only smooths the points between them.

        Args:
            sub_points: A list of (x, y) tuples representing a path.

        Returns:
            A new list of smoothed (x, y) tuples.
        """
        num_sub_points = len(sub_points)
        if num_sub_points < 3:
            return sub_points

        kernel_radius = (len(self._kernel) - 1) // 2
        smoothed_sub_points = []

        for i in range(num_sub_points):
            # Preserve the start and end points of the sub-segment.
            if i == 0 or i == num_sub_points - 1:
                smoothed_sub_points.append(sub_points[i])
                continue

            # Apply the filter, reading from the original sub_points.
            new_x, new_y, total_weight = 0.0, 0.0, 0.0
            for k_idx, k_weight in enumerate(self._kernel):
                p_idx = i - kernel_radius + k_idx
                if 0 <= p_idx < num_sub_points:
                    point = sub_points[p_idx]
                    new_x += point[0] * k_weight
                    new_y += point[1] * k_weight
                    total_weight += k_weight

            if total_weight > 0:
                smoothed_sub_points.append(
                    (new_x / total_weight, new_y / total_weight)
                )
            else:
                smoothed_sub_points.append(sub_points[i])

        return smoothed_sub_points

    def _smooth_segment(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Smooths a list of points using the divide-and-conquer method.

        Args:
            points: The original list of (x, y) tuples.

        Returns:
            A new list of smoothed (x, y) tuples.
        """
        num_points = len(points)
        if not self._kernel or len(self._kernel) <= 1 or num_points < 3:
            return points

        # Stage 1: Find all anchor points (corners and endpoints).
        is_anchor = [False] * num_points
        is_anchor[0] = True
        is_anchor[num_points - 1] = True
        for i in range(1, num_points - 1):
            angle = self._angle_between(
                points[i - 1], points[i], points[i + 1]
            )
            if angle > self.corner_threshold:
                is_anchor[i] = True

        # Stage 2: Divide, smooth each sub-segment, and reassemble.
        final_points = []
        last_anchor_idx = 0
        for i in range(1, num_points):
            if is_anchor[i]:
                # A sub-segment is from the last anchor to this one.
                sub_segment = points[last_anchor_idx:i + 1]
                smoothed_sub = self._smooth_sub_segment(sub_segment)
                # Append the smoothed segment, excluding its last point,
                # which is the start of the next segment.
                final_points.extend(smoothed_sub[:-1])
                last_anchor_idx = i

        # Add the final anchor point of the entire path.
        final_points.append(points[-1])

        return final_points

    def _angle_between(self, p0, p1, p2) -> float:
        """Calculates the angle of the turn at point p1.

        Returns:
            The angle in radians, where 0 is a straight line.
        """
        v1x, v1y = p1[0] - p0[0], p1[1] - p0[1]
        v2x, v2y = p2[0] - p1[0], p2[1] - p1[1]
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)

        if mag1 == 0 or mag2 == 0:
            return 0.0  # Duplicate points form a straight line.

        dot = v1x * v2x + v1y * v2y
        # Clamp to handle potential floating point inaccuracies.
        cos_theta = min(1.0, max(-1.0, dot / (mag1 * mag2)))

        return math.acos(cos_theta)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the transformer's configuration to a dictionary."""
        data = super().to_dict()
        data.update(
            {
                "amount": self.amount,
                "corner_angle_threshold": self.corner_angle_threshold,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Smooth":
        """Creates a Smooth instance from a dictionary."""
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(
            enabled=data.get("enabled", True),
            amount=data.get("amount", 20),
            corner_angle_threshold=data.get("corner_angle_threshold", 45),
        )

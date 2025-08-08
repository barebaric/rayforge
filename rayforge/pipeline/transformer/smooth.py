import math
from typing import Optional, List, Tuple, Dict, Any
from ...core.ops import Ops, LineToCommand, MoveToCommand
from ...shared.tasker.proxy import BaseExecutionContext
from .base import OpsTransformer


class Smooth(OpsTransformer):
    """Smooths path segments using a Gaussian filter.

    This transformer uses a multi-stage "divide and conquer" algorithm:

    1.  **Dynamic Subdivision:** It resamples the path into a high-density
        set of points. The density is proportional to the smoothing
        'amount', ensuring perfect curves even for tiny radii.
    2.  **Anchor Detection:** It identifies all "anchor" points—endpoints
        and sharp corners—that must be preserved.
    3.  **Split & Smooth:** The path is split into independent sub-segments
        between these anchors. Each sub-segment is smoothed in isolation,
        preventing smoothing from "bleeding" across sharp corners.
    4.  **Reassembly:** The smoothed sub-segments are reassembled into the
        final, high-quality path.
    """

    def __init__(
        self, enabled: bool = True, amount=20, corner_angle_threshold=45
    ):
        """Initializes the smoothing filter.

        Args:
            enabled: Whether the transformer is active.
            amount: The smoothing strength (0-100) controlling the curve
                    radius.
            corner_angle_threshold: Corners with an internal angle (in
                                    degrees) smaller than this are
                                    preserved.
        """
        super().__init__(enabled=enabled)
        self._corner_threshold_rad = math.radians(corner_angle_threshold)
        self._kernel: Optional[List[float]] = None
        self._sigma: float = 0.1
        self._amount = -1
        self.amount = amount

    @property
    def amount(self) -> int:
        """The smoothing strength, from 0 (none) to 100 (heavy)."""
        return self._amount

    @amount.setter
    def amount(self, value: int) -> None:
        """Updates the smoothing amount and pre-computes the kernel."""
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
        """Pre-calculates the Gaussian kernel based on the amount."""
        if self._amount == 0:
            self._kernel = [1.0]
            self._sigma = 0.0
            return

        # Map amount (0-100) to a sigma value for the Gaussian function.
        self._sigma = (self._amount / 100.0) * 5.0 + 0.1

        # Kernel radius is typically 3 standard deviations (~99.7% of area).
        radius = math.ceil(self._sigma * 3)
        size = 2 * radius + 1
        kernel = [0.0] * size
        kernel_sum = 0.0

        # Calculate Gaussian function values for the kernel.
        for i in range(size):
            x = i - radius
            val = math.exp(-0.5 * (x / self._sigma) ** 2)
            kernel[i] = val
            kernel_sum += val

        # Normalize the kernel so that its values sum to 1.0.
        self._kernel = [k / kernel_sum for k in kernel]

    @property
    def label(self) -> str:
        return _("Smooth Path")

    @property
    def description(self) -> str:
        return _("Smooths the path by applying a Gaussian filter")

    def run(self, ops: Ops, context: Optional[BaseExecutionContext] = None):
        """
        Executes the smoothing transformation on a set of operations.

        Args:
            ops: The operations object containing path data.
            context: The execution context for cancellation and progress.
        """
        if self.amount == 0:
            return

        segments = list(ops.segments())
        ops.clear()
        total_segments = len(segments)

        for i, segment in enumerate(segments):
            if context and context.is_cancelled():
                return

            points_to_smooth: Optional[List[Tuple[float, float, float]]] = None
            if self._is_line_only_segment(segment):
                # Extract points. The `end` property may be typed as Optional.
                points_to_smooth = [cmd.end for cmd in segment
                                    if cmd.end is not None]
                smoothed = self._smooth_segment(points_to_smooth)
                if smoothed:
                    ops.move_to(*smoothed[0])
                    for point in smoothed[1:]:
                        ops.line_to(*point)
            else:
                # For complex segments (e.g., with curves) or malformed ones,
                # add them back without modification.
                for command in segment:
                    ops.add(command)

            if context and total_segments > 0:
                context.set_progress((i + 1) / total_segments)

    def _is_line_only_segment(self, segment: List) -> bool:
        """Checks if a segment contains only MoveTo and LineTo commands."""
        return (
            len(segment) > 1
            and isinstance(segment[0], MoveToCommand)
            and all(isinstance(c, LineToCommand) for c in segment[1:])
        )

    def _is_closed(
        self, points: List[Tuple[float, float, float]], tol=1e-6
    ) -> bool:
        """Checks if a path is closed by comparing start and end points."""
        return len(points) >= 3 and self._distance(points[0], points[-1]) < tol

    def _distance(
        self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]
    ) -> float:
        """Calculates the 2D Euclidean distance between two points."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _subdivide_path(
        self, points: List[Tuple[float, float, float]], is_closed: bool
    ) -> List[Tuple[float, float, float]]:
        """Resamples a path, adding points to increase density.

        The density is proportional to the smoothing sigma, ensuring that
        the path has enough detail to form smooth curves.
        """
        if not points:
            return []

        # Determine the maximum length of a sub-segment.
        max_len = max(0.1, self._sigma / 4.0)
        new_points = [points[0]]
        num_segments = len(points) if is_closed else len(points) - 1

        for i in range(num_segments):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]  # Wraps for closed paths
            dist = self._distance(p1, p2)

            if dist > max_len:
                # If a segment is too long, subdivide it.
                num_sub = math.ceil(dist / max_len)
                for j in range(1, int(num_sub)):
                    t = j / num_sub
                    # Linear interpolation to create new points.
                    px = p1[0] * (1 - t) + p2[0] * t
                    py = p1[1] * (1 - t) + p2[1] * t
                    # Z is constant at this stage, so we can just take p1's Z
                    new_points.append((px, py, p1[2]))

            # Add the original endpoint, avoiding duplication for closed paths.
            if not (is_closed and i == num_segments - 1):
                new_points.append(p2)

        return new_points

    def _smooth_sub_segment(
        self, sub_points: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """Applies the Gaussian kernel to a single, open list of points.

        The endpoints of the sub-segment are preserved. Z is passed through.
        """
        assert self._kernel is not None, "Kernel must be pre-computed."
        num_pts = len(sub_points)
        if num_pts < 3:
            return sub_points

        kernel_radius = (len(self._kernel) - 1) // 2
        smoothed = [sub_points[0]]  # Preserve the start point.

        # Apply 1D convolution to all interior points.
        for i in range(1, num_pts - 1):
            new_x, new_y = 0.0, 0.0
            for k_idx, k_weight in enumerate(self._kernel):
                # Find the corresponding point, clamping to boundaries.
                p_idx = max(0, min(num_pts - 1, i - kernel_radius + k_idx))
                point = sub_points[p_idx]
                new_x += point[0] * k_weight
                new_y += point[1] * k_weight
            # Z is preserved from the original point at this index
            smoothed.append((new_x, new_y, sub_points[i][2]))

        smoothed.append(sub_points[-1])  # Preserve the end point.
        return smoothed

    def _smooth_segment(
        self, points: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """Orchestrates the full smoothing process for one segment."""
        # A kernel of length 1 means no smoothing.
        if self._kernel is None or len(self._kernel) <= 1 or len(points) < 3:
            return points

        is_closed = self._is_closed(points)
        work_points = points[:-1] if is_closed else points
        prepared_points = self._subdivide_path(work_points, is_closed)
        num_points = len(prepared_points)

        if num_points < 3:
            return points

        # Find all anchor points (corners and endpoints) to preserve.
        anchor_indices = set()
        if not is_closed:
            anchor_indices.update([0, num_points - 1])

        for i in range(num_points):
            p_prev = prepared_points[(i - 1 + num_points) % num_points]
            p_curr = prepared_points[i]
            p_next = prepared_points[(i + 1) % num_points]
            angle = self._angle_between(p_prev, p_curr, p_next)

            # Preserve a corner if its angle is smaller than the threshold.
            # We use `math.isclose` to avoid floating-point errors where an
            # angle is calculated as being infinitesimally smaller than the
            # threshold, causing it to be preserved incorrectly.
            if angle < self.corner_threshold and not math.isclose(
                angle, self.corner_threshold
            ):
                anchor_indices.add(i)

        sorted_anchors = sorted(list(anchor_indices))
        final_points = []

        # --- Path Reassembly ---
        if is_closed:
            if not sorted_anchors:
                # A closed loop with no sharp corners is smoothed circularly.
                return self._smooth_circularly(prepared_points)

            # Split path at anchors, smooth each sub-segment, and stitch.
            num_anchors = len(sorted_anchors)
            for i in range(num_anchors):
                start_idx = sorted_anchors[i]
                end_idx = sorted_anchors[(i + 1) % num_anchors]
                if start_idx < end_idx:
                    sub_seg = prepared_points[start_idx:end_idx + 1]
                else:  # Handle wraparound segment
                    sub_seg = (
                        prepared_points[start_idx:]
                        + prepared_points[: end_idx + 1]
                    )

                smoothed_sub = self._smooth_sub_segment(sub_seg)
                final_points.extend(smoothed_sub[:-1])

            if final_points:  # Close the final path
                final_points.append(final_points[0])
            return final_points
        else:  # Open path
            if len(sorted_anchors) < 2:
                # If there are no internal anchors, smooth the whole path.
                return self._smooth_sub_segment(prepared_points)

            # Split path at anchors, smooth each sub-segment, and stitch.
            last_anchor_idx = sorted_anchors[0]
            for i in range(1, len(sorted_anchors)):
                anchor_idx = sorted_anchors[i]
                sub_seg = prepared_points[last_anchor_idx:anchor_idx + 1]
                smoothed_sub = self._smooth_sub_segment(sub_seg)
                final_points.extend(smoothed_sub[:-1])
                last_anchor_idx = anchor_idx

            final_points.append(prepared_points[-1])  # Add final anchor.
            return final_points

    def _smooth_circularly(
        self, points: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """Applies a wrapping Gaussian filter to a closed loop."""
        assert self._kernel is not None, "Kernel must be pre-computed."
        num_pts = len(points)
        kernel_radius = (len(self._kernel) - 1) // 2
        smoothed = []

        # Apply convolution with circular (wrapping) boundary conditions.
        for i in range(num_pts):
            new_x, new_y = 0.0, 0.0
            for k_idx, k_weight in enumerate(self._kernel):
                # Modulo arithmetic wraps the index around the list.
                p_idx = (i - kernel_radius + k_idx + num_pts) % num_pts
                point = points[p_idx]
                new_x += point[0] * k_weight
                new_y += point[1] * k_weight
            # Z is preserved from the original point
            smoothed.append((new_x, new_y, points[i][2]))

        if smoothed:
            smoothed.append(smoothed[0])  # Close the path.
        return smoothed

    def _angle_between(
        self,
        p0: Tuple[float, float, float],
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
    ) -> float:
        """
        Calculates the internal angle of the corner at point p1 in the
        XY plane.
        """
        # Create vectors from p1 to p0 and p1 to p2.
        v1x, v1y = p0[0] - p1[0], p0[1] - p1[1]
        v2x, v2y = p2[0] - p1[0], p2[1] - p1[1]

        # Calculate magnitudes for normalization.
        mag_prod = self._distance(p0, p1) * self._distance(p1, p2)
        if mag_prod == 0:
            return math.pi  # Straight line if points are coincident.

        # Dot product and normalization to get cosine of the angle.
        dot = v1x * v2x + v1y * v2y
        cos_theta = min(1.0, max(-1.0, dot / mag_prod))

        # Return angle in radians.
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

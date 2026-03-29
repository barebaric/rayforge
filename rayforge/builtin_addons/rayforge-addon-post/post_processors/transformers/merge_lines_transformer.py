from __future__ import annotations

import math
import copy
from collections import defaultdict
from typing import (
    Optional,
    Dict,
    Any,
    List,
    TYPE_CHECKING,
    cast,
    Sequence,
    Set,
    Tuple,
    DefaultDict,
)
from gettext import gettext as _

from rayforge.pipeline.transformer.base import OpsTransformer, ExecutionPhase
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    Command,
)
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from rayforge.core.geo import Geometry


class LineSegment:
    """Represents a line segment with start and end points."""

    __slots__ = (
        "start",
        "end",
        "command_index",
        "segment_index",
        "covered_intervals",
        "dx",
        "dy",
        "length_sq",
    )

    def __init__(
        self,
        start: Tuple[float, ...],
        end: Tuple[float, ...],
        segment_index: int,
        command_index: int,
    ):
        self.start = start
        self.end = end
        self.segment_index = segment_index
        self.command_index = command_index

        # Will hold the parametric ranges [t1, t2] that are covered by other
        # lines
        self.covered_intervals: List[Tuple[float, float]] = []

        self.dx = end[0] - start[0]
        self.dy = end[1] - start[1]
        self.length_sq = self.dx * self.dx + self.dy * self.dy

    def length(self) -> float:
        return math.sqrt(self.length_sq)

    def direction(self) -> Tuple[float, float]:
        length = self.length()
        if length < 1e-9:
            return (0.0, 0.0)
        return (self.dx / length, self.dy / length)


class MergeLinesTransformer(OpsTransformer):
    """
    Merges overlapping/collinear line segments across all paths.

    This transformer detects line segments that are collinear and overlapping
    (typically from adjacent workpieces sharing an edge) and replaces the
    covered sub-segments with travel moves to avoid cutting the same line
    twice.

    The transformer should run before optimization and MultiPassTransformer.
    """

    DEFAULT_TOLERANCE = 0.1

    def __init__(
        self, enabled: bool = True, tolerance: float = DEFAULT_TOLERANCE
    ):
        super().__init__(enabled=enabled)
        self._tolerance = tolerance

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        self._tolerance = max(0.001, value)
        self.changed.send(self)

    @property
    def execution_phase(self) -> ExecutionPhase:
        return ExecutionPhase.POST_PROCESSING

    @property
    def label(self) -> str:
        return _("Merge Lines")

    @property
    def description(self) -> str:
        return _("Merges overlapping lines to avoid double passing.")

    def _z_overlap(self, seg1: LineSegment, seg2: LineSegment) -> bool:
        """
        Ensures segments can only merge if they exist on the same Z plane.
        """
        z1_start = seg1.start[2] if len(seg1.start) > 2 else 0.0
        z1_end = seg1.end[2] if len(seg1.end) > 2 else 0.0
        z2_start = seg2.start[2] if len(seg2.start) > 2 else 0.0
        z2_end = seg2.end[2] if len(seg2.end) > 2 else 0.0

        min_z1 = min(z1_start, z1_end)
        max_z1 = max(z1_start, z1_end)
        min_z2 = min(z2_start, z2_end)
        max_z2 = max(z2_start, z2_end)

        return (min_z1 <= max_z2 + self._tolerance) and (
            min_z2 <= max_z1 + self._tolerance
        )

    def _are_parallel(
        self,
        seg1: LineSegment,
        seg2: LineSegment,
    ) -> bool:
        d1 = seg1.direction()
        d2 = seg2.direction()
        dot = abs(d1[0] * d2[0] + d1[1] * d2[1])
        return dot > 0.9999

    def _point_line_distance(
        self, point: Tuple[float, ...], seg: LineSegment
    ) -> float:
        x0, y0 = point[0], point[1]
        x1, y1 = seg.start[0], seg.start[1]
        x2, y2 = seg.end[0], seg.end[1]
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            return math.hypot(x0 - x1, y0 - y1)
        num = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1)
        return num / math.sqrt(length_sq)

    def _are_collinear(
        self,
        seg1: LineSegment,
        seg2: LineSegment,
    ) -> bool:
        if not self._are_parallel(seg1, seg2):
            return False
        dist1 = self._point_line_distance(seg2.start, seg1)
        dist2 = self._point_line_distance(seg2.end, seg1)
        return dist1 <= self._tolerance and dist2 <= self._tolerance

    def _is_line_segment(self, segment: Sequence[Command]) -> bool:
        return (
            len(segment) > 1
            and isinstance(segment[0], MoveToCommand)
            and all(isinstance(c, LineToCommand) for c in segment[1:])
        )

    def _get_cell_keys(
        self, seg: LineSegment, cell_size: float
    ) -> Set[Tuple[int, int]]:
        min_x = min(seg.start[0], seg.end[0])
        max_x = max(seg.start[0], seg.end[0])
        min_y = min(seg.start[1], seg.end[1])
        max_y = max(seg.start[1], seg.end[1])
        cx1 = int(math.floor((min_x - self._tolerance) / cell_size))
        cx2 = int(math.floor((max_x + self._tolerance) / cell_size))
        cy1 = int(math.floor((min_y - self._tolerance) / cell_size))
        cy2 = int(math.floor((max_y + self._tolerance) / cell_size))
        return {
            (cx, cy)
            for cx in range(cx1, cx2 + 1)
            for cy in range(cy1, cy2 + 1)
        }

    def _extract_line_segments(
        self, segments: List[List[Command]]
    ) -> List[LineSegment]:
        line_segments: List[LineSegment] = []

        for seg_idx, segment in enumerate(segments):
            if not self._is_line_segment(segment):
                continue

            move_cmd = cast(MoveToCommand, segment[0])
            if move_cmd.end is None:
                continue

            current_pos = move_cmd.end

            for cmd_idx, cmd in enumerate(segment[1:], start=1):
                line_cmd = cast(LineToCommand, cmd)
                if line_cmd.end is None:
                    continue

                end_pos = line_cmd.end
                line_seg = LineSegment(current_pos, end_pos, seg_idx, cmd_idx)
                line_segments.append(line_seg)
                current_pos = end_pos

        return line_segments

    def _find_duplicates(self, line_segments: List[LineSegment]) -> None:
        """Finds overlapping segments and computes covered 1D intervals."""
        if not line_segments:
            return

        cell_size = max(self._tolerance * 10, 1.0)
        checked_pairs: Set[Tuple[int, int]] = set()
        index: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)

        for i, seg in enumerate(line_segments):
            for cell_key in self._get_cell_keys(seg, cell_size):
                index[(cell_key[0], cell_key[1])].append(i)

        for cell_key, indices in index.items():
            if len(indices) < 2:
                continue

            for i, idx1 in enumerate(indices):
                seg1 = line_segments[idx1]

                for idx2 in indices[i + 1 :]:
                    seg2 = line_segments[idx2]

                    pair = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    if not self._z_overlap(seg1, seg2):
                        continue

                    if self._are_collinear(seg1, seg2):
                        len1 = seg1.length()
                        len2 = seg2.length()

                        # Decide which line takes precedence (longer line
                        # covers shorter)
                        if len1 > len2 + 1e-5:
                            coverer, coveree = seg1, seg2
                        elif len2 > len1 + 1e-5:
                            coverer, coveree = seg2, seg1
                        else:
                            # Tie breaker: earlier index wins
                            if (seg1.segment_index, seg1.command_index) < (
                                seg2.segment_index,
                                seg2.command_index,
                            ):
                                coverer, coveree = seg1, seg2
                            else:
                                coverer, coveree = seg2, seg1

                        # Calculate projection parameters onto coveree's line
                        P1 = coveree.start
                        dx, dy = coveree.dx, coveree.dy
                        l_sq = coveree.length_sq

                        if l_sq < 1e-12:
                            continue

                        C, D = coverer.start, coverer.end

                        t_C = (
                            (C[0] - P1[0]) * dx + (C[1] - P1[1]) * dy
                        ) / l_sq
                        t_D = (
                            (D[0] - P1[0]) * dx + (D[1] - P1[1]) * dy
                        ) / l_sq

                        t_min_raw = min(t_C, t_D)
                        t_max_raw = max(t_C, t_D)

                        # Apply horizontal tolerance: expand the covered
                        # interval to seamlessly bridge gaps (like the joints
                        # in a brick layout).
                        L = coveree.length()
                        t_tol = self._tolerance / L

                        t_min = max(0.0, t_min_raw - t_tol)
                        t_max = min(1.0, t_max_raw + t_tol)

                        if t_min < t_max - 1e-6:
                            coveree.covered_intervals.append((t_min, t_max))

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[Sequence["Geometry"]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        if ops.is_empty():
            return

        ops.preload_state()

        segments = list(ops.segments())
        line_segments = self._extract_line_segments(segments)

        if not line_segments:
            return

        self._find_duplicates(line_segments)

        segment_map = {}
        has_covered = False
        for seg in line_segments:
            segment_map[(seg.segment_index, seg.command_index)] = seg
            if seg.covered_intervals:
                has_covered = True

        if not has_covered:
            return

        def get_uncovered(
            intervals: List[Tuple[float, float]],
        ) -> List[Tuple[float, float]]:
            if not intervals:
                return [(0.0, 1.0)]
            intervals.sort(key=lambda x: x[0])
            merged = [intervals[0]]
            for current in intervals[1:]:
                last = merged[-1]
                if current[0] <= last[1] + 1e-6:
                    merged[-1] = (last[0], max(last[1], current[1]))
                else:
                    merged.append(current)

            uncovered = []
            current_t = 0.0
            for start, end in merged:
                if start > current_t + 1e-6:
                    uncovered.append((current_t, start))
                current_t = max(current_t, end)
            if current_t < 1.0 - 1e-6:
                uncovered.append((current_t, 1.0))
            return uncovered

        def interpolate(p1, p2, t):
            z1 = p1[2] if len(p1) > 2 else 0.0
            z2 = p2[2] if len(p2) > 2 else 0.0
            return (
                p1[0] + (p2[0] - p1[0]) * t,
                p1[1] + (p2[1] - p1[1]) * t,
                z1 + (z2 - z1) * t,
            )

        def dist3d(p1, p2):
            if p1 is None or p2 is None:
                return float("inf")
            z1 = p1[2] if len(p1) > 2 else 0.0
            z2 = p2[2] if len(p2) > 2 else 0.0
            return math.sqrt(
                (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (z2 - z1) ** 2
            )

        ops.clear()

        machine_pos = None

        for seg_idx, segment in enumerate(segments):
            expected_pos = None

            for cmd_idx, cmd in enumerate(segment):
                line_seg = segment_map.get((seg_idx, cmd_idx))
                has_end = hasattr(cmd, "end") and cmd.end is not None

                if line_seg:
                    if line_seg.covered_intervals:
                        uncovered = get_uncovered(line_seg.covered_intervals)
                        P1 = line_seg.start
                        P2 = line_seg.end
                        seg_L = line_seg.length()

                        # Drop tiny micro-segments that are smaller than
                        # tolerance
                        filtered_uncovered = []
                        for u, v in uncovered:
                            if (v - u) * seg_L > self._tolerance * 0.5:
                                filtered_uncovered.append((u, v))

                        for u, v in filtered_uncovered:
                            start_pt = interpolate(P1, P2, u)
                            end_pt = interpolate(P1, P2, v)

                            if dist3d(machine_pos, start_pt) > 1e-5:
                                new_move = MoveToCommand(start_pt)
                                if hasattr(cmd, "state") and cmd.state:
                                    new_move.state = copy.copy(cmd.state)
                                ops.add(new_move)

                            new_cut = LineToCommand(end_pt)
                            if hasattr(cmd, "state") and cmd.state:
                                new_cut.state = copy.copy(cmd.state)
                            ops.add(new_cut)
                            machine_pos = end_pt
                    else:
                        P1 = line_seg.start
                        if dist3d(machine_pos, P1) > 1e-5:
                            new_move = MoveToCommand(P1)
                            if hasattr(cmd, "state") and cmd.state:
                                new_move.state = copy.copy(cmd.state)
                            ops.add(new_move)

                        ops.add(cmd)
                        machine_pos = cmd.end

                    expected_pos = cmd.end
                else:
                    is_cut = getattr(
                        cmd, "is_cutting_command", lambda: False
                    )()

                    # If it's a non-LineTo cut (like an arc) and we are out
                    # of sync due to skipping prior lines
                    if is_cut and dist3d(machine_pos, expected_pos) > 1e-5:
                        new_move = MoveToCommand(expected_pos)
                        if hasattr(cmd, "state") and cmd.state:
                            new_move.state = copy.copy(cmd.state)
                        ops.add(new_move)
                        machine_pos = expected_pos

                    ops.add(cmd)

                    if has_end:
                        expected_pos = cmd.end
                        if (
                            getattr(cmd, "is_travel_command", lambda: False)()
                            or is_cut
                        ):
                            machine_pos = cmd.end

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["tolerance"] = self._tolerance
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergeLinesTransformer":
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(
            enabled=data.get("enabled", True),
            tolerance=data.get("tolerance", cls.DEFAULT_TOLERANCE),
        )

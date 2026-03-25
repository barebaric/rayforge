from __future__ import annotations

import math
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

    __slots__ = ("start", "end", "command_index", "segment_index", "removed")

    def __init__(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        segment_index: int,
        command_index: int,
    ):
        self.start = start
        self.end = end
        self.segment_index = segment_index
        self.command_index = command_index
        self.removed = False

    def length(self) -> float:
        return math.hypot(
            self.end[0] - self.start[0], self.end[1] - self.start[1]
        )

    def direction(self) -> Tuple[float, float]:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return (0.0, 0.0)
        return (dx / length, dy / length)


class MergeLinesTransformer(OpsTransformer):
    """
    Merges overlapping/collinear line segments across all paths.

    This transformer detects line segments that are collinear and overlapping
    (typically from adjacent workpieces sharing an edge) and removes duplicates
    to avoid cutting the same line twice.

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

    def _points_equal(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> bool:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1]) <= self._tolerance

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
        self, point: Tuple[float, float], seg: LineSegment
    ) -> float:
        x0, y0 = point
        x1, y1 = seg.start
        x2, y2 = seg.end
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

    def _segments_overlap(
        self,
        seg1: LineSegment,
        seg2: LineSegment,
    ) -> bool:
        if not self._are_collinear(seg1, seg2):
            return False

        dir1 = seg1.direction()
        if abs(dir1[0]) > abs(dir1[1]):
            proj1_start = seg1.start[0]
            proj1_end = seg1.end[0]
            proj2_start = seg2.start[0]
            proj2_end = seg2.end[0]
        else:
            proj1_start = seg1.start[1]
            proj1_end = seg1.end[1]
            proj2_start = seg2.start[1]
            proj2_end = seg2.end[1]

        if proj1_start > proj1_end:
            proj1_start, proj1_end = proj1_end, proj1_start
        if proj2_start > proj2_end:
            proj2_start, proj2_end = proj2_end, proj2_start

        overlap_start = max(proj1_start, proj2_start)
        overlap_end = min(proj1_end, proj2_end)

        return overlap_end - overlap_start > self._tolerance

    def _are_identical(self, seg1: LineSegment, seg2: LineSegment) -> bool:
        same_direction = self._points_equal(
            seg1.start, seg2.start
        ) and self._points_equal(seg1.end, seg2.end)
        opposite_direction = self._points_equal(
            seg1.start, seg2.end
        ) and self._points_equal(seg1.end, seg2.start)
        return same_direction or opposite_direction

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

            current_pos = (move_cmd.end[0], move_cmd.end[1])

            for cmd_idx, cmd in enumerate(segment[1:], start=1):
                line_cmd = cast(LineToCommand, cmd)
                if line_cmd.end is None:
                    continue

                end_pos = (line_cmd.end[0], line_cmd.end[1])
                line_seg = LineSegment(current_pos, end_pos, seg_idx, cmd_idx)
                line_segments.append(line_seg)
                current_pos = end_pos

        return line_segments

    def _find_duplicates(self, line_segments: List[LineSegment]) -> Set[int]:
        if not line_segments:
            return set()

        cell_size = max(self._tolerance * 10, 1.0)
        to_remove: Set[int] = set()
        checked_pairs: Set[Tuple[int, int]] = set()

        index: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)

        for i, seg in enumerate(line_segments):
            for cell_key in self._get_cell_keys(seg, cell_size):
                index[(cell_key[0], cell_key[1])].append(i)

        for cell_key, indices in index.items():
            if len(indices) < 2:
                continue

            for i, idx1 in enumerate(indices):
                if idx1 in to_remove:
                    continue
                seg1 = line_segments[idx1]
                if seg1.removed:
                    continue

                for idx2 in indices[i + 1 :]:
                    if idx2 in to_remove:
                        continue
                    seg2 = line_segments[idx2]
                    if seg2.removed:
                        continue

                    pair = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    if self._are_identical(seg1, seg2):
                        to_remove.add(idx2)
                        seg2.removed = True
                    elif self._segments_overlap(seg1, seg2):
                        if seg1.length() >= seg2.length():
                            to_remove.add(idx2)
                            seg2.removed = True
                        else:
                            to_remove.add(idx1)
                            seg1.removed = True
                            break

        return to_remove

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

        to_remove = self._find_duplicates(line_segments)

        if not to_remove:
            return

        removed_by_segment: Dict[int, Set[int]] = {}
        removed_endpoints: Dict[int, Dict[int, Tuple[float, float]]] = {}
        for idx in to_remove:
            seg = line_segments[idx]
            if seg.segment_index not in removed_by_segment:
                removed_by_segment[seg.segment_index] = set()
                removed_endpoints[seg.segment_index] = {}
            removed_by_segment[seg.segment_index].add(seg.command_index)
            removed_endpoints[seg.segment_index][seg.command_index] = seg.end

        ops.clear()

        for seg_idx, segment in enumerate(segments):
            if seg_idx not in removed_by_segment:
                for cmd in segment:
                    ops.add(cmd)
            else:
                removed_commands = removed_by_segment[seg_idx]
                endpoints = removed_endpoints[seg_idx]
                prev_cmd_idx = -1

                for cmd_idx, cmd in enumerate(segment):
                    if cmd_idx == 0:
                        ops.add(cmd)
                        prev_cmd_idx = 0
                    elif cmd_idx in removed_commands:
                        end_pt = endpoints[cmd_idx]
                        new_move = MoveToCommand((end_pt[0], end_pt[1], 0.0))
                        if hasattr(cmd, "state") and cmd.state:
                            new_move.state = cmd.state
                        ops.add(new_move)
                        prev_cmd_idx = cmd_idx
                    else:
                        if cmd_idx != prev_cmd_idx + 1 and prev_cmd_idx >= 0:
                            pass
                        ops.add(cmd)
                        prev_cmd_idx = cmd_idx

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

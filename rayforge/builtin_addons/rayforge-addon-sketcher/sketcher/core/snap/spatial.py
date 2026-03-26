import bisect
from typing import List, Tuple, Iterator, Optional
from dataclasses import dataclass

from .types import SnapLine


@dataclass
class IndexedLine:
    snap_line: Optional[SnapLine]
    coordinate: float

    def __lt__(self, other: "IndexedLine") -> bool:
        return self.coordinate < other.coordinate


class SnapLineIndex:
    def __init__(self) -> None:
        self._horizontal: List[IndexedLine] = []
        self._vertical: List[IndexedLine] = []
        self._dirty: bool = False

    def clear(self) -> None:
        self._horizontal.clear()
        self._vertical.clear()
        self._dirty = False

    def add(self, snap_line: SnapLine) -> None:
        indexed = IndexedLine(snap_line, snap_line.coordinate)
        if snap_line.is_horizontal:
            bisect.insort(self._horizontal, indexed)
        else:
            bisect.insort(self._vertical, indexed)
        self._dirty = True

    def add_all(self, snap_lines: Iterator[SnapLine]) -> None:
        for sl in snap_lines:
            self.add(sl)

    def query_horizontal(
        self, y: float, threshold: float
    ) -> List[Tuple[SnapLine, float]]:
        results: List[Tuple[SnapLine, float]] = []
        low = y - threshold
        high = y + threshold

        left = bisect.bisect_left(self._horizontal, IndexedLine(None, low))
        right = bisect.bisect_right(self._horizontal, IndexedLine(None, high))

        for i in range(left, right):
            indexed = self._horizontal[i]
            if indexed.snap_line is None:
                continue
            dist = abs(y - indexed.coordinate)
            if dist <= threshold:
                results.append((indexed.snap_line, dist))

        return results

    def query_vertical(
        self, x: float, threshold: float
    ) -> List[Tuple[SnapLine, float]]:
        results: List[Tuple[SnapLine, float]] = []
        low = x - threshold
        high = x + threshold

        left = bisect.bisect_left(self._vertical, IndexedLine(None, low))
        right = bisect.bisect_right(self._vertical, IndexedLine(None, high))

        for i in range(left, right):
            indexed = self._vertical[i]
            if indexed.snap_line is None:
                continue
            dist = abs(x - indexed.coordinate)
            if dist <= threshold:
                results.append((indexed.snap_line, dist))

        return results

    def query(
        self, x: float, y: float, threshold: float
    ) -> List[Tuple[SnapLine, float]]:
        results: List[Tuple[SnapLine, float]] = []
        results.extend(self.query_horizontal(y, threshold))
        results.extend(self.query_vertical(x, threshold))
        results.sort(key=lambda t: (t[1], -t[0].line_type.priority))
        return results

    def __len__(self) -> int:
        return len(self._horizontal) + len(self._vertical)

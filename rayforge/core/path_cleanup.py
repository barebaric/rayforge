"""
Clean-up operations for vector paths.

These functions close, join, de-duplicate and split the contours of a
:class:`~raygeo.geo.Geometry`. They never move existing points: gaps are
bridged with a straight segment between two existing end points, so the
bounding box of the result never grows. All distances are expressed in
the units of the geometry passed in; callers that work on normalized
workpiece geometry should scale it to millimetres first.
"""

import math
from collections.abc import Iterator

from raygeo.geo import Geometry, Move

Point = tuple[float, float]

_CLOSED_EPS = 1e-6
_DUPLICATE_SAMPLES = 32


def _start_point(contour: Geometry) -> Point:
    first = contour.data[0]
    return first.end[0], first.end[1]


def _end_point(contour: Geometry) -> Point:
    x, y, _z = contour.get_last_point()
    return x, y


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _segment_count(contour: Geometry) -> int:
    return sum(1 for cmd in contour.data if not isinstance(cmd, Move))


def _drawn_contours(geo: Geometry) -> list[Geometry]:
    """Splits into contours, dropping those that draw nothing."""
    if geo.is_empty():
        return []
    return [c for c in geo.split_into_contours() if _segment_count(c) > 0]


def _is_open(contour: Geometry) -> bool:
    return not contour.is_closed(_CLOSED_EPS)


def _close_if_near(contour: Geometry, tolerance: float) -> bool:
    """
    Closes an open contour of at least two segments in place when its
    start and end are within the tolerance. Returns True if it closed it.
    """
    if not _is_open(contour) or _segment_count(contour) < 2:
        return False
    start = _start_point(contour)
    if _distance(start, _end_point(contour)) > tolerance:
        return False
    contour.line_to(start[0], start[1])
    return True


def _concat(contours: list[Geometry]) -> Geometry:
    result = Geometry()
    for contour in contours:
        result.extend(contour)
    return result


def close_open_contours(
    geo: Geometry, tolerance: float
) -> tuple[Geometry, int]:
    """
    Closes every open contour whose start and end point lie within the
    tolerance of each other.

    Args:
        geo: The geometry to process. It is not modified.
        tolerance: The largest gap that is closed.

    Returns:
        A tuple of the new geometry and the number of closed contours.
    """
    contours = [c.copy() for c in _drawn_contours(geo)]
    closed = sum(1 for c in contours if _close_if_near(c, tolerance))
    return _concat(contours), closed


class _EndpointGrid:
    """A spatial hash of open contour end points for neighbour lookups."""

    def __init__(self, cell_size: float):
        self._cell = max(cell_size, 1e-9)
        self._cells: dict[tuple[int, int], set[tuple[int, int]]] = {}
        self._points: dict[tuple[int, int], Point] = {}

    def _key(self, p: Point) -> tuple[int, int]:
        return math.floor(p[0] / self._cell), math.floor(p[1] / self._cell)

    def add(self, index: int, end: int, p: Point):
        self._points[(index, end)] = p
        self._cells.setdefault(self._key(p), set()).add((index, end))

    def remove_contour(self, index: int):
        for end in (0, 1):
            p = self._points.pop((index, end), None)
            if p is not None:
                self._cells[self._key(p)].discard((index, end))

    def _neighbours(self, p: Point) -> Iterator[tuple[int, int]]:
        kx, ky = self._key(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield from self._cells.get((kx + dx, ky + dy), ())

    def nearest(self, p: Point, tolerance: float) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        best_dist = tolerance
        for entry in sorted(self._neighbours(p)):
            d = _distance(p, self._points[entry])
            if d <= best_dist:
                best, best_dist = entry, d
        return best


def _append_contour(chain: Geometry, piece: Geometry):
    """Appends a piece to the end of a chain, bridging any gap."""
    start = _start_point(piece)
    if _distance(_end_point(chain), start) > 0.0:
        chain.line_to(start[0], start[1])
    chain.extend(piece.filter(set(range(1, len(piece.data)))))


def _grow_tail(
    chain: Geometry,
    contours: list[Geometry],
    grid: _EndpointGrid,
    tolerance: float,
    consumed: set[int],
) -> int:
    """
    Keeps appending the nearest open contour to the chain's end, and
    records the appended contours in ``consumed``.
    """
    joins = 0
    while True:
        match = grid.nearest(_end_point(chain), tolerance)
        if match is None:
            return joins
        index, end = match
        grid.remove_contour(index)
        consumed.add(index)
        piece = contours[index].copy()
        if end == 1:
            piece.reverse_contour()
        _append_contour(chain, piece)
        joins += 1


def join_open_contours(
    geo: Geometry, tolerance: float
) -> tuple[Geometry, int]:
    """
    Joins open contours whose end points meet within the tolerance into
    longer contours, reversing contours where needed. A joined contour
    whose own ends then meet within the tolerance is closed. Closed
    contours are left untouched.

    Args:
        geo: The geometry to process. It is not modified.
        tolerance: The largest gap between two end points that is joined.

    Returns:
        A tuple of the new geometry and the number of joins made.
    """
    contours = _drawn_contours(geo)
    grid = _EndpointGrid(tolerance)
    open_indices = [i for i, c in enumerate(contours) if _is_open(c)]
    for i in open_indices:
        grid.add(i, 0, _start_point(contours[i]))
        grid.add(i, 1, _end_point(contours[i]))

    chains: dict[int, Geometry] = {}
    consumed: set[int] = set()
    joins = 0
    for i in open_indices:
        if i in consumed:
            continue
        grid.remove_contour(i)
        chain = contours[i].copy()
        joins += _grow_tail(chain, contours, grid, tolerance, consumed)
        chain.reverse_contour()
        joins += _grow_tail(chain, contours, grid, tolerance, consumed)
        chain.reverse_contour()
        _close_if_near(chain, tolerance)
        chains[i] = chain

    result = Geometry()
    for i, contour in enumerate(contours):
        if i in chains:
            result.extend(chains[i])
        elif i not in consumed:
            result.extend(contour)
    return result, joins


def _sample_points(contour: Geometry) -> list[Point]:
    """The contour's vertices plus evenly spaced points along it."""
    points = [(cmd.end[0], cmd.end[1]) for cmd in contour.data]
    length = contour.distance()
    if length > 0.0:
        step = length / _DUPLICATE_SAMPLES
        distances = [step * k for k in range(_DUPLICATE_SAMPLES + 1)]
        for _idx, _t, p in contour.get_positions_at_distances(distances):
            points.append((p[0], p[1]))
    return points


def _covers(target: Geometry, points: list[Point], tolerance: float) -> bool:
    for x, y in points:
        hit = target.find_closest_point(x, y)
        if hit is None or _distance((x, y), hit[2]) > tolerance:
            return False
    return True


def _rects_match(a: Geometry, b: Geometry, tolerance: float) -> bool:
    return all(abs(va - vb) <= tolerance for va, vb in zip(a.rect(), b.rect()))


def _contours_match(a: Geometry, b: Geometry, tolerance: float) -> bool:
    """
    True if two contours trace the same path within the tolerance,
    regardless of their direction or start point.
    """
    if _is_open(a) != _is_open(b):
        return False
    if not _rects_match(a, b, tolerance):
        return False
    return _covers(b, _sample_points(a), tolerance) and _covers(
        a, _sample_points(b), tolerance
    )


def _length_slack(tolerance: float) -> float:
    return 4.0 * tolerance + 1e-9


def remove_duplicate_contours(
    geo: Geometry, tolerance: float
) -> tuple[Geometry, int]:
    """
    Removes contours that duplicate an earlier contour within the
    tolerance, regardless of direction or start point. The first
    occurrence is kept and the order of the remaining contours is
    preserved.

    Args:
        geo: The geometry to process. It is not modified.
        tolerance: The largest distance between two matching paths.

    Returns:
        A tuple of the new geometry and the number of removed contours.
    """
    contours = _drawn_contours(geo)
    lengths = [c.distance() for c in contours]
    order = sorted(range(len(contours)), key=lambda i: lengths[i])
    duplicates: set[int] = set()
    slack = _length_slack(tolerance)
    for pos, i in enumerate(order):
        if i in duplicates:
            continue
        for j in order[pos + 1 :]:
            if lengths[j] - lengths[i] > slack:
                break
            if j in duplicates:
                continue
            if _contours_match(contours[i], contours[j], tolerance):
                duplicates.add(max(i, j))
                if max(i, j) == i:
                    break
    kept = [c for i, c in enumerate(contours) if i not in duplicates]
    return _concat(kept), len(duplicates)


def geometries_match(a: Geometry, b: Geometry, tolerance: float) -> bool:
    """
    True if both geometries consist of the same contours within the
    tolerance, in any order, direction or start point. Empty geometries
    never match.
    """
    contours_a = _drawn_contours(a)
    contours_b = _drawn_contours(b)
    if not contours_a or len(contours_a) != len(contours_b):
        return False
    unmatched = list(contours_b)
    for contour in contours_a:
        found = next(
            (
                other
                for other in unmatched
                if _contours_match(contour, other, tolerance)
            ),
            None,
        )
        if found is None:
            return False
        unmatched.remove(found)
    return True


def split_contours(geo: Geometry) -> list[Geometry]:
    """
    Splits a geometry into one geometry per contour. Unlike splitting
    into connected components, holes become parts of their own and open
    contours are kept.
    """
    return [c.copy() for c in _drawn_contours(geo)]

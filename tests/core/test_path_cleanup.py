import math

import pytest
from raygeo.geo import Geometry

from rayforge.core.path_cleanup import (
    close_open_contours,
    geometries_match,
    join_open_contours,
    remove_duplicate_contours,
    split_contours,
)


def _polyline(points, close=False) -> Geometry:
    geo = Geometry()
    geo.move_to(*points[0])
    for p in points[1:]:
        geo.line_to(*p)
    if close:
        geo.line_to(*points[0])
    return geo


def _concat(*geos: Geometry) -> Geometry:
    result = Geometry()
    for geo in geos:
        result.extend(geo)
    return result


def _contours(geo: Geometry) -> list[Geometry]:
    return geo.split_into_contours()


def _start(geo: Geometry) -> tuple[float, float]:
    first = next(iter(geo.iter_typed_commands()))
    return first.end[0], first.end[1]


def _end(geo: Geometry) -> tuple[float, float]:
    x, y, _z = geo.get_last_point()
    return x, y


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


class TestCloseOpenContours:
    def test_closes_contour_within_tolerance(self):
        geo = _polyline(SQUARE + [(0, 0.05)])
        result, count = close_open_contours(geo, 0.1)
        assert count == 1
        contours = _contours(result)
        assert len(contours) == 1
        assert contours[0].is_closed()

    def test_leaves_contour_outside_tolerance_open(self):
        geo = _polyline(SQUARE + [(0, 0.5)])
        result, count = close_open_contours(geo, 0.1)
        assert count == 0
        assert not _contours(result)[0].is_closed()

    def test_does_not_move_existing_points(self):
        geo = _polyline(SQUARE + [(0, 0.05)])
        result, _count = close_open_contours(geo, 0.1)
        assert result.rect() == pytest.approx(geo.rect())

    def test_already_closed_contour_is_untouched(self):
        geo = _polyline(SQUARE, close=True)
        result, count = close_open_contours(geo, 0.1)
        assert count == 0
        assert len(result.data) == len(geo.data)

    def test_handles_each_contour_separately(self):
        geo = _concat(
            _polyline(SQUARE + [(0, 0.05)]),
            _polyline([(20, 0), (30, 0), (30, 10), (20, 3)]),
            _polyline([(40, 0), (50, 0), (50, 10), (40, 0.02)]),
        )
        result, count = close_open_contours(geo, 0.1)
        assert count == 2
        closed = [c.is_closed() for c in _contours(result)]
        assert closed == [True, False, True]

    def test_ignores_degenerate_single_segment(self):
        geo = _polyline([(0, 0), (0.05, 0)])
        _result, count = close_open_contours(geo, 0.1)
        assert count == 0

    def test_empty_geometry(self):
        result, count = close_open_contours(Geometry(), 0.1)
        assert count == 0
        assert result.is_empty()


class TestJoinOpenContours:
    def test_joins_two_pieces_in_order(self):
        geo = _concat(
            _polyline([(0, 0), (10, 0), (10, 10)]),
            _polyline([(10, 10.05), (0, 10), (0, 0.05)]),
        )
        result, joins = join_open_contours(geo, 0.1)
        assert joins == 1
        contours = _contours(result)
        assert len(contours) == 1
        assert contours[0].is_closed()

    def test_joins_reversed_piece(self):
        geo = _concat(
            _polyline([(0, 0), (10, 0)]),
            _polyline([(20, 0), (10.02, 0)]),
        )
        result, joins = join_open_contours(geo, 0.1)
        assert joins == 1
        contours = _contours(result)
        assert len(contours) == 1
        ends = {_start(contours[0]), _end(contours[0])}
        assert ends == {(0.0, 0.0), (20.0, 0.0)}

    def test_joins_pieces_out_of_order(self):
        geo = _concat(
            _polyline([(0, 0), (10, 0)]),
            _polyline([(20, 0), (30, 0)]),
            _polyline([(10, 0), (20, 0)]),
        )
        result, joins = join_open_contours(geo, 0.1)
        assert joins == 2
        contours = _contours(result)
        assert len(contours) == 1
        assert result.distance() == pytest.approx(30.0)

    def test_joins_at_the_start_of_a_piece(self):
        geo = _concat(
            _polyline([(10, 0), (20, 0)]),
            _polyline([(0, 0), (10, 0)]),
        )
        result, joins = join_open_contours(geo, 0.1)
        assert joins == 1
        contours = _contours(result)
        assert len(contours) == 1
        assert {_start(contours[0]), _end(contours[0])} == {
            (0.0, 0.0),
            (20.0, 0.0),
        }

    def test_keeps_pieces_apart_beyond_tolerance(self):
        geo = _concat(
            _polyline([(0, 0), (10, 0)]),
            _polyline([(11, 0), (20, 0)]),
        )
        result, joins = join_open_contours(geo, 0.1)
        assert joins == 0
        assert len(_contours(result)) == 2

    def test_leaves_closed_contours_alone(self):
        square = _polyline(SQUARE, close=True)
        line = _polyline([(0, 0), (-10, 0)])
        geo = _concat(square, line)
        result, joins = join_open_contours(geo, 0.1)
        assert joins == 0
        assert len(_contours(result)) == 2

    def test_closes_single_piece_whose_ends_meet(self):
        geo = _polyline(SQUARE + [(0, 0.05)])
        result, _joins = join_open_contours(geo, 0.1)
        assert _contours(result)[0].is_closed()

    def test_bridge_does_not_change_bounds(self):
        geo = _concat(
            _polyline([(0, 0), (10, 0), (10, 10)]),
            _polyline([(10.05, 10), (0, 10), (0, 0)]),
        )
        result, _joins = join_open_contours(geo, 0.1)
        assert result.rect() == pytest.approx(geo.rect())

    def test_many_segments_form_one_closed_outline(self):
        pts = [
            (10 * math.cos(a), 10 * math.sin(a))
            for a in (2 * math.pi * i / 60 for i in range(60))
        ]
        pieces = []
        for i in range(60):
            a, b = pts[i], pts[(i + 1) % 60]
            if i % 2:
                a, b = b, a
            pieces.append(_polyline([a, b]))
        geo = _concat(*reversed(pieces))
        result, joins = join_open_contours(geo, 0.01)
        assert joins == 59
        contours = _contours(result)
        assert len(contours) == 1
        assert contours[0].is_closed()


class TestRemoveDuplicateContours:
    def test_removes_identical_copy(self):
        geo = _concat(
            _polyline(SQUARE, close=True), _polyline(SQUARE, close=True)
        )
        result, removed = remove_duplicate_contours(geo, 0.01)
        assert removed == 1
        assert len(_contours(result)) == 1

    def test_removes_reversed_copy(self):
        geo = _concat(
            _polyline(SQUARE, close=True),
            _polyline(list(reversed(SQUARE)), close=True),
        )
        _result, removed = remove_duplicate_contours(geo, 0.01)
        assert removed == 1

    def test_removes_copy_with_other_start_point(self):
        shifted = SQUARE[2:] + SQUARE[:2]
        geo = _concat(
            _polyline(SQUARE, close=True), _polyline(shifted, close=True)
        )
        _result, removed = remove_duplicate_contours(geo, 0.01)
        assert removed == 1

    def test_removes_open_reversed_copy(self):
        line = [(0, 0), (5, 5), (10, 0)]
        geo = _concat(_polyline(line), _polyline(list(reversed(line))))
        _result, removed = remove_duplicate_contours(geo, 0.01)
        assert removed == 1

    def test_keeps_shapes_that_only_overlap_partly(self):
        geo = _concat(
            _polyline(SQUARE, close=True),
            _polyline([(0, 0), (10, 0), (10, 10)]),
        )
        result, removed = remove_duplicate_contours(geo, 0.01)
        assert removed == 0
        assert len(_contours(result)) == 2

    def test_keeps_nested_shapes(self):
        inner = [(2, 2), (8, 2), (8, 8), (2, 8)]
        geo = _concat(
            _polyline(SQUARE, close=True), _polyline(inner, close=True)
        )
        _result, removed = remove_duplicate_contours(geo, 0.01)
        assert removed == 0

    def test_tolerance_applies(self):
        nudged = [(x + 0.005, y) for x, y in SQUARE]
        geo = _concat(
            _polyline(SQUARE, close=True), _polyline(nudged, close=True)
        )
        assert remove_duplicate_contours(geo, 0.01)[1] == 1
        assert remove_duplicate_contours(geo, 0.001)[1] == 0

    def test_keeps_first_occurrence_and_order(self):
        other = [(20, 0), (30, 0), (30, 10)]
        geo = _concat(
            _polyline(SQUARE, close=True),
            _polyline(other),
            _polyline(SQUARE, close=True),
        )
        result, removed = remove_duplicate_contours(geo, 0.01)
        assert removed == 1
        contours = _contours(result)
        assert _start(contours[0]) == (0.0, 0.0)
        assert _start(contours[1]) == (20.0, 0.0)

    def test_removes_triple_copies(self):
        geo = _concat(*(_polyline(SQUARE, close=True) for _ in range(3)))
        _result, removed = remove_duplicate_contours(geo, 0.01)
        assert removed == 2


class TestGeometriesMatch:
    def test_same_shapes_in_other_order_match(self):
        a = _concat(
            _polyline(SQUARE, close=True), _polyline([(20, 0), (30, 0)])
        )
        b = _concat(
            _polyline([(30, 0), (20, 0)]), _polyline(SQUARE, close=True)
        )
        assert geometries_match(a, b, 0.01)

    def test_different_shapes_do_not_match(self):
        a = _polyline(SQUARE, close=True)
        b = _polyline([(0, 0), (10, 0), (10, 11), (0, 10)], close=True)
        assert not geometries_match(a, b, 0.01)

    def test_extra_contour_does_not_match(self):
        a = _polyline(SQUARE, close=True)
        b = _concat(
            _polyline(SQUARE, close=True), _polyline([(20, 0), (30, 0)])
        )
        assert not geometries_match(a, b, 0.01)

    def test_empty_geometries_do_not_match(self):
        assert not geometries_match(Geometry(), Geometry(), 0.01)


class TestSplitContours:
    def test_splits_every_contour_including_holes_and_open_paths(self):
        inner = [(2, 2), (8, 2), (8, 8), (2, 8)]
        geo = _concat(
            _polyline(SQUARE, close=True),
            _polyline(inner, close=True),
            _polyline([(20, 0), (30, 0)]),
        )
        parts = split_contours(geo)
        assert len(parts) == 3
        assert parts[2].distance() == pytest.approx(10.0)

    def test_single_contour(self):
        assert len(split_contours(_polyline(SQUARE, close=True))) == 1

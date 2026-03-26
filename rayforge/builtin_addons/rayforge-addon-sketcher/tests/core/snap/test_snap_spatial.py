import pytest

from sketcher.core.snap.types import SnapLine, SnapLineType
from sketcher.core.snap.spatial import SnapLineIndex, IndexedLine


def test_indexed_line_creation():
    """Tests IndexedLine creation."""
    indexed = IndexedLine(snap_line=None, coordinate=10.5)
    assert indexed.snap_line is None
    assert indexed.coordinate == 10.5


def test_indexed_line_with_snap_line():
    """Tests IndexedLine with a SnapLine."""
    snap_line = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    indexed = IndexedLine(snap_line=snap_line, coordinate=10.0)
    assert indexed.snap_line == snap_line
    assert indexed.coordinate == 10.0


def test_indexed_line_comparison():
    """Tests IndexedLine comparison by coordinate."""
    indexed1 = IndexedLine(None, 10.0)
    indexed2 = IndexedLine(None, 20.0)
    indexed3 = IndexedLine(None, 10.0)

    assert indexed1 < indexed2
    assert not (indexed2 < indexed1)
    assert not (indexed1 < indexed3)


def test_snap_line_index_creation():
    """Tests SnapLineIndex creation."""
    index = SnapLineIndex()
    assert len(index._horizontal) == 0
    assert len(index._vertical) == 0
    assert index._dirty is False


def test_snap_line_index_add_horizontal():
    """Tests adding a horizontal snap line to the index."""
    index = SnapLineIndex()
    snap_line = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    index.add(snap_line)
    assert len(index._horizontal) == 1
    assert len(index._vertical) == 0
    assert index._horizontal[0].snap_line == snap_line
    assert index._horizontal[0].coordinate == 10.0
    assert index._dirty is True


def test_snap_line_index_add_vertical():
    """Tests adding a vertical snap line to the index."""
    index = SnapLineIndex()
    snap_line = SnapLine(
        is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
    )
    index.add(snap_line)
    assert len(index._horizontal) == 0
    assert len(index._vertical) == 1
    assert index._vertical[0].snap_line == snap_line
    assert index._vertical[0].coordinate == 20.0
    assert index._dirty is True


def test_snap_line_index_add_multiple():
    """Tests adding multiple snap lines to the index."""
    index = SnapLineIndex()
    snap_line1 = SnapLine(
        is_horizontal=True,
        coordinate=20.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    snap_line2 = SnapLine(
        is_horizontal=True, coordinate=10.0, line_type=SnapLineType.CENTER
    )
    snap_line3 = SnapLine(
        is_horizontal=False, coordinate=15.0, line_type=SnapLineType.MIDPOINT
    )

    index.add(snap_line1)
    index.add(snap_line2)
    index.add(snap_line3)

    assert len(index._horizontal) == 2
    assert len(index._vertical) == 1
    assert index._horizontal[0].coordinate == 10.0
    assert index._horizontal[1].coordinate == 20.0


def test_snap_line_index_add_all():
    """Tests adding multiple snap lines using add_all."""
    index = SnapLineIndex()
    snap_lines = [
        SnapLine(
            is_horizontal=True,
            coordinate=10.0,
            line_type=SnapLineType.ENTITY_POINT,
        ),
        SnapLine(
            is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
        ),
        SnapLine(
            is_horizontal=True,
            coordinate=30.0,
            line_type=SnapLineType.MIDPOINT,
        ),
    ]
    index.add_all(iter(snap_lines))

    assert len(index._horizontal) == 2
    assert len(index._vertical) == 1


def test_snap_line_index_clear():
    """Tests clearing the snap line index."""
    index = SnapLineIndex()
    snap_line1 = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    snap_line2 = SnapLine(
        is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
    )
    index.add(snap_line1)
    index.add(snap_line2)

    assert len(index) == 2

    index.clear()

    assert len(index._horizontal) == 0
    assert len(index._vertical) == 0
    assert index._dirty is False
    assert len(index) == 0


def test_snap_line_index_query_horizontal():
    """Tests querying horizontal snap lines."""
    index = SnapLineIndex()
    snap_line1 = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    snap_line2 = SnapLine(
        is_horizontal=True, coordinate=20.0, line_type=SnapLineType.CENTER
    )
    snap_line3 = SnapLine(
        is_horizontal=True, coordinate=30.0, line_type=SnapLineType.MIDPOINT
    )

    index.add(snap_line1)
    index.add(snap_line2)
    index.add(snap_line3)

    results = index.query_horizontal(22.0, 5.0)
    assert len(results) == 1
    snap_lines, distances = zip(*results)
    assert snap_line2 in snap_lines
    assert 2.0 in distances


def test_snap_line_index_query_vertical():
    """Tests querying vertical snap lines."""
    index = SnapLineIndex()
    snap_line1 = SnapLine(
        is_horizontal=False,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    snap_line2 = SnapLine(
        is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
    )
    snap_line3 = SnapLine(
        is_horizontal=False, coordinate=30.0, line_type=SnapLineType.MIDPOINT
    )

    index.add(snap_line1)
    index.add(snap_line2)
    index.add(snap_line3)

    results = index.query_vertical(22.0, 5.0)
    assert len(results) == 1
    snap_lines, distances = zip(*results)
    assert snap_line2 in snap_lines
    assert 2.0 in distances


def test_snap_line_index_query_combined():
    """Tests querying both horizontal and vertical snap lines."""
    index = SnapLineIndex()
    h_line1 = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    h_line2 = SnapLine(
        is_horizontal=True, coordinate=20.0, line_type=SnapLineType.CENTER
    )
    v_line1 = SnapLine(
        is_horizontal=False, coordinate=15.0, line_type=SnapLineType.MIDPOINT
    )
    v_line2 = SnapLine(
        is_horizontal=False,
        coordinate=25.0,
        line_type=SnapLineType.INTERSECTION,
    )

    index.add(h_line1)
    index.add(h_line2)
    index.add(v_line1)
    index.add(v_line2)

    results = index.query(22.0, 18.0, 5.0)
    assert len(results) == 2
    snap_lines, distances = zip(*results)
    assert h_line2 in snap_lines
    assert v_line2 in snap_lines


def test_snap_line_index_query_empty():
    """Tests querying an empty index."""
    index = SnapLineIndex()
    results = index.query(10.0, 20.0, 5.0)
    assert results == []


def test_snap_line_index_query_no_results():
    """Tests querying with no results."""
    index = SnapLineIndex()
    h_line = SnapLine(
        is_horizontal=True,
        coordinate=100.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    v_line = SnapLine(
        is_horizontal=False, coordinate=200.0, line_type=SnapLineType.CENTER
    )
    index.add(h_line)
    index.add(v_line)

    results = index.query(10.0, 20.0, 5.0)
    assert results == []


def test_snap_line_index_query_exact_match():
    """Tests querying with exact coordinate match."""
    index = SnapLineIndex()
    h_line = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    v_line = SnapLine(
        is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
    )
    index.add(h_line)
    index.add(v_line)

    results = index.query(20.0, 10.0, 1.0)
    assert len(results) == 2
    snap_lines, distances = zip(*results)
    assert h_line in snap_lines
    assert v_line in snap_lines
    assert 0.0 in distances


def test_snap_line_index_len():
    """Tests SnapLineIndex __len__ method."""
    index = SnapLineIndex()
    assert len(index) == 0

    h_line1 = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    index.add(h_line1)
    assert len(index) == 1

    v_line1 = SnapLine(
        is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
    )
    index.add(v_line1)
    assert len(index) == 2

    h_line2 = SnapLine(
        is_horizontal=True, coordinate=30.0, line_type=SnapLineType.MIDPOINT
    )
    index.add(h_line2)
    assert len(index) == 3


def test_snap_line_index_sorting():
    """Tests that snap lines are kept sorted by coordinate."""
    index = SnapLineIndex()
    snap_lines = [
        SnapLine(
            is_horizontal=True,
            coordinate=30.0,
            line_type=SnapLineType.MIDPOINT,
        ),
        SnapLine(
            is_horizontal=True,
            coordinate=10.0,
            line_type=SnapLineType.ENTITY_POINT,
        ),
        SnapLine(
            is_horizontal=True, coordinate=20.0, line_type=SnapLineType.CENTER
        ),
    ]
    for sl in snap_lines:
        index.add(sl)

    assert index._horizontal[0].coordinate == 10.0
    assert index._horizontal[1].coordinate == 20.0
    assert index._horizontal[2].coordinate == 30.0


def test_snap_line_index_query_sorting():
    """Tests that query results are sorted by distance and priority."""
    index = SnapLineIndex()
    h_line1 = SnapLine(
        is_horizontal=True, coordinate=10.0, line_type=SnapLineType.MIDPOINT
    )
    h_line2 = SnapLine(
        is_horizontal=True,
        coordinate=12.0,
        line_type=SnapLineType.ENTITY_POINT,
    )
    v_line1 = SnapLine(
        is_horizontal=False, coordinate=20.0, line_type=SnapLineType.CENTER
    )

    index.add(h_line1)
    index.add(h_line2)
    index.add(v_line1)

    results = index.query(20.0, 11.0, 5.0)
    assert len(results) == 3
    assert results[0][1] == 0.0
    assert results[1][1] == 1.0
    assert results[1][0].line_type == SnapLineType.ENTITY_POINT


def test_snap_line_index_query_with_none_snap_line():
    """Tests that None snap lines are skipped in queries."""
    index = SnapLineIndex()
    index._horizontal.append(IndexedLine(None, 10.0))
    index._horizontal.append(IndexedLine(None, 20.0))

    results = index.query_horizontal(15.0, 5.0)
    assert results == []


def test_snap_line_index_add_all_with_none():
    """Tests that add_all doesn't handle None values."""
    index = SnapLineIndex()
    snap_line1 = SnapLine(
        is_horizontal=True,
        coordinate=10.0,
        line_type=SnapLineType.ENTITY_POINT,
    )

    with pytest.raises(AttributeError):
        index.add_all(iter([snap_line1, None]))  # type: ignore

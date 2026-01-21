import numpy as np
from rayforge.core.geo import Geometry
from rayforge.core.geo.cleanup import (
    remove_duplicate_segments,
    _are_points_equal,
    _get_segment_key,
    _are_segments_equal,
    close_geometry_gaps,
)
from rayforge.core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    CMD_TYPE_BEZIER,
    COL_TYPE,
)


def test_are_points_equal_exact():
    """Tests exact point equality."""
    p1 = (1.0, 2.0, 3.0)
    p2 = (1.0, 2.0, 3.0)
    assert _are_points_equal(p1, p2, 1e-6)


def test_are_points_equal_within_tolerance():
    """Tests point equality within tolerance."""
    p1 = (1.0, 2.0, 3.0)
    p2 = (1.000001, 2.000001, 3.000001)
    assert _are_points_equal(p1, p2, 1e-5)


def test_are_points_equal_outside_tolerance():
    """Tests point inequality outside tolerance."""
    p1 = (1.0, 2.0, 3.0)
    p2 = (1.1, 2.1, 3.1)
    assert not _are_points_equal(p1, p2, 1e-6)


def test_are_points_equal_partial_difference():
    """Tests point equality with only some coordinates differing."""
    p1 = (1.0, 2.0, 3.0)
    p2 = (1.0, 2.0, 3.1)
    assert not _are_points_equal(p1, p2, 1e-6)


def test_get_segment_key_line():
    """Tests segment key generation for line commands."""
    data = np.array(
        [
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    key = _get_segment_key(data, 0, 1e-6)
    assert key is not None
    assert key[0] == "LINE"
    assert key[1] == (10.0, 0.0, 0.0)


def test_get_segment_key_arc():
    """Tests segment key generation for arc commands."""
    data = np.array(
        [
            [CMD_TYPE_ARC, 10.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0],
        ]
    )
    key = _get_segment_key(data, 0, 1e-6)
    assert key is not None
    assert key[0] == "ARC"
    assert key[1] == (10.0, 0.0, 0.0)
    assert key[2] == (5.0, 0.0)
    assert key[3] is True


def test_get_segment_key_bezier():
    """Tests segment key generation for bezier commands."""
    data = np.array(
        [
            [CMD_TYPE_BEZIER, 10.0, 0.0, 0.0, 3.0, 3.0, 7.0, -3.0],
        ]
    )
    key = _get_segment_key(data, 0, 1e-6)
    assert key is not None
    assert key[0] == "BEZIER"
    assert key[1] == (10.0, 0.0, 0.0)
    assert key[2] == (3.0, 3.0)
    assert key[3] == (7.0, -3.0)


def test_get_segment_key_move():
    """Tests that move commands return None as key."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    key = _get_segment_key(data, 0, 1e-6)
    assert key is None


def test_get_segment_key_invalid_index():
    """Tests that invalid index returns None."""
    data = np.array(
        [
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    key = _get_segment_key(data, 10, 1e-6)
    assert key is None


def test_are_segments_equal_line():
    """Tests equality check for line segments."""
    key1 = ("LINE", (0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    key2 = ("LINE", (0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    assert _are_segments_equal(key1, key2, 1e-6)


def test_are_segments_equal_line_within_tolerance():
    """Tests equality check for line segments within tolerance."""
    key1 = ("LINE", (0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    key2 = ("LINE", (0.000001, 0.0, 0.0), (10.0, 0.0, 0.0))
    assert _are_segments_equal(key1, key2, 1e-5)


def test_are_segments_equal_line_different_type():
    """Tests inequality for segments of different types."""
    key1 = ("LINE", (0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    key2 = ("ARC", (0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (5.0, 0.0), True)
    assert not _are_segments_equal(key1, key2, 1e-6)


def test_are_segments_equal_arc():
    """Tests equality check for arc segments."""
    key1 = ("ARC", (0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (5.0, 0.0), True)
    key2 = ("ARC", (0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (5.0, 0.0), True)
    assert _are_segments_equal(key1, key2, 1e-6)


def test_are_segments_equal_arc_different_direction():
    """Tests inequality for arcs with different direction."""
    key1 = ("ARC", (10.0, 0.0, 0.0), (5.0, 0.0), True)
    key2 = ("ARC", (10.0, 0.0, 0.0), (5.0, 0.0), False)
    assert not _are_segments_equal(key1, key2, 1e-6)


def test_are_segments_equal_bezier():
    """Tests equality check for bezier segments."""
    key1 = (
        "BEZIER",
        (10.0, 0.0, 0.0),
        (3.0, 3.0),
        (7.0, -3.0),
    )
    key2 = (
        "BEZIER",
        (10.0, 0.0, 0.0),
        (3.0, 3.0),
        (7.0, -3.0),
    )
    assert _are_segments_equal(key1, key2, 1e-6)


def test_remove_duplicate_segments_empty():
    """Tests that empty array is handled correctly."""
    data = np.array([]).reshape(0, 8)
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 0


def test_remove_duplicate_segments_none():
    """Tests that None input is handled correctly."""
    result = remove_duplicate_segments(None)
    assert result is None


def test_remove_duplicate_segments_single_line():
    """Tests that single line segment is preserved."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 2


def test_remove_duplicate_segments_duplicate_lines():
    """Tests that duplicate line segments are removed."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 2
    assert result[0, 0] == CMD_TYPE_MOVE
    assert result[1, 0] == CMD_TYPE_LINE


def test_remove_duplicate_segments_three_duplicate_lines():
    """Tests that multiple duplicate line segments are removed."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 2


def test_remove_duplicate_segments_different_lines():
    """Tests that different line segments are preserved."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 3


def test_remove_duplicate_segments_duplicate_arcs():
    """Tests that duplicate arc segments are removed."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_ARC, 10.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0],
            [CMD_TYPE_ARC, 10.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 2


def test_remove_duplicate_segments_different_arcs():
    """Tests that different arc segments are preserved."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_ARC, 10.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0],
            [CMD_TYPE_ARC, 10.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 3


def test_remove_duplicate_segments_duplicate_beziers():
    """Tests that duplicate bezier segments are removed."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_BEZIER, 10.0, 0.0, 0.0, 3.0, 3.0, 7.0, -3.0],
            [CMD_TYPE_BEZIER, 10.0, 0.0, 0.0, 3.0, 3.0, 7.0, -3.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 2


def test_remove_duplicate_segments_different_beziers():
    """Tests that different bezier segments are preserved."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_BEZIER, 10.0, 0.0, 0.0, 3.0, 3.0, 7.0, -3.0],
            [CMD_TYPE_BEZIER, 10.0, 0.0, 0.0, 3.0, 3.0, 7.0, -4.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 3


def test_remove_duplicate_segments_mixed_types():
    """Tests handling of mixed segment types."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_ARC, 10.0, 0.0, 0.0, 2.5, 0.0, 1.0, 0.0],
            [CMD_TYPE_LINE, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 3


def test_remove_duplicate_segments_multiple_paths():
    """Tests handling of multiple separate paths."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_MOVE, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 4


def test_remove_duplicate_segments_within_tolerance():
    """Tests that segments within tolerance are considered duplicates."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data, tolerance=1e-5)
    assert result is not None
    assert len(result) == 2


def test_remove_duplicate_segments_outside_tolerance():
    """Tests that segments outside tolerance are not duplicates."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data, tolerance=1e-4)
    assert result is not None
    assert len(result) == 3


def test_remove_duplicate_segments_preserves_z():
    """Tests that Z coordinates are considered in duplicate check."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 3


def test_remove_duplicate_segments_complex_path():
    """Tests handling of a complex path with multiple segments."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 5


def test_remove_duplicate_segments_no_duplicates():
    """Tests that path without duplicates remains unchanged."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 4


def test_remove_duplicate_segments_consecutive_same_start():
    """Tests segments with same start but different endpoints."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 3


def test_remove_duplicate_segments_move_only():
    """Tests that move commands are always preserved."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_MOVE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_MOVE, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 3


def test_remove_duplicate_segments_arc_center_offset():
    """Tests that arc center offset affects duplicate detection."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_ARC, 10.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0],
            [CMD_TYPE_ARC, 10.0, 0.0, 0.0, 5.00001, 0.0, 1.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data, tolerance=1e-5)
    assert result is not None
    assert len(result) == 2


def test_remove_duplicate_segments_bezier_control_points():
    """Tests that bezier control points affect duplicate detection."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_BEZIER, 10.0, 0.0, 0.0, 3.0, 3.0, 7.0, -3.0],
            [CMD_TYPE_BEZIER, 10.0, 0.0, 0.0, 3.0, 3.0, 7.001, -3.0],
        ]
    )
    result = remove_duplicate_segments(data, tolerance=1e-5)
    assert result is not None
    assert len(result) == 3


def test_remove_duplicate_segments_non_consecutive_duplicates():
    """Tests that non-consecutive duplicates are also removed."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_MOVE, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 4


def test_remove_duplicate_segments_default_tolerance():
    """Tests that default tolerance works correctly."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 2


def test_remove_duplicate_segments_zero_tolerance():
    """Tests behavior with zero tolerance."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data, tolerance=0.0)
    assert result is not None
    assert len(result) == 2


def test_remove_duplicate_segments_large_tolerance():
    """Tests behavior with large tolerance."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data, tolerance=1.0)
    assert result is not None
    assert len(result) == 2


def test_remove_duplicate_segments_vertical_line():
    """Tests duplicate detection on vertical lines."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 2


def test_remove_duplicate_segments_diagonal_line():
    """Tests duplicate detection on diagonal lines."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 2


def test_remove_duplicate_segments_all_duplicates():
    """Tests when all segments are duplicates."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 2


def test_remove_duplicate_segments_arc_ccw_vs_cw():
    """Tests that CCW and CW arcs are not considered duplicates."""
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_ARC, 10.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0],
            [CMD_TYPE_ARC, 10.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        ]
    )
    result = remove_duplicate_segments(data)
    assert result is not None
    assert len(result) == 3


def test_close_geometry_gaps_functional():
    """Tests the core logic of the close_geometry_gaps function."""
    geo_intra = Geometry()
    geo_intra.move_to(0, 0)
    geo_intra.line_to(10, 0)
    geo_intra.line_to(10, 10)
    geo_intra.line_to(0.000001, 10)
    geo_intra.line_to(0.000002, 0.000003)
    original_intra_data = geo_intra.copy().data
    assert original_intra_data is not None

    result_intra = close_geometry_gaps(geo_intra, tolerance=1e-5)
    assert result_intra.data is not None
    assert result_intra is not geo_intra
    assert geo_intra.data is not None
    assert not np.array_equal(result_intra.data, geo_intra.data)
    assert np.all(geo_intra.data[-1, 1:4] == original_intra_data[-1, 1:4])
    assert np.all(result_intra.data[0, 1:4] == (0, 0, 0))
    assert np.all(result_intra.data[-1, 1:4] == (0, 0, 0))

    geo_inter = Geometry()
    geo_inter.move_to(0, 0)
    geo_inter.line_to(10, 10)
    geo_inter.move_to(10.000001, 10.000002)
    geo_inter.line_to(20, 20)
    original_inter_data = geo_inter.copy().data
    assert original_inter_data is not None

    result_inter = close_geometry_gaps(geo_inter, tolerance=1e-5)
    assert result_inter.data is not None
    assert result_inter is not geo_inter
    assert geo_inter.data is not None
    assert not np.array_equal(result_inter.data, geo_inter.data)
    assert geo_inter.data[2, COL_TYPE] == CMD_TYPE_MOVE
    assert np.all(geo_inter.data[2, 1:4] == original_inter_data[2, 1:4])
    assert result_inter.data[2, COL_TYPE] == CMD_TYPE_LINE
    assert np.all(result_inter.data[2, 1:4] == (10, 10, 0))


def test_close_geometry_gaps_respects_tolerance():
    """Tests that the tolerance parameter is correctly used."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)
    geo.move_to(10.1, 10.1)
    geo.line_to(20, 20)

    result1 = close_geometry_gaps(geo, tolerance=0.1)
    assert result1.data is not None
    assert result1.data[2, COL_TYPE] == CMD_TYPE_MOVE
    assert np.all(result1.data[2, 1:4] == (10.1, 10.1, 0))

    result2 = close_geometry_gaps(geo, tolerance=0.2)
    assert result2.data is not None
    assert result2.data[2, COL_TYPE] == CMD_TYPE_LINE
    assert np.all(result2.data[2, 1:4] == (10, 10, 0))

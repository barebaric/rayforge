import math

from post_processors.transformers.tabs_transformer import _ClipPoint
from raygeo.geo.shape.bezier import get_bezier_point_at
from raygeo.ops import Ops
from raygeo.ops.types import CommandCategory, CommandType, SectionType

_P0 = (0.0, 0.0, 0.0)
_C1 = (33.0, 10.0, 0.0)
_C2 = (66.0, 10.0, 0.0)
_P1 = (100.0, 0.0, 0.0)


def _bezier_point_2d(t):
    return get_bezier_point_at(_P0[:2], _C1[:2], _C2[:2], _P1[:2], t)


def _make_sectioned_bezier_ops():
    ops = Ops()
    uid = "test-wp"
    ops.ops_section_start(
        section_type=SectionType.VECTOR_OUTLINE,
        workpiece_uid=uid,
    )
    ops.move_to(*_P0)
    ops.bezier_to(_C1, _C2, _P1)
    ops.ops_section_end(section_type=SectionType.VECTOR_OUTLINE)
    return ops


def _make_sectioned_mixed_ops():
    ops = Ops()
    uid = "test-wp"
    ops.ops_section_start(
        section_type=SectionType.VECTOR_OUTLINE,
        workpiece_uid=uid,
    )
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(30.0, 0.0, 0.0)
    ops.bezier_to(
        (40.0, 10.0, 0.0),
        (60.0, 10.0, 0.0),
        (70.0, 0.0, 0.0),
    )
    ops.line_to(100.0, 0.0, 0.0)
    ops.ops_section_end(section_type=SectionType.VECTOR_OUTLINE)
    return ops


def _count_in_ops(ops, cmd_type):
    return sum(
        1 for i in range(ops.len()) if ops.command_type(i) == cmd_type
    )


# ------------------------------------------------------------------
# Gap mode tests via ops.apply_tab_gaps
# ------------------------------------------------------------------


def test_bezier_no_clips_passes_through():
    ops = _make_sectioned_bezier_ops()
    ops.apply_tab_gaps([])

    assert _count_in_ops(ops, CommandType.BEZIER_TO) == 1
    assert _count_in_ops(ops, CommandType.MOVE_TO) >= 1


def test_bezier_gap_splits_into_two_beziers():
    ops = _make_sectioned_bezier_ops()
    mid = _bezier_point_2d(0.5)
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]
    ops.apply_tab_gaps(clips)

    bezier_count = _count_in_ops(ops, CommandType.BEZIER_TO)
    assert bezier_count == 2, (
        f"Expected 2 bezier segments, got {bezier_count}"
    )


def test_bezier_gap_preserves_start_and_end():
    ops = _make_sectioned_bezier_ops()
    mid = _bezier_point_2d(0.5)
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]
    ops.apply_tab_gaps(clips)

    first_moving = None
    last_moving = None
    for i in range(ops.len()):
        if ops.category(i) == CommandCategory.MOVING:
            pt = ops.endpoint(i)
            if first_moving is None:
                first_moving = pt
            last_moving = pt

    assert first_moving is not None
    assert math.dist(first_moving[:2], _P0[:2]) < 1e-3

    assert last_moving is not None
    assert math.dist(last_moving[:2], _P1[:2]) < 1e-3


def test_bezier_gap_multiple_clips():
    ops = _make_sectioned_bezier_ops()
    pt25 = _bezier_point_2d(0.25)
    pt75 = _bezier_point_2d(0.75)
    clips = [
        _ClipPoint(x=pt25[0], y=pt25[1], width=6.0),
        _ClipPoint(x=pt75[0], y=pt75[1], width=6.0),
    ]
    ops.apply_tab_gaps(clips)

    bezier_count = _count_in_ops(ops, CommandType.BEZIER_TO)
    assert bezier_count >= 3

    last_end = None
    for ri in range(ops.len() - 1, -1, -1):
        ct = ops.command_type(ri)
        if ct in (CommandType.BEZIER_TO, CommandType.LINE_TO):
            last_end = ops.endpoint(ri)
            break
    assert last_end is not None
    assert math.dist(last_end[:2], _P1[:2]) < 1e-3


def test_mixed_lines_and_bezier_gap():
    ops = _make_sectioned_mixed_ops()
    mid = get_bezier_point_at(
        (30.0, 0.0), (40.0, 10.0), (60.0, 10.0), (70.0, 0.0), 0.5
    )
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]
    ops.apply_tab_gaps(clips)

    has_bezier = _count_in_ops(ops, CommandType.BEZIER_TO) > 0
    has_line = _count_in_ops(ops, CommandType.LINE_TO) > 0
    assert has_bezier
    assert has_line


def test_bezier_gap_via_apply_tab_gaps():
    ops = _make_sectioned_bezier_ops()
    mid = _bezier_point_2d(0.5)
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]
    ops.apply_tab_gaps(clips)

    bezier_count = _count_in_ops(ops, CommandType.BEZIER_TO)
    assert bezier_count == 2


# ------------------------------------------------------------------
# Power mode tests via ops.apply_tab_power
# ------------------------------------------------------------------


def test_bezier_power_mode_splits_and_inserts_power():
    ops = _make_sectioned_bezier_ops()
    mid = _bezier_point_2d(0.5)
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]
    tab_power = 0.3
    original_power = 1.0
    ops.apply_tab_power(clips, tab_power, original_power)

    bezier_count = _count_in_ops(ops, CommandType.BEZIER_TO)
    power_count = _count_in_ops(ops, CommandType.SET_POWER)

    assert bezier_count >= 2
    assert power_count >= 2

    powers = [
        ops.power(i)
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.SET_POWER
    ]
    assert tab_power in powers
    assert original_power in powers


def test_bezier_power_mode_no_overlap():
    ops = _make_sectioned_bezier_ops()
    clips = [_ClipPoint(x=200.0, y=200.0, width=10.0)]
    tab_power = 0.3
    original_power = 1.0
    ops.apply_tab_power(clips, tab_power, original_power)

    bezier_count = _count_in_ops(ops, CommandType.BEZIER_TO)
    assert bezier_count == 1

    bezier_idx = next(
        i for i in range(ops.len())
        if ops.command_type(i) == CommandType.BEZIER_TO
    )
    assert ops.endpoint(bezier_idx) == _P1


def test_mixed_lines_and_bezier_power():
    ops = _make_sectioned_mixed_ops()
    mid = get_bezier_point_at(
        (30.0, 0.0), (40.0, 10.0), (60.0, 10.0), (70.0, 0.0), 0.5
    )
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]
    tab_power = 0.3
    original_power = 1.0
    ops.apply_tab_power(clips, tab_power, original_power)

    bezier_count = _count_in_ops(ops, CommandType.BEZIER_TO)
    power_count = _count_in_ops(ops, CommandType.SET_POWER)

    assert bezier_count >= 1
    assert power_count >= 1


# ------------------------------------------------------------------
# Non-curve path tests
# ------------------------------------------------------------------


def test_non_curve_path_unchanged_behavior():
    ops = Ops()
    ops.ops_section_start(
        section_type=SectionType.VECTOR_OUTLINE,
        workpiece_uid="test",
    )
    ops.move_to(0, 0)
    ops.line_to(100, 0)
    ops.ops_section_end(section_type=SectionType.VECTOR_OUTLINE)

    clips = [_ClipPoint(x=50.0, y=0.0, width=10.0)]
    ops.apply_tab_gaps(clips)

    line_count = _count_in_ops(ops, CommandType.LINE_TO)
    move_count = _count_in_ops(ops, CommandType.MOVE_TO)

    assert line_count == 2
    assert move_count >= 2

import math

from raygeo.shape.bezier import get_bezier_point_at
from rayforge.core.ops import (
    Ops,
    SectionType,
)
from rayforge.core.ops.enums import CommandType, CommandCategory
from post_processors.transformers import TabOpsTransformer
from post_processors.transformers.tabs_transformer import _ClipPoint


_P0 = (0.0, 0.0, 0.0)
_C1 = (33.0, 10.0, 0.0)
_C2 = (66.0, 10.0, 0.0)
_P1 = (100.0, 0.0, 0.0)


def _bezier_point_2d(t):
    return get_bezier_point_at(_P0[:2], _C1[:2], _C2[:2], _P1[:2], t)


def _make_bezier_subpath():
    ops = Ops()
    ops.move_to(*_P0)
    ops.bezier_to(_C1, _C2, _P1)
    return ops


def _make_mixed_subpath():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(30.0, 0.0, 0.0)
    ops.bezier_to(
        (40.0, 10.0, 0.0),
        (60.0, 10.0, 0.0),
        (70.0, 0.0, 0.0),
    )
    ops.line_to(100.0, 0.0, 0.0)
    return ops


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


# ------------------------------------------------------------------
# Gap mode tests via _clip_subpath_with_gaps
# ------------------------------------------------------------------


def test_bezier_no_clips_passes_through():
    transformer = TabOpsTransformer()
    sub_ops = _make_bezier_subpath()
    result = transformer._clip_subpath_with_gaps(sub_ops, [])

    assert result.len() == sub_ops.len()
    assert result.command_type(0) == CommandType.MOVE_TO
    assert result.command_type(1) == CommandType.BEZIER_TO
    assert result.endpoint(1) == _P1
    c1, c2 = result.bezier_params(1)
    assert c1 == _C1
    assert c2 == _C2


def test_bezier_gap_splits_into_two_beziers():
    transformer = TabOpsTransformer()
    sub_ops = _make_bezier_subpath()
    mid = _bezier_point_2d(0.5)
    clip_width = 10.0

    clips = [_ClipPoint(x=mid[0], y=mid[1], width=clip_width)]
    result = transformer._clip_subpath_with_gaps(sub_ops, clips)

    bezier_indices = [
        i
        for i in range(result.len())
        if result.command_type(i) == CommandType.BEZIER_TO
    ]
    move_indices = [
        i
        for i in range(result.len())
        if result.command_type(i) == CommandType.MOVE_TO
    ]

    assert len(bezier_indices) == 2, (
        f"Expected 2 bezier segments, got {len(bezier_indices)}"
    )
    assert len(move_indices) >= 2

    first_end_x = result.endpoint(bezier_indices[0])[0]
    assert first_end_x < mid[0]


def test_bezier_gap_preserves_start_and_end():
    transformer = TabOpsTransformer()
    sub_ops = _make_bezier_subpath()
    mid = _bezier_point_2d(0.5)

    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]
    result = transformer._clip_subpath_with_gaps(sub_ops, clips)

    first_moving = None
    last_moving = None
    for i in range(result.len()):
        if result.category(i) == CommandCategory.MOVING:
            pt = result.endpoint(i)
            if first_moving is None:
                first_moving = pt
            last_moving = pt

    assert first_moving is not None
    assert math.dist(first_moving[:2], _P0[:2]) < 1e-3

    assert last_moving is not None
    assert math.dist(last_moving[:2], _P1[:2]) < 1e-3


def test_bezier_gap_multiple_clips():
    transformer = TabOpsTransformer()
    sub_ops = _make_bezier_subpath()

    pt25 = _bezier_point_2d(0.25)
    pt75 = _bezier_point_2d(0.75)
    clips = [
        _ClipPoint(x=pt25[0], y=pt25[1], width=6.0),
        _ClipPoint(x=pt75[0], y=pt75[1], width=6.0),
    ]

    result = transformer._clip_subpath_with_gaps(sub_ops, clips)

    bezier_indices = [
        i
        for i in range(result.len())
        if result.command_type(i) == CommandType.BEZIER_TO
    ]
    assert len(bezier_indices) >= 3

    last_end = None
    for ri in range(result.len() - 1, -1, -1):
        ct = result.command_type(ri)
        if ct in (CommandType.BEZIER_TO, CommandType.LINE_TO):
            last_end = result.endpoint(ri)
            break
    assert last_end is not None
    assert math.dist(last_end[:2], _P1[:2]) < 1e-3


def test_mixed_lines_and_bezier_gap():
    transformer = TabOpsTransformer()
    sub_ops = _make_mixed_subpath()

    mid = get_bezier_point_at(
        (30.0, 0.0), (40.0, 10.0), (60.0, 10.0), (70.0, 0.0), 0.5
    )
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]

    result = transformer._clip_subpath_with_gaps(sub_ops, clips)

    has_bezier = any(
        result.command_type(i) == CommandType.BEZIER_TO
        for i in range(result.len())
    )
    has_line = any(
        result.command_type(i) == CommandType.LINE_TO
        for i in range(result.len())
    )
    assert has_bezier
    assert has_line


def test_bezier_gap_via_apply_tab_gaps():
    ops = _make_sectioned_bezier_ops()
    mid = _bezier_point_2d(0.5)
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]

    transformer = TabOpsTransformer()
    transformer._apply_tab_gaps(ops, clips)

    bezier_count = sum(
        1
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.BEZIER_TO
    )
    assert bezier_count == 2


# ------------------------------------------------------------------
# Power mode tests via _insert_power_commands
# ------------------------------------------------------------------


def test_bezier_power_mode_splits_and_inserts_power():
    sub_ops = _make_bezier_subpath()
    mid = _bezier_point_2d(0.5)
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]
    tab_power = 0.3
    original_power = 1.0

    transformer = TabOpsTransformer()
    result = transformer._insert_power_commands(
        sub_ops, clips, tab_power, original_power
    )

    bezier_indices = [
        i
        for i in range(result.len())
        if result.command_type(i) == CommandType.BEZIER_TO
    ]
    power_indices = [
        i
        for i in range(result.len())
        if result.command_type(i) == CommandType.SET_POWER
    ]

    assert len(bezier_indices) >= 2
    assert len(power_indices) >= 2

    powers = [result.power(i) for i in power_indices]
    assert tab_power in powers
    assert original_power in powers


def test_bezier_power_mode_no_overlap():
    sub_ops = _make_bezier_subpath()
    clips = [_ClipPoint(x=200.0, y=200.0, width=10.0)]
    tab_power = 0.3
    original_power = 1.0

    transformer = TabOpsTransformer()
    result = transformer._insert_power_commands(
        sub_ops, clips, tab_power, original_power
    )

    bezier_indices = [
        i
        for i in range(result.len())
        if result.command_type(i) == CommandType.BEZIER_TO
    ]
    assert len(bezier_indices) == 1
    assert result.endpoint(bezier_indices[0]) == _P1
    c1, c2 = result.bezier_params(bezier_indices[0])
    assert c1 == _C1
    assert c2 == _C2


def test_mixed_lines_and_bezier_power():
    sub_ops = _make_mixed_subpath()

    mid = get_bezier_point_at(
        (30.0, 0.0), (40.0, 10.0), (60.0, 10.0), (70.0, 0.0), 0.5
    )
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]
    tab_power = 0.3
    original_power = 1.0

    transformer = TabOpsTransformer()
    result = transformer._insert_power_commands(
        sub_ops, clips, tab_power, original_power
    )

    bezier_count = sum(
        1
        for i in range(result.len())
        if result.command_type(i) == CommandType.BEZIER_TO
    )
    power_count = sum(
        1
        for i in range(result.len())
        if result.command_type(i) == CommandType.SET_POWER
    )

    assert bezier_count >= 1
    assert power_count >= 1


# ------------------------------------------------------------------
# Helper method tests
# ------------------------------------------------------------------


def test_extract_bezier_subsegment_full():
    transformer = TabOpsTransformer()
    sub = transformer._extract_bezier_subsegment_3d(
        _P0, _C1, _C2, _P1, 0.0, 1.0
    )
    assert sub[0] == _P0
    assert sub[3] == _P1


def test_extract_bezier_subsegment_first_half():
    transformer = TabOpsTransformer()
    sub = transformer._extract_bezier_subsegment_3d(
        _P0, _C1, _C2, _P1, 0.0, 0.5
    )
    assert sub[0] == _P0
    mid_x = sub[3][0]
    assert 40 < mid_x < 60


def test_extract_bezier_subsegment_second_half():
    transformer = TabOpsTransformer()
    sub = transformer._extract_bezier_subsegment_3d(
        _P0, _C1, _C2, _P1, 0.5, 1.0
    )
    mid_x = sub[0][0]
    assert 40 < mid_x < 60
    assert sub[3] == _P1


def test_bezier_arc_length():
    transformer = TabOpsTransformer()
    length = transformer._bezier_arc_length_2d(_P0, _C1, _C2, _P1)
    assert length > 100.0
    assert length < 200.0


def test_bezier_distance_to_t():
    transformer = TabOpsTransformer()

    t = transformer._bezier_distance_to_t(_P0, _C1, _C2, _P1, 0.0)
    assert abs(t) < 0.01

    length = transformer._bezier_arc_length_2d(_P0, _C1, _C2, _P1)
    t_end = transformer._bezier_distance_to_t(_P0, _C1, _C2, _P1, length)
    assert abs(t_end - 1.0) < 0.01

    t_mid = transformer._bezier_distance_to_t(_P0, _C1, _C2, _P1, length / 2)
    assert 0.3 < t_mid < 0.7


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

    transformer = TabOpsTransformer()
    transformer._apply_tab_gaps(ops, clips)

    line_count = sum(
        1
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.LINE_TO
    )
    move_count = sum(
        1
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.MOVE_TO
    )

    assert line_count == 2
    assert move_count >= 2

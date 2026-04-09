import math

from rayforge.core.geo.bezier import evaluate_bezier
from rayforge.core.ops import (
    BezierToCommand,
    LineToCommand,
    MoveToCommand,
    Ops,
    OpsSectionEndCommand,
    OpsSectionStartCommand,
    SectionType,
    SetPowerCommand,
)
from post_processors.transformers import TabOpsTransformer
from post_processors.transformers.tabs_transformer import _ClipPoint


_P0 = (0.0, 0.0, 0.0)
_C1 = (33.0, 10.0, 0.0)
_C2 = (66.0, 10.0, 0.0)
_P1 = (100.0, 0.0, 0.0)


def _bezier_point_2d(t):
    return evaluate_bezier(_P0[:2], _C1[:2], _C2[:2], _P1[:2], t)


def _make_bezier_subpath():
    return [
        MoveToCommand(_P0),
        BezierToCommand(end=_P1, control1=_C1, control2=_C2),
    ]


def _make_mixed_subpath():
    return [
        MoveToCommand((0.0, 0.0, 0.0)),
        LineToCommand((30.0, 0.0, 0.0)),
        BezierToCommand(
            end=(70.0, 0.0, 0.0),
            control1=(40.0, 10.0, 0.0),
            control2=(60.0, 10.0, 0.0),
        ),
        LineToCommand((100.0, 0.0, 0.0)),
    ]


def _make_sectioned_bezier_ops():
    ops = Ops()
    uid = "test-wp"
    ops.commands.append(
        OpsSectionStartCommand(
            section_type=SectionType.VECTOR_OUTLINE,
            workpiece_uid=uid,
        )
    )
    ops.move_to(*_P0)
    ops.commands.append(BezierToCommand(end=_P1, control1=_C1, control2=_C2))
    ops.commands.append(
        OpsSectionEndCommand(section_type=SectionType.VECTOR_OUTLINE)
    )
    return ops


# ------------------------------------------------------------------
# Gap mode tests via _clip_subpath_with_gaps
# ------------------------------------------------------------------


def test_bezier_no_clips_passes_through():
    transformer = TabOpsTransformer()
    cmds = _make_bezier_subpath()
    result = transformer._clip_subpath_with_gaps(cmds, [])

    assert len(result) == len(cmds)
    assert isinstance(result[0], MoveToCommand)
    assert isinstance(result[1], BezierToCommand)
    assert result[1].end == _P1
    assert result[1].control1 == _C1
    assert result[1].control2 == _C2


def test_bezier_gap_splits_into_two_beziers():
    transformer = TabOpsTransformer()
    cmds = _make_bezier_subpath()
    mid = _bezier_point_2d(0.5)
    clip_width = 10.0

    clips = [_ClipPoint(x=mid[0], y=mid[1], width=clip_width)]
    result = transformer._clip_subpath_with_gaps(cmds, clips)

    bezier_cmds = [c for c in result if isinstance(c, BezierToCommand)]
    move_cmds = [c for c in result if isinstance(c, MoveToCommand)]

    assert len(bezier_cmds) == 2, (
        f"Expected 2 bezier segments, got {len(bezier_cmds)}"
    )
    assert len(move_cmds) >= 2

    first_end_x = bezier_cmds[0].end[0]
    assert first_end_x < mid[0]


def test_bezier_gap_preserves_start_and_end():
    transformer = TabOpsTransformer()
    cmds = _make_bezier_subpath()
    mid = _bezier_point_2d(0.5)

    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]
    result = transformer._clip_subpath_with_gaps(cmds, clips)

    first_moving = None
    last_moving = None
    for cmd in result:
        if isinstance(cmd, (MoveToCommand, LineToCommand, BezierToCommand)):
            if cmd.end:
                if first_moving is None:
                    first_moving = cmd.end
                last_moving = cmd.end

    assert first_moving is not None
    assert math.dist(first_moving[:2], _P0[:2]) < 1e-3

    assert last_moving is not None
    assert math.dist(last_moving[:2], _P1[:2]) < 1e-3


def test_bezier_gap_multiple_clips():
    transformer = TabOpsTransformer()
    cmds = _make_bezier_subpath()

    pt25 = _bezier_point_2d(0.25)
    pt75 = _bezier_point_2d(0.75)
    clips = [
        _ClipPoint(x=pt25[0], y=pt25[1], width=6.0),
        _ClipPoint(x=pt75[0], y=pt75[1], width=6.0),
    ]

    result = transformer._clip_subpath_with_gaps(cmds, clips)

    bezier_cmds = [c for c in result if isinstance(c, BezierToCommand)]
    assert len(bezier_cmds) >= 3

    last_end = None
    for c in reversed(result):
        if isinstance(c, (BezierToCommand, LineToCommand)):
            last_end = c.end
            break
    assert last_end is not None
    assert math.dist(last_end[:2], _P1[:2]) < 1e-3


def test_mixed_lines_and_bezier_gap():
    transformer = TabOpsTransformer()
    cmds = _make_mixed_subpath()

    mid = evaluate_bezier(
        (30.0, 0.0), (40.0, 10.0), (60.0, 10.0), (70.0, 0.0), 0.5
    )
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]

    result = transformer._clip_subpath_with_gaps(cmds, clips)

    has_bezier = any(isinstance(c, BezierToCommand) for c in result)
    has_line = any(isinstance(c, LineToCommand) for c in result)
    assert has_bezier
    assert has_line


def test_bezier_gap_via_apply_tab_gaps():
    ops = _make_sectioned_bezier_ops()
    mid = _bezier_point_2d(0.5)
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]

    transformer = TabOpsTransformer()
    transformer._apply_tab_gaps(ops, clips)

    bezier_cmds = [c for c in ops.commands if isinstance(c, BezierToCommand)]
    assert len(bezier_cmds) == 2


# ------------------------------------------------------------------
# Power mode tests via _insert_power_commands
# ------------------------------------------------------------------


def test_bezier_power_mode_splits_and_inserts_power():
    cmds = _make_bezier_subpath()
    mid = _bezier_point_2d(0.5)
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]
    tab_power = 0.3
    original_power = 1.0

    transformer = TabOpsTransformer()
    result = transformer._insert_power_commands(
        cmds, clips, tab_power, original_power
    )

    bezier_cmds = [c for c in result if isinstance(c, BezierToCommand)]
    power_cmds = [c for c in result if isinstance(c, SetPowerCommand)]

    assert len(bezier_cmds) >= 2
    assert len(power_cmds) >= 2

    powers = [c.power for c in power_cmds]
    assert tab_power in powers
    assert original_power in powers


def test_bezier_power_mode_no_overlap():
    cmds = _make_bezier_subpath()
    clips = [_ClipPoint(x=200.0, y=200.0, width=10.0)]
    tab_power = 0.3
    original_power = 1.0

    transformer = TabOpsTransformer()
    result = transformer._insert_power_commands(
        cmds, clips, tab_power, original_power
    )

    bezier_cmds = [c for c in result if isinstance(c, BezierToCommand)]
    assert len(bezier_cmds) == 1
    assert bezier_cmds[0].end == _P1
    assert bezier_cmds[0].control1 == _C1
    assert bezier_cmds[0].control2 == _C2


def test_mixed_lines_and_bezier_power():
    cmds = _make_mixed_subpath()

    mid = evaluate_bezier(
        (30.0, 0.0), (40.0, 10.0), (60.0, 10.0), (70.0, 0.0), 0.5
    )
    clips = [_ClipPoint(x=mid[0], y=mid[1], width=10.0)]
    tab_power = 0.3
    original_power = 1.0

    transformer = TabOpsTransformer()
    result = transformer._insert_power_commands(
        cmds, clips, tab_power, original_power
    )

    bezier_cmds = [c for c in result if isinstance(c, BezierToCommand)]
    power_cmds = [c for c in result if isinstance(c, SetPowerCommand)]

    assert len(bezier_cmds) >= 1
    assert len(power_cmds) >= 1


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
    ops.commands.append(
        OpsSectionStartCommand(
            section_type=SectionType.VECTOR_OUTLINE,
            workpiece_uid="test",
        )
    )
    ops.move_to(0, 0)
    ops.line_to(100, 0)
    ops.commands.append(
        OpsSectionEndCommand(section_type=SectionType.VECTOR_OUTLINE)
    )

    clips = [_ClipPoint(x=50.0, y=0.0, width=10.0)]

    transformer = TabOpsTransformer()
    transformer._apply_tab_gaps(ops, clips)

    line_cmds = [c for c in ops.commands if isinstance(c, LineToCommand)]
    move_cmds = [c for c in ops.commands if isinstance(c, MoveToCommand)]

    assert len(line_cmds) == 2
    assert len(move_cmds) >= 2

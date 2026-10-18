import pytest
from sketcher.core.constraints import (
    CoincidentConstraint,
    HorizontalConstraint,
    PointOnLineConstraint,
    VerticalConstraint,
)
from sketcher.core.entities import Line, Point
from sketcher.core.snap import SnapLineType
from sketcher.core.snap.types import SnapLine, SnapPoint, SnapResult
from sketcher.ui_gtk.tools.snap_mixin import SnapMixin


class Harness(SnapMixin):
    def __init__(self):
        self.current_snap_result = None


def axis_lines(source_pid):
    src = Point(source_pid, 0.0, 0.0)
    return [
        SnapLine(
            is_horizontal=True,
            coordinate=0.0,
            line_type=SnapLineType.ENTITY_POINT,
            source=src,
        ),
        SnapLine(
            is_horizontal=False,
            coordinate=0.0,
            line_type=SnapLineType.ENTITY_POINT,
            source=src,
        ),
    ]


def both_axes_result(source_pid):
    return SnapResult(
        snapped=True,
        position=(0.0, 0.0),
        snap_lines=axis_lines(source_pid),
        distance=1.0,
    )


def point_pairs(constraints, cls):
    return [{c.p1, c.p2} for c in constraints if isinstance(c, cls)]


def test_both_axis_snaps_onto_same_point_become_coincident():
    """A horizontal and a vertical axis snap onto the same point pin
    the endpoint onto it completely: one coincidence constraint is
    the faithful encoding, not two orientation constraints."""
    harness = Harness()

    constraints = harness._build_axis_constraints(
        both_axes_result(source_pid=7), end_pid=3, existing=[]
    )

    assert point_pairs(constraints, CoincidentConstraint) == [{3, 7}]
    assert point_pairs(constraints, HorizontalConstraint) == []
    assert point_pairs(constraints, VerticalConstraint) == []


def test_single_axis_snap_still_becomes_orientation_constraint():
    harness = Harness()
    result = both_axes_result(source_pid=7)
    result.snap_lines = result.snap_lines[:1]

    constraints = harness._build_axis_constraints(
        result, end_pid=3, existing=[]
    )

    assert point_pairs(constraints, HorizontalConstraint) == [{3, 7}]
    assert point_pairs(constraints, CoincidentConstraint) == []


def test_axis_constraints_skip_existing_pair():
    """If the sketch already expresses one axis of the pair, only the
    missing axis is added — the pair is not duplicated."""
    harness = Harness()

    constraints = harness._build_axis_constraints(
        both_axes_result(source_pid=7),
        end_pid=3,
        existing=[HorizontalConstraint(7, 3)],
    )

    assert point_pairs(constraints, VerticalConstraint) == [{3, 7}]
    assert point_pairs(constraints, CoincidentConstraint) == []

    constraints = harness._build_axis_constraints(
        both_axes_result(source_pid=7),
        end_pid=3,
        existing=[
            HorizontalConstraint(7, 3),
            VerticalConstraint(7, 3),
        ],
    )

    assert constraints == []


def test_axis_constraints_skip_primary_snap_point():
    """The point an ENTITY_POINT snap already pins the endpoint to
    needs no axis constraints on top of its coincidence."""
    harness = Harness()
    result = both_axes_result(source_pid=7)
    result.primary_snap_point = SnapPoint(
        x=0.0,
        y=0.0,
        line_type=SnapLineType.ENTITY_POINT,
        source=Point(7, 0.0, 0.0),
    )

    constraints = harness._build_axis_constraints(
        result, end_pid=3, existing=[]
    )

    assert constraints == []


def test_endpoint_snap_onto_segment_creates_coincident_not_trio():
    """Closing a loop by dropping the endpoint onto a segment whose
    endpoint is axis-snapped yields a point-on-line plus ONE
    coincidence — not a point-on-line with a horizontal and a
    vertical constraint stacked on top."""
    harness = Harness()
    line = Line(31, 27, 30)
    result = SnapResult(
        snapped=True,
        position=(0.0, 0.0),
        snap_lines=axis_lines(27),
        distance=1.0,
    )
    result.primary_snap_point = SnapPoint(
        x=0.5,
        y=0.0,
        line_type=SnapLineType.ON_ENTITY,
        source=line,
    )
    harness.current_snap_result = result

    constraints = harness.build_snap_constraints(55, end_pid=3)

    on_line = [c for c in constraints if isinstance(c, PointOnLineConstraint)]
    assert len(on_line) == 1
    assert on_line[0].point_id == 3
    assert on_line[0].shape_id == 31
    assert point_pairs(constraints, CoincidentConstraint) == [{3, 27}]
    assert point_pairs(constraints, HorizontalConstraint) == []
    assert point_pairs(constraints, VerticalConstraint) == []


def test_existing_coincident_suppresses_pair():
    harness = Harness()

    constraints = harness._build_axis_constraints(
        both_axes_result(source_pid=7),
        end_pid=3,
        existing=[CoincidentConstraint(p1=7, p2=3)],
    )

    assert constraints == []


@pytest.mark.parametrize("source_pid,end_pid", [(7, 3), (3, 7)])
def test_pair_key_is_order_independent(source_pid, end_pid):
    harness = Harness()

    constraints = harness._build_axis_constraints(
        both_axes_result(source_pid=source_pid),
        end_pid=end_pid,
        existing=[CoincidentConstraint(p1=3, p2=7)],
    )

    assert constraints == []

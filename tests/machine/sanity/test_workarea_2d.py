from rayforge.machine.sanity import (
    IssueCategory,
    IssueSeverity,
    SanityContext,
)
from rayforge.machine.sanity.checks.workarea_2d import WorkareaCheck2D


def _run_check(ops, machine):
    ctx = SanityContext(
        ops=ops,
        machine=machine,
        work_area=machine.work_area,
        axis_extents=machine.axis_extents,
        enabled_zones={},
    )
    return WorkareaCheck2D().run(ctx)


def test_within_workarea(isolated_machine, make_line_ops):
    isolated_machine.set_axis_extents(200, 200)
    isolated_machine.set_work_margins(10, 10, 10, 10)
    ops = make_line_ops([(20, 20, False), (100, 100, True)])
    assert len(_run_check(ops, isolated_machine)) == 0


def test_x_below_workarea(isolated_machine, make_line_ops):
    isolated_machine.set_axis_extents(200, 200)
    isolated_machine.set_work_margins(50, 10, 10, 10)
    ops = make_line_ops([(10, 50, False), (10, 50, True)])
    issues = _run_check(ops, isolated_machine)
    assert len(issues) == 1
    assert issues[0].category == IssueCategory.WORKAREA
    assert issues[0].severity == IssueSeverity.WARNING
    assert issues[0].message.startswith("X=")
    assert "<" in issues[0].message


def test_y_above_workarea(isolated_machine, make_line_ops):
    isolated_machine.set_axis_extents(200, 200)
    isolated_machine.set_work_margins(10, 10, 10, 50)
    ops = make_line_ops([(50, 150, False), (50, 160, True)])
    issues = _run_check(ops, isolated_machine)
    assert len(issues) == 1
    assert issues[0].message.startswith("Y=")
    assert ">" in issues[0].message


def test_no_margins_entire_extent_is_workarea(isolated_machine, make_line_ops):
    isolated_machine.set_axis_extents(200, 200)
    ops = make_line_ops([(0, 0, False), (200, 200, True)])
    assert len(_run_check(ops, isolated_machine)) == 0


def test_deduplicated_across_commands(isolated_machine, make_line_ops):
    isolated_machine.set_axis_extents(200, 200)
    isolated_machine.set_work_margins(50, 50, 50, 50)
    ops = make_line_ops(
        [
            (10, 10, False),
            (10, 10, True),
            (5, 5, True),
        ]
    )
    issues = _run_check(ops, isolated_machine)
    axes = {i.message[0] for i in issues}
    assert "X" in axes
    assert "Y" in axes

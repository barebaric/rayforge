from rayforge.core.ops import Ops
from rayforge.machine.sanity import IssueCategory, IssueSeverity, SanityContext
from rayforge.machine.sanity.checks.extent_2d import ExtentCheck2D


def _run_check(ops, axis_extents, machine):
    ctx = SanityContext(
        ops=ops,
        machine=machine,
        work_area=machine.work_area,
        axis_extents=axis_extents,
        enabled_zones={},
    )
    return ExtentCheck2D().run(ctx)


def test_all_within_extent(isolated_machine):
    isolated_machine.set_axis_extents(200, 200)
    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(50, 50)
    ops.line_to(100, 100)
    assert len(_run_check(ops, (200, 200), isolated_machine)) == 0


def test_x_exceeds_max(isolated_machine):
    isolated_machine.set_axis_extents(200, 200)
    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(250, 10)
    issues = _run_check(ops, (200, 200), isolated_machine)
    assert len(issues) == 1
    assert issues[0].category == IssueCategory.MACHINE_EXTENT
    assert issues[0].severity == IssueSeverity.ERROR
    assert issues[0].message.startswith("X=")


def test_y_negative(isolated_machine):
    isolated_machine.set_axis_extents(200, 200)
    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(10, -5)
    issues = _run_check(ops, (200, 200), isolated_machine)
    assert len(issues) == 1
    assert issues[0].category == IssueCategory.MACHINE_EXTENT
    assert issues[0].message.startswith("Y=")


def test_empty_ops(isolated_machine):
    ops = Ops()
    assert len(_run_check(ops, (200, 200), isolated_machine)) == 0


def test_exact_boundary_is_ok(isolated_machine):
    isolated_machine.set_axis_extents(200, 150)
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(200, 150)
    assert len(_run_check(ops, (200, 150), isolated_machine)) == 0


def test_deduplicated_multiple_violations(isolated_machine):
    isolated_machine.set_axis_extents(200, 200)
    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(250, 250)
    ops.line_to(300, -10)
    issues = _run_check(ops, (200, 200), isolated_machine)
    axes = {i.message[0] for i in issues}
    assert "X" in axes
    assert "Y" in axes
    assert len(issues) == 3

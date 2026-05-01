from rayforge.core.ops import Ops
from rayforge.machine.models.zone import Zone, ZoneShape
from rayforge.machine.sanity import (
    IssueCategory,
    IssueSeverity,
    SanityContext,
)
from rayforge.machine.sanity.checks.nogo_zones_2d import NoGoZoneCheck2D


def _run_check(ops, machine, zones):
    ctx = SanityContext(
        ops=ops,
        machine=machine,
        work_area=machine.work_area,
        axis_extents=machine.axis_extents,
        enabled_zones=zones,
    )
    return NoGoZoneCheck2D().run(ctx)


def test_no_zones(isolated_machine, make_line_ops):
    ops = make_line_ops([(10, 10, False), (20, 20, True)])
    assert len(_run_check(ops, isolated_machine, {})) == 0


def test_line_hits_rect_zone(isolated_machine, make_line_ops, make_rect_zone):
    zone = make_rect_zone(5, 5, 10, 10, "Clamp")
    ops = make_line_ops([(0, 10, False), (20, 10, True)])
    issues = _run_check(ops, isolated_machine, {"z1": zone})
    assert len(issues) == 1
    assert issues[0].zone_name == "Clamp"
    assert issues[0].zone_uid == "z1"
    assert issues[0].category == IssueCategory.NOGO_ZONE
    assert issues[0].severity == IssueSeverity.ERROR


def test_line_misses_rect_zone(
    isolated_machine, make_line_ops, make_rect_zone
):
    zone = make_rect_zone(50, 50, 10, 10)
    ops = make_line_ops([(0, 0, False), (10, 10, True)])
    assert len(_run_check(ops, isolated_machine, {"z1": zone})) == 0


def test_travel_triggers(isolated_machine, make_rect_zone):
    zone = make_rect_zone(0, 0, 20, 20)
    ops = Ops()
    ops.move_to(0, 0)
    ops.move_to(50, 50)
    assert len(_run_check(ops, isolated_machine, {"z1": zone})) == 1


def test_arc_hits_zone(isolated_machine):
    zone = Zone()
    zone.shape = ZoneShape.CYLINDER
    zone.params = {
        "x": 7,
        "y": 7,
        "z": 0,
        "radius": 5,
        "height": 10,
    }
    ops = Ops()
    ops.move_to(10, 0)
    ops.arc_to(0, 10, -10, 0, False)
    assert len(_run_check(ops, isolated_machine, {"z1": zone})) == 1


def test_multiple_zones_hit(isolated_machine, make_line_ops, make_rect_zone):
    z1 = make_rect_zone(5, 5, 5, 5, "Zone A")
    z2 = make_rect_zone(15, 15, 5, 5, "Zone B")
    ops = make_line_ops([(0, 0, False), (20, 20, True)])
    issues = _run_check(ops, isolated_machine, {"z1": z1, "z2": z2})
    assert len(issues) == 2
    hit_uids = {i.zone_uid for i in issues}
    assert hit_uids == {"z1", "z2"}


def test_one_issue_per_zone(isolated_machine, make_rect_zone):
    zone = make_rect_zone(0, 0, 50, 50, "Big Zone")
    ops = Ops()
    ops.move_to(10, 10)
    ops.line_to(30, 30)
    ops.line_to(10, 10)
    ops.line_to(40, 40)
    issues = _run_check(ops, isolated_machine, {"z1": zone})
    assert len(issues) == 1


def test_disabled_zones_excluded(isolated_machine, make_line_ops):
    ops = make_line_ops([(0, 10, False), (20, 10, True)])
    assert len(_run_check(ops, isolated_machine, {})) == 0


def test_empty_ops(isolated_machine, make_rect_zone):
    zone = make_rect_zone(0, 0, 10, 10)
    ops = Ops()
    assert len(_run_check(ops, isolated_machine, {"z1": zone})) == 0

import pytest

from rayforge.machine.sanity import (
    CheckMode,
    IssueCategory,
    SanityChecker,
)
from rayforge.pipeline.coordspace import WorkspaceOrientation


def test_clean_job(isolated_machine, make_line_ops):
    isolated_machine.set_axis_extents(200, 200)
    ops = make_line_ops([(10, 10, False), (50, 50, True), (100, 100, True)])
    report = SanityChecker(isolated_machine).check(ops, mode=CheckMode.FAST)
    assert report.is_clean
    assert report.mode == CheckMode.FAST


def test_extent_violation_reported(isolated_machine, make_line_ops):
    isolated_machine.set_axis_extents(200, 200)
    ops = make_line_ops([(10, 10, False), (250, 10, True)])
    report = SanityChecker(isolated_machine).check(ops)
    assert report.has_errors
    categories = {i.category for i in report.issues}
    assert IssueCategory.MACHINE_EXTENT in categories


def test_zone_violation_reported(
    isolated_machine, make_line_ops, make_rect_zone
):
    isolated_machine.set_axis_extents(200, 200)
    zone = make_rect_zone(5, 5, 10, 10, "Clamp")
    isolated_machine.add_nogo_zone(zone)
    ops = make_line_ops([(0, 10, False), (20, 10, True)])
    report = SanityChecker(isolated_machine).check(ops)
    assert report.has_errors
    zone_issues = [
        i for i in report.issues if i.category == IssueCategory.NOGO_ZONE
    ]
    assert len(zone_issues) == 1


@pytest.mark.parametrize(
    "orientation, points",
    [
        (
            WorkspaceOrientation.ROTATED_LEFT,
            [(130, 25, False), (190, 25, True)],
        ),
        (
            WorkspaceOrientation.ROTATED_RIGHT,
            [(10, 75, False), (70, 75, True)],
        ),
    ],
)
def test_rotated_zone_violation_reported_at_workspace_location(
    isolated_machine, make_line_ops, make_rect_zone, orientation, points
):
    """Safety checks project physical no-go zones with the workspace."""
    isolated_machine.set_axis_extents(100, 200)
    zone = make_rect_zone(10, 20, 30, 40, "Clamp")
    isolated_machine.add_nogo_zone(zone)
    isolated_machine.set_workspace_orientation(orientation)

    report = SanityChecker(isolated_machine).check(make_line_ops(points))

    zone_issues = [
        issue
        for issue in report.issues
        if issue.category == IssueCategory.NOGO_ZONE
    ]
    assert len(zone_issues) == 1
    assert zone.params == {"x": 10, "y": 20, "w": 30, "h": 40}


def test_zone_projection_is_detached_and_always_reflects_native_state(
    isolated_machine, make_rect_zone
):
    isolated_machine.set_axis_extents(100, 200)
    zone = make_rect_zone(10, 20, 30, 40, "Clamp")
    isolated_machine.add_nogo_zone(zone)

    native = isolated_machine.workspace.nogo_zones
    assert native[zone.uid] is not zone
    assert native[zone.uid].params == zone.params

    isolated_machine.set_workspace_orientation(
        WorkspaceOrientation.ROTATED_RIGHT
    )

    first = isolated_machine.workspace.nogo_zones
    assert first[zone.uid] is not zone
    assert isolated_machine.workspace.nogo_zones is not first

    zone.set_param("x", 20)
    second = isolated_machine.workspace.nogo_zones
    assert second is not first
    assert second[zone.uid].params["y"] == pytest.approx(50)

    isolated_machine.set_axis_extents(200, 200)
    third = isolated_machine.workspace.nogo_zones
    assert third is not second
    assert third[zone.uid].params["y"] == pytest.approx(150)


def test_workarea_violation_reported(isolated_machine, make_line_ops):
    isolated_machine.set_axis_extents(200, 200)
    isolated_machine.set_work_margins(50, 50, 50, 50)
    ops = make_line_ops([(10, 10, False), (60, 60, True)])
    report = SanityChecker(isolated_machine).check(ops)
    assert report.has_warnings
    wa_issues = [
        i for i in report.issues if i.category == IssueCategory.WORKAREA
    ]
    assert len(wa_issues) >= 1


def test_multiple_issue_categories(
    isolated_machine, make_line_ops, make_rect_zone
):
    isolated_machine.set_axis_extents(100, 100)
    isolated_machine.set_work_margins(10, 10, 10, 10)
    zone = make_rect_zone(5, 5, 10, 10)
    isolated_machine.add_nogo_zone(zone)
    ops = make_line_ops([(0, 0, False), (150, 50, True)])
    report = SanityChecker(isolated_machine).check(ops)
    categories = {i.category for i in report.issues}
    assert IssueCategory.NOGO_ZONE in categories
    assert IssueCategory.MACHINE_EXTENT in categories


def test_disabled_zone_not_included(
    isolated_machine, make_line_ops, make_rect_zone
):
    isolated_machine.set_axis_extents(200, 200)
    zone = make_rect_zone(5, 5, 10, 10)
    zone.enabled = False
    isolated_machine.add_nogo_zone(zone)
    ops = make_line_ops([(0, 10, False), (20, 10, True)])
    report = SanityChecker(isolated_machine).check(ops)
    zone_issues = [
        i for i in report.issues if i.category == IssueCategory.NOGO_ZONE
    ]
    assert len(zone_issues) == 0

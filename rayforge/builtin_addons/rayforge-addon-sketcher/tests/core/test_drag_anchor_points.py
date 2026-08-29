import math

import pytest
from sketcher.core.constraints import (
    DistanceConstraint,
    DragConstraint,
    PerpendicularConstraint,
)
from sketcher.core.entities import Arc, Circle, Ellipse, Line
from sketcher.core.params import ParameterContext
from sketcher.core.registry import EntityRegistry
from sketcher.core.solver import Solver


def build_ellipse(reg: EntityRegistry):
    """Ellipse with helper lines and a perpendicular constraint, like
    the ellipse creation command builds it."""
    c = reg.add_point(0, 0)
    r1 = reg.add_point(10, 0)
    r2 = reg.add_point(0, 5)
    l1 = reg.add_line(c, r1)
    l2 = reg.add_line(c, r2)
    ellipse_id = reg.add_ellipse(c, r1, r2)
    ellipse = reg.get_entity(ellipse_id)
    assert isinstance(ellipse, Ellipse)
    ellipse.helper_line_ids = [l1, l2]
    return c, r1, r2, l1, l2, ellipse


def test_line_has_no_drag_anchor_points():
    reg = EntityRegistry()
    p1 = reg.add_point(0, 0)
    p2 = reg.add_point(10, 0)
    line = reg.get_entity(reg.add_line(p1, p2))
    assert isinstance(line, Line)
    assert line.get_drag_anchor_points(p1) == []
    assert line.get_drag_anchor_points(p2) == []


def test_ellipse_drag_anchor_points():
    reg = EntityRegistry()
    c, r1, r2, _l1, _l2, ellipse = build_ellipse(reg)
    assert ellipse.get_drag_anchor_points(r1) == [c]
    assert ellipse.get_drag_anchor_points(r2) == [c]
    assert ellipse.get_drag_anchor_points(c) == []
    assert ellipse.get_drag_anchor_points(9999) == []


def test_circle_drag_anchor_points():
    reg = EntityRegistry()
    c = reg.add_point(0, 0)
    rim = reg.add_point(5, 0)
    circle = reg.get_entity(reg.add_circle(c, rim))
    assert isinstance(circle, Circle)
    assert circle.get_drag_anchor_points(rim) == [c]
    assert circle.get_drag_anchor_points(c) == []
    assert circle.get_drag_anchor_points(9999) == []


def test_arc_drag_anchor_points():
    reg = EntityRegistry()
    c = reg.add_point(0, 0)
    start = reg.add_point(5, 0)
    end = reg.add_point(0, 5)
    arc = reg.get_entity(reg.add_arc(start, end, c))
    assert isinstance(arc, Arc)
    assert arc.get_drag_anchor_points(start) == [c]
    assert arc.get_drag_anchor_points(end) == [c]
    assert arc.get_drag_anchor_points(c) == []
    assert arc.get_drag_anchor_points(9999) == []


def test_registry_collects_drag_anchor_points():
    reg = EntityRegistry()
    c, r1, r2, _l1, _l2, _ellipse = build_ellipse(reg)
    assert reg.get_drag_anchor_points(r1) == [c]
    assert reg.get_drag_anchor_points(r2) == [c]
    assert reg.get_drag_anchor_points(c) == []


def build_rotation_drag_constraints(
    reg: EntityRegistry, center_weight: float | None
):
    """Constraints for dragging the ellipse's rx point from (10, 0) to
    (15, 0). The distance constraint locks the radius to 10, so the
    solver can only reach the target by moving the center - unless the
    center is pinned."""
    c, r1, r2, l1, l2, _ellipse = build_ellipse(reg)
    params = ParameterContext()
    constraints = [
        PerpendicularConstraint(l1, l2),
        DistanceConstraint(c, r1, 10.0),
    ]
    if center_weight is not None:
        constraints.append(DragConstraint(c, 0.0, 0.0, weight=center_weight))
    constraints.extend(
        [
            DragConstraint(r2, 0.0, 5.0, weight=0.01),
            DragConstraint(r1, 15.0, 0.0, weight=0.1),
        ]
    )
    solver = Solver(reg, params, constraints)
    solver.solve()
    return reg.get_point(c), reg.get_point(r1)


def test_radius_drag_without_pin_drifts_the_center():
    reg = EntityRegistry()
    center, _r1 = build_rotation_drag_constraints(reg, center_weight=0.01)
    # With only a weak hold, the solver prefers drifting the center
    # towards the target over falling short of it.
    assert center.x > 1.0


def test_radius_drag_with_pin_rotates_around_center():
    reg = EntityRegistry()
    center, r1 = build_rotation_drag_constraints(reg, center_weight=1.0)
    assert center.x == pytest.approx(0.0, abs=0.2)
    assert center.y == pytest.approx(0.0, abs=0.2)
    # The radius point still moves towards the target as far as the
    # pinned center allows. The weight-1 distance constraint bends by
    # a fraction of a unit against the 0.1-weighted drag.
    radius = math.hypot(r1.x - center.x, r1.y - center.y)
    assert radius == pytest.approx(10.0, abs=0.1)
    assert 10.0 <= r1.x < 11.0

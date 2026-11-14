"""Build raygeo constraints from entity references.

raygeo constraints are value objects over resolved point ids; these
helpers resolve entity references through the sketch registry so tests
can express constraints the way the sketcher UI does.
"""

from __future__ import annotations

import math

from raygeo.sketch.constraints.angle import AngleConstraint
from raygeo.sketch.constraints.diameter import DiameterConstraint
from raygeo.sketch.constraints.equal_length import EqualLengthConstraint
from raygeo.sketch.constraints.perpendicular import (
    PerpendicularConstraint,
    PerpendicularTarget,
)
from raygeo.sketch.constraints.point_on_curve import PointOnCurveConstraint
from raygeo.sketch.constraints.radius import RadiusConstraint
from raygeo.sketch.constraints.tangent import TangentConstraint


def _reg(sketch):
    return getattr(sketch, "registry", sketch)


def _radius_point(shape):
    if shape.type_name() == "circle":
        return shape.radius_pt_idx
    return shape.start_idx


def tangent_from_entities(sketch, line_id, shape_id):
    reg = _reg(sketch)
    line = reg.get_entity_object(line_id)
    shape = reg.get_entity_object(shape_id)
    return TangentConstraint(
        line.p1_idx, line.p2_idx, shape.center_idx, _radius_point(shape)
    )


def angle_from_lines(sketch, l1, l2, value, expression=None):
    reg = _reg(sketch)
    a = reg.get_entity_object(l1)
    b = reg.get_entity_object(l2)
    constraint = AngleConstraint(a.p1_idx, a.p2_idx, b.p1_idx, b.p2_idx, value)
    if expression is not None:
        constraint.expression = expression
    return constraint


def radius_from_entity(sketch, arc_id, value):
    reg = _reg(sketch)
    arc = reg.get_entity_object(arc_id)
    return RadiusConstraint(arc.center_idx, _radius_point(arc), value)


def diameter_from_entity(sketch, circle_id, value):
    reg = _reg(sketch)
    circle = reg.get_entity_object(circle_id)
    return DiameterConstraint(circle.center_idx, circle.radius_pt_idx, value)


def perpendicular_from_lines(sketch, l1, l2):
    reg = _reg(sketch)
    a = reg.get_entity_object(l1)
    b = reg.get_entity_object(l2)
    return PerpendicularConstraint(
        PerpendicularTarget.LineLine(a.p1_idx, a.p2_idx, b.p1_idx, b.p2_idx)
    )


def equal_length_from_entities(sketch, entity_ids):
    reg = _reg(sketch)
    entities = [reg.get_entity_object(eid) for eid in entity_ids]
    base = entities[0]
    pairs = [
        (base.p1_idx, base.p2_idx, e.p1_idx, e.p2_idx) for e in entities[1:]
    ]
    return EqualLengthConstraint(pairs)


def add_arc_by_angle(sketch, center_pid, radius_pid, start_deg, end_deg):
    reg = _reg(sketch)
    c = reg.get_point(center_pid)
    r = reg.get_point(radius_pid)
    radius = math.hypot(r[0] - c[0], r[1] - c[1])
    a0 = math.radians(start_deg)
    a1 = math.radians(end_deg)
    start = sketch.add_point(
        c[0] + radius * math.cos(a0), c[1] + radius * math.sin(a0)
    )
    end = sketch.add_point(
        c[0] + radius * math.cos(a1), c[1] + radius * math.sin(a1)
    )
    return sketch.add_arc(start, end, center_pid)


def point_on_curve_from_entity(sketch, point_pid, bezier_id):
    reg = _reg(sketch)
    b = reg.get_entity_object(bezier_id)
    constraint = PointOnCurveConstraint(
        point_pid,
        b.start_idx,
        b.end_idx,
        b.cp1_offset or (0.0, 0.0),
        b.cp2_offset or (0.0, 0.0),
        b.is_line,
    )
    constraint.shape_id = bezier_id
    return constraint

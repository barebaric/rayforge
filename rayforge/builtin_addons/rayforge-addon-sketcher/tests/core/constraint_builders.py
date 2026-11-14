"""Build raygeo constraints from entity references.

raygeo constraints are value objects over resolved point ids; these
helpers resolve entity references through the sketch registry so tests
can express constraints the way the sketcher UI does.
"""

from __future__ import annotations

from raygeo.sketch.constraints.angle import AngleConstraint
from raygeo.sketch.constraints.diameter import DiameterConstraint
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
    constraint = AngleConstraint(
        a.p1_idx, a.p2_idx, b.p1_idx, b.p2_idx, value
    )
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
    return DiameterConstraint(
        circle.center_idx, circle.radius_pt_idx, value
    )

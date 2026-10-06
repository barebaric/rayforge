import cairo
import pytest
from gi.repository import PangoCairo
from sketcher.core.constraints import (
    AngleConstraint,
    DiameterConstraint,
    DistanceConstraint,
    EqualDistanceConstraint,
    EqualLengthConstraint,
    RadiusConstraint,
)
from sketcher.core.constraints.base import ConstraintStatus
from sketcher.core.registry import EntityRegistry


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("distance", "10.0"),
        ("angle", "90.0\u00b0"),
        ("radius", "R10.0"),
        ("diameter", "\u00d810.0"),
        ("equal_length", "="),
        ("equal_distance", "="),
    ],
)
@pytest.mark.parametrize("status", list(ConstraintStatus))
@pytest.mark.parametrize("selected, hovered", [(False, False), (True, True)])
def test_constraint_text_uses_pango(
    mocker, kind, expected, status, selected, hovered
):
    registry = EntityRegistry()
    center = registry.add_point(0, 0)
    right = registry.add_point(10, 0)
    up = registry.add_point(0, 10)
    line1 = registry.add_line(center, right)
    line2 = registry.add_line(center, up)
    circle = registry.add_circle(center, right)
    constraints = {
        "distance": DistanceConstraint(center, right, 10),
        "angle": AngleConstraint(line1, line2, 90),
        "radius": RadiusConstraint(circle, 10),
        "diameter": DiameterConstraint(circle, 10),
        "equal_length": EqualLengthConstraint([line1, line2]),
        "equal_distance": EqualDistanceConstraint(center, right, center, up),
    }
    constraint = constraints[kind]
    constraint.status = status
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 200, 200)
    ctx = cairo.Context(surface)
    show_layout = mocker.spy(PangoCairo, "show_layout")

    constraint.draw(
        ctx,
        registry,
        lambda point: (100 + point[0], 100 + point[1]),
        is_selected=selected,
        is_hovered=hovered,
    )

    assert show_layout.call_count == (2 if kind == "equal_length" else 1)
    for call in show_layout.call_args_list:
        assert call.args[1].get_text() == expected
    assert any(surface.get_data())

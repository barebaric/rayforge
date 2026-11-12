from unittest.mock import MagicMock

import pytest
from raygeo.geo.shape.text import FontConfig
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

LABEL_FONT_PX = 12.0
PT_TO_MM = 25.4 / 72.0


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("distance", "10.0"),
        ("angle", "90.0°"),
        ("radius", "R10.0"),
        ("diameter", "Ø10.0"),
        ("equal_length", "="),
        ("equal_distance", "="),
    ],
)
@pytest.mark.parametrize("status", list(ConstraintStatus))
@pytest.mark.parametrize("selected, hovered", [(False, False), (True, True)])
def test_constraint_text_rendering(kind, expected, status, selected, hovered):
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
    ctx = MagicMock()

    constraint.draw(
        ctx,
        registry,
        lambda point: (100 + point[0], 100 + point[1]),
        is_selected=selected,
        is_hovered=hovered,
    )

    assert [call for call in ctx.method_calls if call[0] == "stroke"]
    labeled = kind not in ("equal_length", "equal_distance")
    rectangles = [call for call in ctx.method_calls if call[0] == "rectangle"]
    assert bool(rectangles) == labeled
    if labeled:
        config = FontConfig("sans-serif", LABEL_FONT_PX / PT_TO_MM)
        width = config.get_text_width(expected)
        assert rectangles[0].args[2] == pytest.approx(width + 8.0)

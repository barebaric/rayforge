"""Regression tests for sketch symbol drawing."""

import cairo
from raygeo.sketch.constraints.equal_distance import EqualDistanceConstraint
from raygeo.sketch.constraints.perpendicular import PerpendicularConstraint
from raygeo.sketch.registry import EntityRegistry


def test_text_labels_do_not_connect_to_the_next_arc():
    """A text label must not leave its current point behind.

    Cairo's show_text leaves the current point set. If the next
    constraint draws an arc without starting a clean path, Cairo joins
    the label position to the arc with a line, producing a stray line
    from an equals label to a distant perpendicular marker.
    """
    registry = EntityRegistry()
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)
    p3 = registry.add_point(0, 10)
    p4 = registry.add_point(10, 10)

    a1 = registry.add_point(90, 100)
    a2 = registry.add_point(110, 100)
    b1 = registry.add_point(100, 90)
    b2 = registry.add_point(100, 110)
    line_a = registry.add_line(a1, a2)
    line_b = registry.add_line(b1, b2)

    equal_distance = EqualDistanceConstraint(p1, p2, p3, p4)
    perpendicular = PerpendicularConstraint(line_a, line_b)

    def to_screen(point):
        return (point[0] * 5 + 100, 1000 - point[1] * 5)

    width, height = 800, 1000
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.paint()

    equal_distance.draw(ctx, registry, to_screen)
    perpendicular.draw(ctx, registry, to_screen)

    surface.flush()
    data = surface.get_data()
    stride = surface.get_stride()

    stray = 0
    for y in range(550, 901):
        row = data[y * stride : (y + 1) * stride]
        for x in range(200, 551):
            red, green, blue, alpha = row[x * 4 : x * 4 + 4]
            if (
                alpha
                and green > 80
                and green > red * 1.6
                and green > blue * 1.6
            ):
                stray += 1

    assert stray == 0

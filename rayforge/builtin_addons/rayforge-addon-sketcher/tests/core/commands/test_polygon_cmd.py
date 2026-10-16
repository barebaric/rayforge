import math

import pytest
from sketcher.core import Sketch
from sketcher.core.commands import (
    PolygonCommand,
    PolygonPreviewState,
    polygon_vertices,
)
from sketcher.core.constraints import (
    AspectRatioConstraint,
    EqualLengthConstraint,
    PointOnLineConstraint,
    RadiusConstraint,
)
from sketcher.core.entities import Circle, Line


def _lines(sketch):
    return [
        e
        for e in sketch.registry.entities
        if isinstance(e, Line) and not e.construction
    ]


def _circles(sketch):
    return [e for e in sketch.registry.entities if isinstance(e, Circle)]


def _corner_positions(sketch):
    registry = sketch.registry
    return [
        (registry.get_point(line.p1_idx).x, registry.get_point(line.p1_idx).y)
        for line in _lines(sketch)
    ]


def _execute(sides=6, star_ratio=None, fixed_radius=None, end=(10.0, 0.0)):
    sketch = Sketch()
    center = sketch.add_point(0, 0)
    cmd = PolygonCommand(
        sketch,
        center,
        end,
        sides=sides,
        star_ratio=star_ratio,
        fixed_radius=fixed_radius,
    )
    cmd.execute()
    return sketch, center, cmd


class TestPolygonVertices:
    def test_regular_polygon(self):
        vertices = polygon_vertices((0, 0), (10, 0), 4)
        expected = [(10, 0), (0, 10), (-10, 0), (0, -10)]
        for (x, y), (ex, ey) in zip(vertices, expected):
            assert x == pytest.approx(ex, abs=1e-9)
            assert y == pytest.approx(ey, abs=1e-9)

    def test_first_vertex_follows_the_drag_point(self):
        x, y = polygon_vertices((1, 1), (1, 6), 5)[0]
        assert (x, y) == pytest.approx((1, 6))

    def test_star_alternates_outer_and_inner(self):
        vertices = polygon_vertices((0, 0), (10, 0), 5, star_ratio=0.5)
        assert len(vertices) == 10
        radii = [math.hypot(x, y) for x, y in vertices]
        assert radii[0::2] == pytest.approx([10] * 5)
        assert radii[1::2] == pytest.approx([5] * 5)
        assert math.degrees(math.atan2(*vertices[1][::-1])) == pytest.approx(
            36
        )


class TestPolygonCommand:
    def test_creates_plain_points_and_lines(self):
        sketch, _center, _cmd = _execute(sides=6)
        lines = _lines(sketch)
        assert len(lines) == 6
        assert len(sketch.registry.points) == 1 + 1 + 6
        for a, b in zip(lines, lines[1:] + lines[:1]):
            assert a.p2_idx == b.p1_idx

    def test_corners_are_regular(self):
        sketch, _center, _cmd = _execute(sides=6, end=(0.0, 10.0))
        corners = _corner_positions(sketch)
        assert corners[0] == pytest.approx((0.0, 10.0))
        for x, y in corners:
            assert math.hypot(x, y) == pytest.approx(10.0)

    def test_construction_circle_around_the_center(self):
        sketch, center, _cmd = _execute(sides=6)
        (circle,) = _circles(sketch)
        assert circle.construction
        assert circle.center_idx == center

    def test_constraints_keep_it_regular(self):
        sketch, _center, _cmd = _execute(sides=6)
        constraints = sketch.constraints
        (equal,) = [
            c for c in constraints if isinstance(c, EqualLengthConstraint)
        ]
        assert sorted(equal.entity_ids) == sorted(
            line.id for line in _lines(sketch)
        )
        on_circle = [
            c for c in constraints if isinstance(c, PointOnLineConstraint)
        ]
        assert len(on_circle) == 5

    def test_typed_radius_adds_a_radius_constraint(self):
        sketch, _center, _cmd = _execute(sides=6, fixed_radius=25.0)
        (radius,) = [
            c for c in sketch.constraints if isinstance(c, RadiusConstraint)
        ]
        assert radius.value == 25.0
        assert math.hypot(*_corner_positions(sketch)[0]) == pytest.approx(25)

    def test_star(self):
        sketch, _center, _cmd = _execute(sides=5, star_ratio=0.4)
        assert len(_lines(sketch)) == 10
        circles = _circles(sketch)
        assert len(circles) == 2
        assert all(c.construction for c in circles)
        (ratio,) = [
            c
            for c in sketch.constraints
            if isinstance(c, AspectRatioConstraint)
        ]
        assert ratio.ratio == pytest.approx(0.4)

    def test_solver_keeps_the_shape(self):
        sketch, _center, _cmd = _execute(sides=7, star_ratio=0.5)
        before = _corner_positions(sketch)
        assert sketch.solve()
        assert _corner_positions(sketch) == pytest.approx(before, abs=1e-6)

    def test_dragging_a_corner_keeps_it_regular(self):
        sketch, center, _cmd = _execute(sides=6)
        first = _lines(sketch)[0].p1_idx
        point = sketch.registry.get_point(first)
        point.x, point.y = 0.0, 20.0
        sketch.solve()
        c = sketch.registry.get_point(center)
        radii = [
            math.hypot(x - c.x, y - c.y) for x, y in _corner_positions(sketch)
        ]
        assert radii == pytest.approx([radii[0]] * 6, rel=1e-4)

    def test_snapped_first_corner_is_reused(self):
        sketch = Sketch()
        center = sketch.add_point(0, 0)
        target = sketch.add_point(0, 10)
        cmd = PolygonCommand(sketch, center, (0.0, 9.0), end_pid=target)
        cmd.execute()
        assert _lines(sketch)[0].p1_idx == target
        assert _corner_positions(sketch)[1] == pytest.approx(
            (-10 * math.sin(math.pi / 3), 5.0)
        )
        assert len(sketch.registry.points) == 8
        cmd.undo()
        assert sketch.registry.get_point(target) is not None

    def test_typed_radius_ignores_the_snapped_point(self):
        sketch = Sketch()
        center = sketch.add_point(0, 0)
        target = sketch.add_point(0, 10)
        PolygonCommand(
            sketch, center, (0.0, 10.0), end_pid=target, fixed_radius=4.0
        ).execute()
        assert _lines(sketch)[0].p1_idx != target
        assert _corner_positions(sketch)[0] == pytest.approx((0.0, 4.0))

    def test_rejects_fewer_than_three_sides(self):
        with pytest.raises(ValueError):
            _execute(sides=2)

    def test_undo_removes_everything(self):
        sketch, _center, cmd = _execute(sides=6)
        cmd.undo()
        assert _lines(sketch) == []
        assert _circles(sketch) == []
        assert len(sketch.registry.points) == 2

    def test_temporary_center_is_taken_over(self):
        sketch = Sketch()
        registry = sketch.registry
        center = registry.add_point(5, 5)
        cmd = PolygonCommand(
            sketch, center, (15, 5), sides=4, is_center_temp=True
        )
        cmd.execute()
        assert registry.get_point(center) is not None
        cmd.undo()
        assert all(p.id != center for p in registry.points)


class TestPolygonPreview:
    def test_preview_draws_the_polygon(self):
        sketch = Sketch()
        registry = sketch.registry
        state = PolygonCommand.start_preview(registry, 0.0, 0.0, sides=5)
        assert isinstance(state, PolygonPreviewState)
        PolygonCommand.update_preview(registry, state, 10.0, 0.0)
        assert len(state.line_ids) == 5
        assert registry.get_point(state.radius_id).x == pytest.approx(10.0)

    def test_changing_the_shape_rebuilds_the_preview(self):
        sketch = Sketch()
        registry = sketch.registry
        state = PolygonCommand.start_preview(registry, 0.0, 0.0)
        PolygonCommand.update_preview(registry, state, 10.0, 0.0)
        state.set_shape(registry, sides=5, star_ratio=0.5)
        assert len(state.line_ids) == 10
        PolygonCommand.cleanup_preview(registry, state)
        assert registry.entities == []
        assert {p.id for p in registry.points} == {0, state.center_id}

    def test_rebuilding_keeps_point_usage_counts(self):
        sketch = Sketch()
        registry = sketch.registry
        state = PolygonCommand.start_preview(registry, 0.0, 0.0)
        PolygonCommand.update_preview(registry, state, 10.0, 0.0)
        old_corners = list(state.corner_ids)
        state.set_shape(registry, sides=5, star_ratio=None)
        assert not any(registry.is_point_used(p) for p in old_corners)
        preview_ids = state.get_preview_point_ids()
        PolygonCommand.cleanup_preview(registry, state)
        assert not any(registry.is_point_used(p) for p in preview_ids)

    def test_radius_input_scales_the_preview(self):
        sketch = Sketch()
        registry = sketch.registry
        state = PolygonCommand.start_preview(registry, 0.0, 0.0)
        PolygonCommand.update_preview(registry, state, 3.0, 4.0)
        state.set_radius(registry, 10.0)
        p = registry.get_point(state.radius_id)
        assert (p.x, p.y) == pytest.approx((6.0, 8.0))
        PolygonCommand.update_preview(registry, state, 1.0, 1.0)
        p = registry.get_point(state.radius_id)
        assert math.hypot(p.x, p.y) == pytest.approx(10.0)

    def test_dimensions(self):
        sketch = Sketch()
        registry = sketch.registry
        state = PolygonCommand.start_preview(registry, 0.0, 0.0)
        PolygonCommand.update_preview(registry, state, 10.0, 0.0)
        labels = [d.label for d in state.get_dimensions(registry)]
        assert labels[0].startswith("R")
        assert labels[1] == "6"

import pytest
from rayforge.core.geo import Geometry
from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.geometry_import_error import GeometryImportError
from rayforge.core.sketcher.entities import Line, Arc


class TestSketchFromGeometry:
    """Tests for Sketch.from_geometry() method."""

    def test_from_empty_geometry(self):
        """Converting an empty geometry should return an empty sketch."""
        geo = Geometry()
        sketch = Sketch.from_geometry(geo)

        assert sketch is not None
        assert isinstance(sketch, Sketch)
        assert sketch.is_empty

    def test_from_geometry_with_single_line(self):
        """Converting a geometry with a single line segment."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 10)

        sketch = Sketch.from_geometry(geo)

        assert len(sketch.registry.entities) == 1
        assert isinstance(sketch.registry.entities[0], Line)

        points = sketch.registry.points
        assert len(points) == 3
        assert points[0].fixed is True

    def test_from_geometry_with_multiple_lines(self):
        """Converting a geometry with multiple connected lines."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 0)
        geo.line_to(10, 10)
        geo.line_to(0, 10)

        sketch = Sketch.from_geometry(geo)

        assert len(sketch.registry.entities) == 3
        for entity in sketch.registry.entities:
            assert isinstance(entity, Line)

    def test_from_geometry_with_arc(self):
        """Converting a geometry with an arc."""
        geo = Geometry()
        geo.move_to(10, 0)
        geo.arc_to(0, 10, i=-10, j=0, clockwise=False)

        sketch = Sketch.from_geometry(geo)

        assert len(sketch.registry.entities) == 1
        arc = sketch.registry.entities[0]
        assert isinstance(arc, Arc)
        assert arc.clockwise is False

    def test_from_geometry_with_clockwise_arc(self):
        """Converting a geometry with a clockwise arc."""
        geo = Geometry()
        geo.move_to(0, 10)
        geo.arc_to(10, 0, i=0, j=-10, clockwise=True)

        sketch = Sketch.from_geometry(geo)

        assert len(sketch.registry.entities) == 1
        arc = sketch.registry.entities[0]
        assert isinstance(arc, Arc)
        assert arc.clockwise is True

    def test_from_geometry_with_bezier_raises_error(self):
        """Converting geometry with bezier curves should raise
        GeometryImportError."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.bezier_to(10, 10, 3, 3, 7, 7)

        with pytest.raises(GeometryImportError) as exc_info:
            Sketch.from_geometry(geo)

        assert "bezier" in str(exc_info.value).lower()

    def test_from_geometry_point_deduplication(self):
        """Points at the same coordinates should be deduplicated."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 0)
        geo.line_to(10, 10)
        geo.line_to(0, 0)

        sketch = Sketch.from_geometry(geo)

        assert len(sketch.registry.points) == 4
        assert len(sketch.registry.entities) == 3

    def test_from_geometry_with_disconnected_paths(self):
        """Converting geometry with multiple disconnected paths."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 0)
        geo.move_to(20, 20)
        geo.line_to(30, 20)

        sketch = Sketch.from_geometry(geo)

        assert len(sketch.registry.entities) == 2
        assert len(sketch.registry.points) == 5

    def test_from_geometry_mixed_lines_and_arcs(self):
        """Converting geometry with both lines and arcs."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 0)
        geo.arc_to(0, 10, i=-10, j=0, clockwise=False)

        sketch = Sketch.from_geometry(geo)

        assert len(sketch.registry.entities) == 2
        assert isinstance(sketch.registry.entities[0], Line)
        assert isinstance(sketch.registry.entities[1], Arc)

    def test_from_geometry_roundtrip(self):
        """Test that converting to sketch and back to geometry preserves
        the shape."""
        original_geo = Geometry()
        original_geo.move_to(0, 0)
        original_geo.line_to(10, 0)
        original_geo.line_to(10, 10)
        original_geo.line_to(0, 10)
        original_geo.close_path()

        sketch = Sketch.from_geometry(original_geo)
        result_geo = sketch.to_geometry()

        original_rect = original_geo.rect()
        result_rect = result_geo.rect()

        assert original_rect == pytest.approx(result_rect, rel=1e-6)

    def test_from_geometry_arc_roundtrip(self):
        """Test arc roundtrip conversion."""
        original_geo = Geometry()
        original_geo.move_to(10, 0)
        original_geo.arc_to(0, 10, i=-10, j=0, clockwise=False)

        sketch = Sketch.from_geometry(original_geo)
        result_geo = sketch.to_geometry()

        original_rect = original_geo.rect()
        result_rect = result_geo.rect()

        assert original_rect == pytest.approx(result_rect, rel=1e-6)

    def test_from_geometry_preserves_arc_direction(self):
        """Arc direction (clockwise/counter-clockwise) should be preserved."""
        geo_cw = Geometry()
        geo_cw.move_to(0, 10)
        geo_cw.arc_to(10, 0, i=0, j=-10, clockwise=True)

        sketch_cw = Sketch.from_geometry(geo_cw)
        arc_cw = sketch_cw.registry.entities[0]
        assert isinstance(arc_cw, Arc)
        assert arc_cw.clockwise is True

        geo_ccw = Geometry()
        geo_ccw.move_to(10, 0)
        geo_ccw.arc_to(0, 10, i=-10, j=0, clockwise=False)

        sketch_ccw = Sketch.from_geometry(geo_ccw)
        arc_ccw = sketch_ccw.registry.entities[0]
        assert isinstance(arc_ccw, Arc)
        assert arc_ccw.clockwise is False

    def test_from_geometry_multiple_beziers_raises_error(self):
        """Multiple beziers should still raise error."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.bezier_to(10, 10, 3, 3, 7, 7)
        geo.bezier_to(20, 0, 13, 13, 17, 7)

        with pytest.raises(GeometryImportError):
            Sketch.from_geometry(geo)

    def test_from_geometry_mixed_with_bezier_raises_error(self):
        """Mixed geometry with bezier should raise error."""
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(10, 0)
        geo.bezier_to(20, 10, 13, 3, 17, 7)

        with pytest.raises(GeometryImportError):
            Sketch.from_geometry(geo)

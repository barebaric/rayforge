import pytest

from raygeo import Geometry
from rayforge.image.geo_renderer import render_geometry_to_png


@pytest.fixture
def closed_rectangle():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    geo.line_to(0, 10)
    geo.close_path()
    geo.sync_to_data()
    return geo


def test_render_geometry_to_png_returns_bytes(closed_rectangle):
    result = render_geometry_to_png(closed_rectangle, 100)

    assert result is not None
    assert isinstance(result, bytes)
    assert len(result) > 0
    assert result[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_geometry_to_png_empty_geometry():
    geo = Geometry()

    result = render_geometry_to_png(geo, 100)

    assert result is None


def test_render_geometry_to_png_zero_dimensions():
    geo = Geometry()
    geo.move_to(5, 5)
    geo.line_to(5, 5)
    geo.sync_to_data()

    result = render_geometry_to_png(geo, 100)

    assert result is None


def test_render_geometry_to_png_custom_color():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    geo.line_to(0, 10)
    geo.close_path()
    geo.sync_to_data()

    result = render_geometry_to_png(geo, 100, color=(1.0, 0.0, 0.0, 1.0))

    assert result is not None


def test_render_geometry_to_png_custom_line_width():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    geo.line_to(0, 10)
    geo.close_path()
    geo.sync_to_data()

    result = render_geometry_to_png(geo, 100, line_width=2.0)

    assert result is not None


def test_render_geometry_to_png_single_point():
    geo = Geometry()
    geo.move_to(5, 5)
    geo.sync_to_data()

    result = render_geometry_to_png(geo, 100)

    assert result is None


def test_render_geometry_to_png_arc():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.arc_to(10, 0, i=5, j=0)
    geo.sync_to_data()

    result = render_geometry_to_png(geo, 100)

    assert result is not None


def test_render_geometry_to_png_bezier():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.bezier_to(10, 0, 2, 10, 8, 10, 0.0)
    geo.sync_to_data()

    result = render_geometry_to_png(geo, 100)

    assert result is not None

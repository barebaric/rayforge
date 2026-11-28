from raygeo.geo import Geometry

from rayforge.image.ops_renderer import OpsRenderer


def test_line_only_geometry_renders():
    geometry = Geometry()
    geometry.move_to(5, 0)
    geometry.line_to(5, 10)

    surface = OpsRenderer()._render_to_cairo_surface(geometry, 20, 100)

    assert surface is not None
    width = surface.get_width()
    height = surface.get_height()
    stride = surface.get_stride()
    data = surface.get_data()
    visible = any(
        data[row * stride + col * 4 + 3]
        for row in range(height)
        for col in range(width)
    )
    assert visible

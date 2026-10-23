"""
Unit tests for the WorkOriginElement rendering.
"""

import cairo
import numpy as np
import pytest
from raygeo.geo import Matrix

from rayforge.ui_gtk.canvas2d.elements.work_origin import WorkOriginElement


def _render_marker(scale: float) -> np.ndarray:
    """Renders the marker into a small image and returns it as RGB.

    The parent canvas normally applies the element transform to the
    Cairo context before draw(); the test does the same.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 120, 120)
    ctx = cairo.Context(surface)
    # Dark background like the canvas theme, so diluted strokes show up
    ctx.set_source_rgb(0.1, 0.1, 0.1)
    ctx.paint()

    element = WorkOriginElement()
    matrix = Matrix.translation(20, 20) @ Matrix.scale(scale, scale)
    ctx.transform(cairo.Matrix(*matrix.for_cairo()))
    element.draw(ctx)
    surface.flush()

    buf = np.ndarray(
        shape=(120, 120, 4), dtype=np.uint8, buffer=surface.get_data()
    )
    return buf[:, :, :3]


@pytest.mark.parametrize("scale", [1.0, 2.0, 6.0])
def test_marker_stays_visible_at_zoom_levels(scale):
    """The stroke width is device-space: the marker must deposit enough
    green to survive anti-aliasing at every zoom level. The old
    world-unit 0.2 mm width dilutes into sub-pixel lines whose total
    green energy stays near zero regardless of zoom."""
    # numpy gives BGRA order for little-endian ARGB: [1] is green
    rgb = _render_marker(scale).astype(int)
    g, b = rgb[:, :, 1], rgb[:, :, 0]
    # The stroke must fully cover at least one pixel: a world-unit
    # 0.2 mm width covers ~20% of a pixel at low zoom, which the light
    # theme background washes out completely.
    peak_green = int((g - b).max())
    assert peak_green >= 92, (
        f"marker stroke too faint at scale {scale}: peak green "
        f"coverage {peak_green} (expected >= 92)"
    )


def test_marker_points_into_the_bed():
    """Top-right origin: both axes point left and down on screen."""
    element = WorkOriginElement()
    element.set_axis_direction(x_axis_right=True, y_axis_down=True)
    assert element._x_dir == (-1.0, 0.0)
    assert element._y_dir == (0.0, -1.0)

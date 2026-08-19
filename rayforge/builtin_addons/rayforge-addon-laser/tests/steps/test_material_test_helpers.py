"""Tests for the material test preview drawing helpers.

The preview drawn by ``draw_preview`` is the image the Engrave step
rasterises, so it must fill the full target surface; otherwise the
engraved grid comes out smaller than the workpiece.
"""

import cairo
import numpy as np
import pytest
from laser_essentials.material_test_helpers import draw_preview


@pytest.fixture
def params():
    return {
        "test_type": "Engrave",
        "grid_mode": "Power vs Speed",
        "speed_range": (3000.0, 24000.0),
        "power_range": (10.0, 70.0),
        "passes_range": (1, 5),
        "offset_range": (-0.5, 0.5),
        "fixed_speed": 1000.0,
        "fixed_power": 50.0,
        "grid_dimensions": (5, 5),
        "shape_size": 5.0,
        "spacing": 2.0,
        "include_labels": True,
        "label_power_percent": 10.0,
        "label_speed": 1000.0,
        "line_interval_mm": None,
    }


def _content_bbox(surface: cairo.ImageSurface):
    buf = np.ndarray(
        shape=(surface.get_height(), surface.get_width()),
        dtype=np.uint32,
        buffer=surface.get_data(),
    )
    alpha = (buf >> 24) & 0xFF
    rows = (alpha > 0).nonzero()[0]
    cols = (alpha > 0).nonzero()[1]
    if rows.size == 0:
        return None
    return (cols.min(), rows.min(), cols.max(), rows.max())


@pytest.mark.parametrize(
    "size",
    [
        (810, 405),  # Engrave with a 0.1mm spot on a square workpiece
        (256, 256),
        (128, 96),
    ],
)
def test_draw_preview_fills_target_surface(size, params):
    width, height = size
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    draw_preview(ctx, width, height, params)
    surface.flush()

    bbox = _content_bbox(surface)
    assert bbox is not None
    left, top, right, bottom = bbox
    # The grid must span essentially the whole surface (a small sliver
    # is allowed for the label/margin layout).
    assert right - left >= width * 0.95
    assert bottom - top >= height * 0.95

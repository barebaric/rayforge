"""Tests for the stock element's texture rendering."""

import cairo
import numpy as np
import pytest
import pyvips
from raygeo.geo import Geometry, Matrix

from rayforge.core.doc import Doc
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.ui_gtk.canvas2d.elements.stock import StockElement
from rayforge.ui_gtk.shared.texture_loader import load_texture_cairo_surface

pytestmark = pytest.mark.ui


def _stock_matrix(world_w: float, world_h: float) -> Matrix:
    """Matrix mapping the element's 1x1 local box to world millimeters."""
    return Matrix.translation(10, 20) @ Matrix.scale(world_w, world_h)


def _rectangle_geometry(width=100.0, height=50.0) -> Geometry:
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(width, 0)
    geo.line_to(width, height)
    geo.line_to(0, height)
    geo.close_path()
    return geo


def _stock_item(material_uid: str | None) -> StockItem:
    doc = Doc()
    asset = StockAsset(name="Test Stock", geometry=_rectangle_geometry())
    asset.material_uid = material_uid
    doc.add_asset(asset)
    item = StockItem(stock_asset_uid=asset.uid, name="Stock")
    item.parent = doc
    return item


def _render(element: StockElement) -> cairo.ImageSurface:
    """Render the element like the canvas would, at 2 px/mm."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 800, 400)
    ctx = cairo.Context(surface)
    ctx.translate(20, 300)
    ctx.scale(2.0, -2.0)
    ctx.translate(10, 20)
    ctx.scale(300, 150)
    ctx.rectangle(0, 0, 1, 1)
    ctx.clip()
    element.draw(ctx)
    return surface


def _opaque_pixels(surface: cairo.ImageSurface) -> np.ndarray:
    buf = np.frombuffer(surface.get_data(), dtype=np.uint8)
    return buf.reshape(surface.get_height(), surface.get_width(), 4)


def test_load_texture_cairo_surface(tmp_path):
    """The texture loader returns a Cairo surface at full resolution."""
    img = pyvips.Image.black(64, 32).addalpha()
    path = tmp_path / "tex.webp"
    img.webpsave(str(path), Q=90)

    result = load_texture_cairo_surface(path)

    assert result is not None
    surface, _buffer = result
    assert surface.get_width() == 64
    assert surface.get_height() == 32


def test_load_texture_cairo_surface_missing(tmp_path):
    """A missing texture file returns None."""
    assert load_texture_cairo_surface(tmp_path / "nope.webp") is None


def test_stock_element_draw_renders_texture(ui_context_initializer):
    """A material with a texture actually paints opaque pixels."""
    item = _stock_item("oak")
    item.matrix = _stock_matrix(300, 150)
    element = StockElement(item)

    surface = _render(element)
    pixels = _opaque_pixels(surface)

    # The texture fill is opaque; the hairline stroke is only ~40% alpha.
    opaque_fill = pixels[:, :, 3] > 200
    assert opaque_fill.sum() > 1000, "texture fill did not render"

    # Oak is warm brown: red must dominate blue (guards against the
    # RGBA -> BGRA channel swap for cairo ARGB32). Pixels are BGRA.
    fill = pixels[opaque_fill]
    assert fill[:, 2].mean() > fill[:, 0].mean()


def test_stock_element_draw_without_texture(ui_context_initializer):
    """Materials without a texture fall back to the color path."""
    for material_uid in (None, "mdf"):
        item = _stock_item(material_uid)
        item.matrix = _stock_matrix(300, 150)
        element = StockElement(item)

        surface = _render(element)
        pixels = _opaque_pixels(surface)

        # Semi-transparent color fill (alpha 0.5) should still paint.
        assert (pixels[:, :, 3] > 0).sum() > 1000


def _pattern_matrix(element: StockElement, surface, tile_mm):
    """Apply the tile transform and return the resulting pattern matrix."""
    out = cairo.ImageSurface(cairo.FORMAT_ARGB32, 16, 16)
    ctx = cairo.Context(out)
    ctx.scale(1.0 / 100, 1.0 / 50)
    element._set_tiled_texture_source(ctx, surface, tile_mm, 100, 50)
    source = ctx.get_source()
    assert isinstance(source, cairo.SurfacePattern)
    return source.get_matrix()


def test_tile_scale_respects_geometry_span(ui_context_initializer):
    """
    The pattern matrix must be scaled by the geometry span: it is
    applied in geometry coordinates, where one image should cover
    `tile_mm` world millimeters.
    """
    item = _stock_item("oak")
    item.matrix = _stock_matrix(300, 150)
    element = StockElement(item)
    material = item.material
    texture_path = material.get_texture_path() if material else None
    assert texture_path is not None
    texture = load_texture_cairo_surface(texture_path)
    assert texture is not None
    surface, _buffer = texture

    matrix = _pattern_matrix(element, surface, 300)

    # img_w * world_w / (tile_mm * geo_w) = 1000 * 300 / (300 * 100)
    assert matrix.xx == pytest.approx(10.0)
    assert matrix.yy == pytest.approx(10.0)

import logging

import cairo
from raygeo.geo import Matrix

from ....context import get_context
from ....core.stock import StockItem
from ....image.geo_renderer import geometry_to_cairo
from ...canvas import CanvasElement
from ...shared.texture_loader import (
    load_texture_cairo_surface,
    tinted_texture_cairo_surface,
)

logger = logging.getLogger(__name__)


class StockElement(CanvasElement):
    """
    A CanvasElement that visualizes a single StockItem model.
    """

    def __init__(self, stock_item: StockItem, **kwargs):
        self.data: StockItem = stock_item
        super().__init__(
            0,
            0,
            1.0,
            1.0,  # Geometry is 1x1, transform handles size
            data=stock_item,
            buffered=False,
            pixel_perfect_hit=False,  # Bbox is fine for stock
            **kwargs,
        )
        self.data.updated.connect(self._on_model_content_changed)
        self.data.transform_changed.connect(self._on_transform_changed)
        self._view_visible = True
        self._on_transform_changed(self.data)
        self._update_visible()

    def remove(self):
        """Disconnects signals before removal."""
        self.data.updated.disconnect(self._on_model_content_changed)
        self.data.transform_changed.disconnect(self._on_transform_changed)
        super().remove()

    def set_visible(self, visible: bool = True):
        self.selectable = visible
        if not visible and self.selected:
            self.selected = False
        return super().set_visible(visible)

    def set_view_visible(self, visible: bool):
        """Sets the canvas-level view toggle for stock visibility.

        Stock items remain selectable only while the view toggle and the
        item's own visibility flag are both enabled.
        """
        self._view_visible = visible
        self._update_visible()

    def _update_visible(self):
        self.set_visible(self._view_visible and self.data.visible)

    def _on_model_content_changed(self, stock_item: StockItem):
        """Handler for when the stock item's geometry changes."""
        logger.debug(
            f"Model content changed for '{stock_item.name}', "
            "triggering update."
        )
        self._on_visibility_changed()
        if self.canvas:
            self.canvas.queue_draw()

    def _on_visibility_changed(self):
        """Handler for when the stock item's visibility changes."""
        self._update_visible()

    def _on_transform_changed(
        self, stock_item: StockItem, *, old_matrix: Matrix | None = None
    ):
        """Handler for when the stock item's transform changes."""
        if not self.canvas or self.transform == stock_item.matrix:
            return
        self.set_transform(stock_item.matrix)

    def draw(self, ctx: cairo.Context):
        """Draws the stock geometry directly to the main canvas context."""
        if self.data.geometry.is_empty() or not self.visible:
            return

        ctx.save()

        min_x, min_y, max_x, max_y = self.data.geometry.rect()
        geo_width = max_x - min_x
        geo_height = max_y - min_y

        # Scale and translate context to fit geometry inside the 1x1 element
        if geo_width > 1e-9 and geo_height > 1e-9:
            ctx.scale(1.0 / geo_width, 1.0 / geo_height)
            ctx.translate(-min_x, -min_y)

        # Draw the geometry path using the standard method
        geometry_to_cairo(self.data.geometry, ctx)

        # Texture the stock with the material's texture, tiled so that
        # one image covers `texture_size_mm` world millimeters.
        material = self.data.material
        if material is None:
            material = get_context().material_mgr.get_default_material()
        texture_source = None
        texture_size_mm = None
        texture_path = material.get_texture_path()
        if texture_path is not None:
            tint = self.data.get_effective_rgba()
            if tint is not None:
                texture_source = tinted_texture_cairo_surface(
                    texture_path, tint
                )
            else:
                texture_source = load_texture_cairo_surface(texture_path)
            texture_size_mm = material.appearance.texture_size_mm

        if texture_source is not None and texture_size_mm is not None:
            surface, _buffer = texture_source
            self._set_tiled_texture_source(
                ctx, surface, texture_size_mm, geo_width, geo_height
            )
        else:
            r, g, b, a = material.get_display_rgba(0.5)
            ctx.set_source_rgba(r, g, b, a)

        ctx.fill_preserve()

        # Stroke the path with a crisp, 1-device-pixel hairline
        ctx.set_source_rgba(0.2, 0.2, 0.2, 0.8)
        ctx.set_hairline(True)
        ctx.stroke()

        ctx.restore()

    def _set_tiled_texture_source(
        self,
        ctx: cairo.Context,
        surface: cairo.ImageSurface,
        tile_mm: float,
        geo_width: float,
        geo_height: float,
    ):
        """
        Set the source to the texture tiled across the stock's world
        size at `tile_mm` per repeat.

        The pattern is applied in the current user space, which is the
        geometry's own coordinate system (the draw transform maps it to
        the element's 1x1 local box, which the element transform scales
        to the stock's world size in millimeters).
        """
        world_rect = self.data.get_world_transform().transform_rectangle(
            (0.0, 0.0, 1.0, 1.0)
        )
        world_w = max(world_rect[2], 1e-9)
        world_h = max(world_rect[3], 1e-9)
        geo_w = max(float(geo_width), 1e-9)
        geo_h = max(float(geo_height), 1e-9)
        tile_mm = max(float(tile_mm), 1e-9)

        img_w = surface.get_width()
        img_h = surface.get_height()

        pattern = cairo.SurfacePattern(surface)
        pattern.set_extend(cairo.Extend.REPEAT)
        matrix = cairo.Matrix()
        matrix.scale(
            img_w * world_w / (tile_mm * geo_w),
            img_h * world_h / (tile_mm * geo_h),
        )
        pattern.set_matrix(matrix)
        ctx.set_source(pattern)

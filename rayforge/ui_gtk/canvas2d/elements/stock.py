import logging
import math
import cairo
from ....core.stock import StockItem
from ....core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_I,
    COL_J,
    COL_CW,
)
from ...canvas import CanvasElement


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
        self._on_transform_changed(self.data)
        self._on_visibility_changed()

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
        self.set_visible(self.data.visible)

    def _on_transform_changed(self, stock_item: StockItem):
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

        # Build the path from geometry data
        last_point = (0.0, 0.0)
        data = self.data.geometry.data
        if data is not None:
            for row in data:
                cmd_type = row[COL_TYPE]
                x, y = row[COL_X], row[COL_Y]

                if cmd_type == CMD_TYPE_MOVE:
                    ctx.move_to(x, y)
                elif cmd_type == CMD_TYPE_LINE:
                    ctx.line_to(x, y)
                elif cmd_type == CMD_TYPE_ARC:
                    start_x, start_y = last_point
                    i, j = row[COL_I], row[COL_J]
                    center_x, center_y = start_x + i, start_y + j
                    radius = math.dist(
                        (start_x, start_y), (center_x, center_y)
                    )
                    if radius < 1e-6:
                        ctx.line_to(x, y)
                        last_point = (x, y)
                        continue

                    angle1 = math.atan2(start_y - center_y, start_x - center_x)
                    angle2 = math.atan2(y - center_y, x - center_x)

                    is_clockwise = bool(row[COL_CW])
                    if is_clockwise:
                        ctx.arc(center_x, center_y, radius, angle1, angle2)
                    else:
                        ctx.arc_negative(
                            center_x, center_y, radius, angle1, angle2
                        )
                last_point = (x, y)

        # Get the material color if available
        material = self.data.material
        if material:
            # Use material color with 0.5 alpha
            r, g, b, a = material.get_display_rgba(0.5)
            ctx.set_source_rgba(r, g, b, a)
        else:
            # Use default color when no material is assigned
            ctx.set_source_rgba(0.5, 0.5, 0.5, 0.3)

        ctx.fill_preserve()

        # Stroke the path with a crisp, 1-device-pixel hairline
        ctx.set_source_rgba(0.2, 0.2, 0.2, 0.8)
        ctx.set_hairline(True)
        ctx.stroke()

        ctx.restore()

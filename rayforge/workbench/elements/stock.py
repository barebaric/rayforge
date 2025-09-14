import logging
import math
from typing import Optional
import cairo
from ...core.stock import StockItem
from ...core.geometry import (
    MoveToCommand as GeoMove,
    LineToCommand as GeoLine,
    ArcToCommand as GeoArc,
)
from ..canvas import CanvasElement


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
            buffered=True,  # Good for complex vector shapes
            pixel_perfect_hit=False,  # Bbox is fine for stock
            **kwargs,
        )
        self.data.updated.connect(self._on_model_content_changed)
        self.data.transform_changed.connect(self._on_transform_changed)
        self._on_transform_changed(self.data)
        self.trigger_update()

    def remove(self):
        """Disconnects signals before removal."""
        self.data.updated.disconnect(self._on_model_content_changed)
        self.data.transform_changed.disconnect(self._on_transform_changed)
        super().remove()

    def _on_model_content_changed(self, stock_item: StockItem):
        """Handler for when the stock item's geometry changes."""
        logger.debug(
            f"Model content changed for '{stock_item.name}', "
            "triggering update."
        )
        self.trigger_update()

    def _on_transform_changed(self, stock_item: StockItem):
        """Handler for when the stock item's transform changes."""
        if not self.canvas or self.transform == stock_item.matrix:
            return
        self.set_transform(stock_item.matrix)

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """Renders the stock item's geometry to a new surface."""
        if width <= 0 or height <= 0 or not self.data.geometry:
            return None

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        # Fill with a semi-transparent color
        ctx.set_source_rgba(0.5, 0.5, 0.5, 0.3)
        ctx.paint()

        # Get the bounding box of the geometry to scale it correctly
        min_x, min_y, max_x, max_y = self.data.geometry.rect()
        geo_width = max_x - min_x
        geo_height = max_y - min_y

        # Translate and scale the context so the geometry fills the surface
        if geo_width > 1e-9 and geo_height > 1e-9:
            ctx.translate(
                -min_x * (width / geo_width), -min_y * (height / geo_height)
            )
            ctx.scale(width / geo_width, height / geo_height)

        # This block iterates through the geometry commands and builds a
        # cairo path.
        for cmd in self.data.geometry:
            if cmd.end is None:
                continue

            x, y, z = cmd.end

            match cmd:
                case GeoMove():
                    ctx.move_to(x, y)
                case GeoLine():
                    ctx.line_to(x, y)
                case GeoArc():
                    start_x, start_y = ctx.get_current_point()
                    i, j = cmd.center_offset
                    center_x, center_y = start_x + i, start_y + j
                    radius = math.dist(
                        (start_x, start_y), (center_x, center_y)
                    )
                    if radius < 1e-6:
                        ctx.line_to(x, y)
                        continue

                    angle1 = math.atan2(start_y - center_y, start_x - center_x)
                    angle2 = math.atan2(y - center_y, x - center_x)

                    if cmd.clockwise:
                        ctx.arc(center_x, center_y, radius, angle1, angle2)
                    else:
                        ctx.arc_negative(
                            center_x, center_y, radius, angle1, angle2
                        )

        # Now, style and stroke the path that was just created.
        ctx.set_source_rgba(0.2, 0.2, 0.2, 0.8)

        # To achieve a crisp line of a consistent apparent width in device
        # space, we must set the line width in the scaled user space. With
        # non-uniform scaling, a circular pen in user space becomes an
        # ellipse. Using the geometric mean of the inverse scaling factors
        # provides a good compromise for the line width. A target width of
        # 1.5px is chosen as it often appears sharper than 2px.
        if width > 1e-9 and height > 1e-9:
            avg_inv_scale = math.sqrt(
                (geo_width / width) * (geo_height / height)
            )
            ctx.set_line_width(1.5 * avg_inv_scale)
        else:
            ctx.set_line_width(1.5)

        # Use round caps and joins for a smoother appearance, which helps
        # mitigate aliasing effects at sharp corners.
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.stroke()

        return surface

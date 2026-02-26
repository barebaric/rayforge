import cairo
from ...canvas import CanvasElement


class WorkareaBackgroundElement(CanvasElement):
    """
    A non-interactive CanvasElement that draws a gray background
    for the workarea within the machine bed.
    """

    def __init__(self, **kwargs):
        super().__init__(
            x=0,
            y=0,
            width=200.0,
            height=200.0,
            selectable=False,
            draggable=False,
            clip=False,
            **kwargs,
        )
        self._color = (0.8, 0.8, 0.8, 0.1)

    def set_color(self, r: float, g: float, b: float, a: float = 1.0):
        """Sets the background color."""
        self._color = (r, g, b, a)
        if self.canvas:
            self.canvas.queue_draw()

    def draw(self, ctx: cairo.Context):
        """Renders the workarea background as a filled rectangle."""
        ctx.save()
        ctx.set_source_rgba(*self._color)
        ctx.rectangle(0, 0, self.width, self.height)
        ctx.fill()
        ctx.restore()


class AxisExtentFrameElement(CanvasElement):
    """
    A non-interactive CanvasElement that draws a red frame outline
    representing the full axis extents of the machine. This frame
    surrounds the work surface when the work surface is smaller than
    the axis extents.
    """

    def __init__(self, **kwargs):
        super().__init__(
            x=0,
            y=0,
            width=200.0,
            height=200.0,
            selectable=False,
            draggable=False,
            clip=False,
            **kwargs,
        )
        self._color = (1.0, 0.0, 0.0, 0.5)

    def set_size(self, width: float, height: float):
        """Updates the size of the extent frame."""
        if self.width == width and self.height == height:
            return
        super().set_size(width, height)
        if self.canvas:
            self.canvas.queue_draw()

    def set_color(self, r: float, g: float, b: float, a: float = 1.0):
        """Sets the frame color."""
        self._color = (r, g, b, a)
        if self.canvas:
            self.canvas.queue_draw()

    def draw(self, ctx: cairo.Context):
        """
        Renders the extent frame as a simple rectangle outline.
        Uses a 1-pixel stroke width regardless of zoom level.
        """
        ctx.save()

        ctx.set_source_rgba(*self._color)
        ctx.set_hairline(True)

        ctx.new_path()
        ctx.rectangle(0, 0, self.width, self.height)
        ctx.stroke()

        ctx.restore()

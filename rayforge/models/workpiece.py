import cairo
from typing import Optional
from blinker import Signal
from ..config import config
from ..render import Renderer


class WorkPiece:
    """
    A WorkPiece represents a real world work piece, It is usually
    loaded from an image file and serves as input for all other
    operations.
    """
    def __init__(self, name):
        self.name = name
        self.data: bytes = None
        self.renderer: Optional[Renderer] = None
        self._renderer_ref_for_pyreverse: Renderer
        self.pos: tuple[float, float] = None, None   # in mm
        self.size: tuple[float, float] = None, None  # in mm
        self.surface: cairo.Surface = None
        self.changed: Signal = Signal()
        self.size_changed: Signal = Signal()

    def set_pos(self, x_mm: float, y_mm: float):
        self.pos = float(x_mm), float(y_mm)
        self.changed.send(self)

    def set_size(self, width_mm: float, height_mm: float):
        self.size = float(width_mm), float(height_mm)
        self.changed.send(self)
        self.size_changed.send(self)

    def get_default_size(self):
        size = self.renderer.get_natural_size(self.data)
        if None not in size:
            return size

        aspect = self.get_aspect_ratio()
        width_mm = config.machine.dimensions[0]
        height_mm = width_mm/aspect
        if height_mm > config.machine.dimensions[1]:
            height_mm = config.machine.dimensions[1]
            width_mm = height_mm*aspect

        return width_mm, height_mm

    def get_aspect_ratio(self):
        return self.renderer.get_aspect_ratio(self.data)

    @classmethod
    def from_file(cls, filename, renderer):
        wp = cls(filename)
        with open(filename, 'rb') as fp:
            wp.data = renderer.prepare(fp.read())
        wp.renderer = renderer
        wp.size = wp.get_default_size()
        return wp

    def render(self,
               pixels_per_mm_x: int,
               pixels_per_mm_y: int,
               size: tuple[float, float] = None,
               force: bool = False):
        size = self.get_default_size() if size is None else size
        width = size[0] * pixels_per_mm_x
        height = size[1] * pixels_per_mm_y

        if self.surface \
                and self.surface.get_width() == width \
                and self.surface.get_height() == height \
                and not force:
            return self.surface, False

        self.surface = self.renderer.render_workpiece(self.data,
                                                      width,
                                                      height)

        return self.surface, True

    def dump(self, indent=0):
        print("  "*indent, self.name, self.renderer.label)

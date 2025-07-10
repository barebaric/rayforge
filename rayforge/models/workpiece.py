import logging
import cairo
from typing import Generator, Optional, Tuple, cast
from blinker import Signal
from ..config import config
from ..render import Renderer


logger = logging.getLogger(__name__)


class WorkPiece:
    """
    A WorkPiece represents a real world work piece, It is usually
    loaded from an image file and serves as input for all other
    operations.
    """
    def __init__(self, name: str, data: bytes, renderer: Renderer):
        self.name = name
        self.data = data
        self.renderer = renderer
        self._renderer_ref_for_pyreverse: Renderer
        self.pos: Optional[Tuple[float, float]] = None
        self.size: Optional[Tuple[float, float]] = None  # in mm
        self.surface: Optional[cairo.ImageSurface] = None
        self.changed: Signal = Signal()
        self.pos_changed: Signal = Signal()
        self.size_changed: Signal = Signal()

    def set_pos(self, x_mm: float, y_mm: float):
        if (x_mm, y_mm) == self.pos:
            return   # avoid triggering event
        self.pos = float(x_mm), float(y_mm)
        self.changed.send(self)
        self.pos_changed.send(self)

    def set_size(self, width_mm: float, height_mm: float):
        logger.debug(f"Setting size: {width_mm}x{height_mm} for {self.name}")
        if (width_mm, height_mm) == self.size:
            return   # avoid needless events
        self.size = float(width_mm), float(height_mm)
        self.changed.send(self)
        self.size_changed.send(self)

    def get_default_size(self) -> Tuple[float, float]:
        size = self.renderer.get_natural_size(self.data)
        if None not in size:
            return cast(Tuple[float, float], size)

        aspect = self.get_default_aspect_ratio()
        width_mm = machine_width_mm = config.machine.dimensions[0]
        machine_height_mm = config.machine.dimensions[1]
        height_mm = width_mm/aspect if aspect else machine_height_mm
        if height_mm > machine_height_mm:
            height_mm = machine_height_mm
            width_mm = height_mm*aspect if aspect else machine_width_mm

        return width_mm, height_mm

    def get_current_size(self) -> Optional[Tuple[float, float]]:
        logger.debug(f"Current size: {self.size}")
        logger.debug(f"Current default size: {self.get_default_size()}")
        if not self.size:
            return self.get_default_size()
        return self.size

    def get_default_aspect_ratio(self):
        return self.renderer.get_aspect_ratio(self.data)

    def get_current_aspect_ratio(self) -> Optional[float]:
        return (self.size[0] / self.size[1]
                if self.size and self.size[1] else None)

    @classmethod
    def from_file(cls, filename, renderer):
        with open(filename, 'rb') as fp:
            data = renderer.prepare(fp.read())
        wp = cls(filename, data, renderer)
        wp.size = wp.get_default_size()
        return wp

    def render(self,
               pixels_per_mm_x: float,
               pixels_per_mm_y: float,
               size: Optional[Tuple[float, float]] = None,
               force: bool = False
               ) -> Tuple[Optional[cairo.ImageSurface], bool]:
        size = self.get_default_size() if size is None else size
        if not size:
            return None, False

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

    def render_chunk(
            self,
            pixels_per_mm_x: int,
            pixels_per_mm_y: int,
            size: Optional[Tuple[float, float]] = None,
            force: bool = False
        ) -> Generator[Tuple[cairo.ImageSurface, Tuple[float, float]],
                       None,
                       None]:
        natsize = self.get_default_size()
        size = natsize if size is None else size
        if not size:
            return

        width = size[0] * pixels_per_mm_x
        height = size[1] * pixels_per_mm_y

        if self.surface \
                and self.surface.get_width() == width \
                and self.surface.get_height() == height \
                and not force:
            yield self.surface, (0, 0)

        for chunk in self.renderer.render_chunk(self.data, width, height):
            yield chunk

    def dump(self, indent=0):
        print("  "*indent, self.name, self.renderer.label)

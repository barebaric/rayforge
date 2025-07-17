import logging
import cairo
from typing import Generator, Optional, Tuple, cast
from blinker import Signal
from ..config import config
from ..render import Renderer


logger = logging.getLogger(__name__)


class WorkPiece:
    """
    Represents a real-world workpiece.

    It is defined by its name and a renderer instance, which holds all
    information about the source image. The WorkPiece itself does not
    store image data, only its position and size on the canvas.
    """

    def __init__(self, name: str, renderer: Renderer):
        self.name = name
        self.renderer = renderer
        self._renderer_ref_for_pyreverse: Renderer
        self.pos: Optional[Tuple[float, float]] = None  # in mm
        self.size: Optional[Tuple[float, float]] = None  # in mm
        self.changed: Signal = Signal()
        self.pos_changed: Signal = Signal()
        self.size_changed: Signal = Signal()

    def set_pos(self, x_mm: float, y_mm: float):
        if (x_mm, y_mm) == self.pos:
            return
        self.pos = float(x_mm), float(y_mm)
        self.changed.send(self)
        self.pos_changed.send(self)

    def set_size(self, width_mm: float, height_mm: float):
        if (width_mm, height_mm) == self.size:
            return
        self.size = float(width_mm), float(height_mm)
        self.changed.send(self)
        self.size_changed.send(self)

    def get_default_size(self) -> Tuple[float, float]:
        size = self.renderer.get_natural_size()
        if None not in size:
            return cast(Tuple[float, float], size)

        aspect = self.get_default_aspect_ratio()
        machine_width = config.machine.dimensions[0]
        machine_height = config.machine.dimensions[1]
        width_mm = machine_width
        height_mm = width_mm / aspect if aspect else machine_height
        if height_mm > machine_height:
            height_mm = machine_height
            width_mm = height_mm * aspect if aspect else machine_width

        return width_mm, height_mm

    def get_current_size(self) -> Optional[Tuple[float, float]]:
        if not self.size:
            return self.get_default_size()
        return self.size

    def get_default_aspect_ratio(self):
        return self.renderer.get_aspect_ratio()

    def get_current_aspect_ratio(self) -> Optional[float]:
        return (self.size[0] / self.size[1]
                if self.size and self.size[1] else None)

    @classmethod
    def from_file(cls, filename: str, renderer_class: type[Renderer]):
        with open(filename, 'rb') as fp:
            data = fp.read()
        renderer = renderer_class(data)
        wp = cls(filename, renderer)
        wp.size = wp.get_default_size()
        return wp

    def render_for_ops(
        self,
        pixels_per_mm_x: float,
        pixels_per_mm_y: float,
        size: Optional[Tuple[float, float]] = None
    ) -> Optional[cairo.ImageSurface]:
        current_size = self.get_current_size() if size is None else size
        if not current_size:
            return None

        width_mm, height_mm = current_size

        target_width_px = int(width_mm * pixels_per_mm_x)
        target_height_px = int(height_mm * pixels_per_mm_y)

        return self.renderer.render_to_pixels(
            width=target_width_px, height=target_height_px
        )

    def render_chunk(
        self,
        pixels_per_mm_x: int,
        pixels_per_mm_y: int,
        size: Optional[Tuple[float, float]] = None,
        max_chunk_width: Optional[int] = None,
        max_chunk_height: Optional[int] = None,
        max_memory_size: Optional[int] = None,
    ) -> Generator[Tuple[cairo.ImageSurface, Tuple[float, float]], None, None]:
        natsize = self.get_default_size()
        size = natsize if size is None else size
        if not size:
            return

        width = int(size[0] * pixels_per_mm_x)
        height = int(size[1] * pixels_per_mm_y)

        for chunk in self.renderer.render_chunk(
            width,
            height,
            max_chunk_width=max_chunk_width,
            max_chunk_height=max_chunk_height,
            max_memory_size=max_memory_size,
        ):
            yield chunk

    def dump(self, indent=0):
        print("  " * indent, self.name, self.renderer.label)

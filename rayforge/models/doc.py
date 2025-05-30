import cairo
from typing import List, Optional
from blinker import Signal
from ..config import config
from .workpiece import WorkPiece
from .workplan import WorkPlan


class Doc:
    """
    Represents a loaded Rayforge document.
    """
    workpieces: List[WorkPiece]
    workplan: WorkPlan

    def __init__(self):
        self.workpieces: List[WorkPiece] = []
        self._workpiece_ref_for_pyreverse: WorkPiece
        self.workplan: WorkPlan = WorkPlan(self, "Default plan")
        self.surface: Optional[cairo.ImageSurface] = None
        self.changed = Signal()
        self.workplan.changed.connect(self.changed.send)

    def __iter__(self):
        return iter(self.workpieces)

    def add_workpiece(self, workpiece):
        self.workpieces.append(workpiece)
        self.workplan.set_workpieces(self.workpieces)
        self.changed.send(self)

    def remove_workpiece(self, workpiece):
        if workpiece not in self.workpieces:
            return
        self.workpieces.remove(workpiece)
        self.workplan.set_workpieces(self.workpieces)
        self.changed.send(self)

    def has_workpiece(self):
        return bool(self.workpieces)

    def has_result(self):
        return self.workplan.has_steps() and len(self.workpieces) > 0

    def render(self,
               pixels_per_mm_x: int,
               pixels_per_mm_y: int,
               force: bool = False):
        surface_width_mm, surface_height_mm = config.machine.dimensions
        width = surface_width_mm * pixels_per_mm_x
        height = surface_height_mm * pixels_per_mm_y

        if self.surface \
                and self.surface.get_width() == width \
                and self.surface.get_height() == height \
                and not force:
            return self.surface, False

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        for workpiece in self.workpieces:
            surface, changed = workpiece.render(pixels_per_mm_x,
                                                pixels_per_mm_y,
                                                size=(width, height),
                                                force=force)
            if changed:
                pos_x_mm, pos_y_mm = workpiece.pos or (0, 0)
                pos_x = pos_x_mm * pixels_per_mm_x
                pos_y = pos_y_mm * pixels_per_mm_y
                ctx = cairo.Context(self.surface)
                ctx.set_source_surface(surface, pos_x, pos_y)
                ctx.paint()

        return self.surface, True

    def save_bitmap(self, filename, pixels_per_mm_x, pixels_per_mm_y):
        surface, changed = self.render(pixels_per_mm_x, pixels_per_mm_y)
        surface.write_to_png(filename)

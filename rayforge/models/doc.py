import cairo
from typing import List
from blinker import Signal
from pathlib import Path
from ..config import config
from ..undo import HistoryManager
from .workpiece import WorkPiece
from .workplan import WorkPlan


class Doc:
    """
    Represents a loaded Rayforge document.
    """
    workpieces: List[WorkPiece]
    workplan: WorkPlan
    history_manager: HistoryManager

    def __init__(self):
        self.workpieces: List[WorkPiece] = []
        self._workpiece_ref_for_pyreverse: WorkPiece
        self.workplan: WorkPlan = WorkPlan(self, "Default plan")
        self.history_manager = HistoryManager()
        self.changed = Signal()
        self.workplan.changed.connect(self.changed.send)

    def __iter__(self):
        return iter(self.workpieces)

    def add_workpiece(self, workpiece):
        workpiece.doc = self
        self.workpieces.append(workpiece)
        self.workplan.set_workpieces(self.workpieces)
        self.changed.send(self)

    def remove_workpiece(self, workpiece):
        if workpiece not in self.workpieces:
            return
        workpiece.doc = None
        self.workpieces.remove(workpiece)
        self.workplan.set_workpieces(self.workpieces)
        self.changed.send(self)

    def set_workpieces(self, workpieces: List[WorkPiece]):
        """
        Replaces the entire list of workpieces and notifies listeners.
        """
        for wp in self.workpieces:
            wp.doc = None
        self.workpieces = workpieces
        for wp in self.workpieces:
            wp.doc = self
        self.workplan.set_workpieces(self.workpieces)
        self.changed.send(self)

    def has_workpiece(self):
        return bool(self.workpieces)

    def has_result(self):
        return self.workplan.has_steps() and len(self.workpieces) > 0

    def render(self,
               pixels_per_mm_x: int,
               pixels_per_mm_y: int) -> cairo.ImageSurface:
        """
        Renders the entire document to a new surface.
        """
        surface_width_mm, surface_height_mm = config.machine.dimensions
        width = int(surface_width_mm * pixels_per_mm_x)
        height = int(surface_height_mm * pixels_per_mm_y)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        for workpiece in self.workpieces:
            wp_surface = workpiece.render_for_ops(pixels_per_mm_x,
                                                  pixels_per_mm_y)

            if wp_surface:
                pos_x_mm, pos_y_mm = workpiece.pos or (0, 0)
                pos_x = pos_x_mm * pixels_per_mm_x
                pos_y = pos_y_mm * pixels_per_mm_y

                ctx.set_source_surface(wp_surface, pos_x, pos_y)
                ctx.paint()

        return surface

    def save_bitmap(
        self, filename: Path, pixels_per_mm_x: int, pixels_per_mm_y: int
    ):
        surface = self.render(pixels_per_mm_x, pixels_per_mm_y)
        surface.write_to_png(str(filename))

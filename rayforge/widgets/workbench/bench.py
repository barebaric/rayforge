from gi.repository import Gtk  # type: ignore
import logging
from .axis import Axis
from .workpieceelem import WorkPieceElement
from .workstepelem import WorkStepElement
from .surface import WorkSurface


logger = logging.getLogger(__name__)


class WorkBench(Gtk.Grid):
    """
    A WorkBench wraps the WorkSurface to add an X and Y axis.
    """
    def __init__(self, width_mm, height_mm, **kwargs):
        super().__init__(**kwargs)
        self.axis_thickness = 25
        self.doc = None

        # Create a work area to display the image and paths
        self.surface: "WorkSurface" = WorkSurface(width_mm=width_mm, height_mm=height_mm)
        self.surface.set_hexpand(True)
        self.surface.set_vexpand(True)
        self.surface.set_halign(Gtk.Align.FILL)
        self.surface.set_valign(Gtk.Align.FILL)
        self.attach(self.surface, 1, 0, 1, 1)
        self.surface.set_size(width_mm, height_mm)
        self.surface.elem_removed.connect(self.on_elem_removed)

        # Add the X axis
        self.axis_x = Axis(width_mm,
                           thickness=self.axis_thickness,
                           orientation=Gtk.Orientation.HORIZONTAL)
        self.attach(self.axis_x, 1, 1, 1, 1)

        # Add the Y axis
        self.axis_y = Axis(height_mm,
                           thickness=self.axis_thickness,
                           orientation=Gtk.Orientation.VERTICAL)
        self.attach(self.axis_y, 0, 0, 1, 1)

    def set_size(self, width_mm, height_mm):
        self.surface.set_size(width_mm, height_mm)
        self.axis_x.set_length(width_mm)
        self.axis_y.set_length(height_mm)

    def set_workpieces_visible(self, visible=True):
        self.surface.set_workpieces_visible(visible)

    def set_laser_dot_visible(self, visible):
        self.surface.set_laser_dot_visible(visible)

    def set_laser_dot_position(self, x_mm, y_mm):
        self.surface.set_laser_dot_position(x_mm, y_mm)

    def clear(self):
        self.surface.clear_workpieces()

    def update(self, doc):
        self.doc = doc

        # Remove anything from the canvas that no longer exists.
        for elem in self.surface.find_by_type(WorkStepElement):
            if elem.data not in doc.workplan:
                elem.remove()
        for elem in self.surface.find_by_type(WorkPieceElement):
            if elem.data not in doc:
                elem.remove()

        # Add any new elements.
        for workpiece in doc.workpieces:
            self.surface.add_workpiece(workpiece)
        for workstep in doc.workplan:
            self.surface.add_workstep(workstep)

    def on_elem_removed(self, parent, child):
        if not self.doc:
            return
        self.doc.remove_workpiece(child.data)

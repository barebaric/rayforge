from gi.repository import Gtk, Graphene  # type: ignore
import cairo
import logging
from .worksurface import WorkSurface, WorkPieceElement, WorkStepElement


logger = logging.getLogger(__name__)


class Axis(Gtk.DrawingArea):
    """
    This widget displays a simple axis line with labels.
    """
    def __init__(self,
                 length_mm=100,
                 orientation=Gtk.Orientation.HORIZONTAL,
                 thickness=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.orientation = orientation
        self.length_mm = length_mm
        self.grid_size = 10  # in mm
        self.stroke = 1
        self.label_padding = 2

        # We need a temporary context to figure out the label size.
        temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0)
        temp_context = cairo.Context(temp_surface)
        label = f"{self.length_mm}"
        extents = temp_context.text_extents(label)

        if self.orientation == Gtk.Orientation.HORIZONTAL:
            self.thickness = thickness \
                    or extents.height+2*self.label_padding+self.stroke
            self.set_size_request(-1, self.thickness)
        else:
            self.thickness = thickness \
                    or extents.width+2*self.label_padding+self.stroke
            self.set_size_request(self.thickness, -1)

    def set_length(self, length_mm):
        self.length_mm = length_mm
        self.queue_draw()

    def do_snapshot(self, snapshot):
        # Calculate size in pixels.
        if self.orientation == Gtk.Orientation.HORIZONTAL:
            length = self.get_width()
            start = 0, 0
            end = length, 0
            width, height = length, self.thickness
        else:
            length = self.get_height()
            start = self.thickness, 0
            end = self.thickness, length
            width, height = self.thickness, length

        # Create a Cairo context for the snapshot
        ctx = snapshot.append_cairo(
            Graphene.Rect().init(0, 0, width, height)
        )

        # Draw axis line.
        ctx.set_line_width(self.stroke)
        ctx.set_source_rgb(0, 0, 0)
        ctx.move_to(*start)
        ctx.line_to(*end)
        ctx.stroke()

        # Draw axis labels
        interval = self.grid_size
        for pos in range(interval, int(self.length_mm)+1, interval):
            pos_px = int(pos*length/self.length_mm)
            label = f"{pos}"
            extents = ctx.text_extents(label)
            if self.orientation == Gtk.Orientation.HORIZONTAL:
                if pos_px+int(extents.width/2) >= length:
                    pos_px -= int(extents.width/2)
                ctx.move_to(pos_px-int(extents.width/2),
                            self.stroke+self.label_padding+extents.height)
            else:
                if height-pos_px <= 0:
                    pos_px -= int(extents.height/2)
                ctx.move_to(width-self.stroke-self.label_padding-extents.width,
                            height-pos_px+int(extents.height/2))
            ctx.show_text(label)


class WorkBench(Gtk.Grid):
    """
    A WorkBench wraps the WorkSurface to add an X and Y axis.
    """
    def __init__(self, width_mm, height_mm, **kwargs):
        super().__init__(**kwargs)
        self.axis_thickness = 25
        self.doc = None

        # Create a work area to display the image and paths
        self.surface = WorkSurface(width_mm=width_mm, height_mm=height_mm)
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
        self.doc.remove_workpiece(child.data)

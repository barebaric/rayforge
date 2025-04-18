import math
import logging
from gi.repository import Graphene, GLib
import cairo
from ..models.ops import Ops
from ..opsencoder.cairoencoder import CairoEncoder
from ..config import config
from ..models.workpiece import WorkPiece
from ..models.workplan import WorkStep
from .canvas import Canvas, CanvasElement
logger = logging.getLogger(__name__)


def _copy_surface(source, target, width, height, clip):
    in_width, in_height = source.get_width(), source.get_height()
    scale_x = width/in_width
    scale_y = height/in_height
    ctx = cairo.Context(target)
    clip_x, clip_y, clip_w, clip_h = clip
    ctx.rectangle(0, 0, clip_x+clip_w, clip_y+clip_h)
    ctx.clip()
    ctx.scale(scale_x, scale_y)
    ctx.set_source_surface(source, clip_x, clip_y)
    ctx.paint()
    return target


class WorkPieceElement(CanvasElement):
    """
    WorkPieceElements display WorkPiece objects on the WorkSurface.
    This is the "standard" element used to display workpieces on the
    WorkSurface.
    """

    def __init__(self, workpiece, x_mm, y_mm, width_mm, height_mm, **kwargs):
        super().__init__(x_mm,
                         y_mm,
                         width_mm,
                         height_mm,
                         data=workpiece,
                         **kwargs)

    def set_pos(self, x_mm, y_mm):
        super().set_pos(x_mm, y_mm)
        self.data.set_pos(x_mm, y_mm)

    def set_size(self, width_mm, height_mm):
        super().set_size(width_mm, height_mm)
        self.data.set_size(width_mm, height_mm)
        self.allocate()
        self.dirty = True

    def render(self, clip):
        assert self.surface is not None
        pixels_per_mm_x, pixels_per_mm_y = self.get_pixels_per_mm()
        workpiece = self.data
        surface, changed = workpiece.render(pixels_per_mm_x,
                                            pixels_per_mm_y,
                                            workpiece.size)
        if not changed:
            return
        width, height = self.size_px()
        self.surface = _copy_surface(surface,
                                     self.surface,
                                     width,
                                     height,
                                     clip)


class WorkPieceOpsElement(CanvasElement):
    """Displays the generated Ops for a single WorkPiece."""
    def __init__(self, workpiece, x_mm, y_mm, width_mm, height_mm,
                 **kwargs):
        super().__init__(x_mm,
                         y_mm,
                         width_mm,
                         height_mm,
                         data=workpiece,
                         selectable=False,
                         **kwargs)
        workpiece.changed.connect(self._on_workpiece_changed)
        self._accumulated_ops = Ops()

    def _on_workpiece_changed(self, workpiece: WorkPiece):
        self.set_pos(*workpiece.pos)
        self.set_size(*workpiece.size)
        self.allocate()
        self.canvas.queue_draw()

    def clear_ops(self):
        """Clears the accumulated operations and the drawing surface."""
        self._accumulated_ops = Ops()
        self.clear_surface()
        self.dirty = True

    def add_ops(self, ops_chunk: Ops):
        """Adds a chunk of operations to the accumulated total."""
        if not ops_chunk:
            return
        self._accumulated_ops += ops_chunk
        self.dirty = True

    def render(self, clip):
        """Renders the accumulated Ops to the element's surface."""
        super().render(clip)
        if not self.parent:
            return

        # Replace the current bitmap by the rendered accumulated Ops.
        self.clear_surface()
        if not self._accumulated_ops:
            return

        pixels_per_mm = self.get_pixels_per_mm()
        encoder = CairoEncoder()
        show_travel = self.canvas.show_travel_moves if self.canvas else False
        encoder.encode(self._accumulated_ops,
                       config.machine,
                       self.surface,
                       pixels_per_mm,
                       show_travel_moves=show_travel)


class WorkStepElement(CanvasElement):
    """
    WorkStepElements display the result of a WorkStep on the
    WorkSurface. The output represents the laser path.
    """
    def __init__(self, workstep, x_mm, y_mm, width_mm, height_mm, **kwargs):
        super().__init__(x_mm,
                         y_mm,
                         width_mm,
                         height_mm,
                         data=workstep,
                         selectable=False,
                         **kwargs)
        workstep.changed.connect(self._on_workstep_changed)
        # Connect to the actual signals from WorkStep's async pipeline
        workstep.ops_generation_starting.connect(
            self._on_ops_generation_starting
        )
        # Note: There is no explicit 'cleared' signal in the async pipeline,
        # starting implies clearing for the UI representation.
        workstep.ops_chunk_available.connect(
            self._on_ops_chunk_available
        )
        workstep.ops_generation_finished.connect(
            self._on_ops_generation_finished
        )
        for workpiece in workstep.workpieces():
            self.add_workpiece(workpiece)

    def add_workpiece(self, workpiece):
        elem = self.find_by_data(workpiece)
        if elem:
            elem.dirty = True
            return elem
        elem = WorkPieceOpsElement(workpiece,
                                   *workpiece.pos,
                                   *workpiece.size,
                                   canvas=self.canvas,
                                   parent=self)
        self.add(elem)
        return elem

    def _on_workstep_changed(self, step: WorkStep):
        for elem in self.children:
            if elem.data not in step.workpieces():
                elem.remove()
        # We do not need to add new workpieces here, because they are
        # dynamically added once the Ops is ready in _on_ops_changed()

    def _find_or_add_workpiece_elem(
            self, workpiece: WorkPiece) -> WorkPieceOpsElement | None:
        """Finds the element for a workpiece, creating if necessary."""
        elem = self.find_by_data(workpiece)
        if not elem:
            # This might happen if the workpiece was added *during* generation
            # Although ideally _on_workstep_changed handles workpiece
            # additions/removals before generation starts. Add defensively.
            elem = self.add_workpiece(workpiece)
        return elem

    def _on_ops_generation_starting(self, sender: WorkStep,
                                    workpiece: WorkPiece):
        """Called before ops generation starts for a workpiece."""
        logger.debug(
            f"WorkStepElem '{sender.name}': Received ops_generation_starting "
            f"for {workpiece.name}"
        )
        elem = self._find_or_add_workpiece_elem(workpiece)
        if elem:
            # Ensure it's clean before starting
            logger.debug(
                f"WorkStepElem '{sender.name}': Calling clear_ops() for "
                f"{workpiece.name}"
            )
            elem.clear_ops()
            if self.canvas:
                # Initial clear needs a redraw
                logger.debug(
                    f"WorkStepElem '{sender.name}': Calling queue_draw() via "
                    f"idle_add for {workpiece.name}"
                )
                GLib.idle_add(self.canvas.queue_draw)

    # Removed _on_ops_cleared as there's no corresponding signal in the
    # async pipeline. Clearing happens in _on_ops_generation_starting.

    def _on_ops_chunk_available(self, sender: WorkStep, workpiece: WorkPiece,
                                chunk: Ops):
        """Called when a chunk of ops is available for a workpiece."""
        logger.debug(
            f"WorkStepElem '{sender.name}': Received ops_chunk_available for "
            f"{workpiece.name} (chunk size: {len(chunk)})"
        )
        elem = self._find_or_add_workpiece_elem(workpiece)
        if elem:
            # The chunk includes initial/final commands, add them directly.
            # Translation is handled within the WorkStep generation now.
            elem.add_ops(chunk)
            if self.canvas:
                # Trigger redraw incrementally
                logger.debug(
                    f"WorkStepElem '{sender.name}': Calling queue_draw() via "
                    f"idle_add for chunk {workpiece.name}"
                )
                GLib.idle_add(self.canvas.queue_draw)

    def _on_ops_generation_finished(self, sender: WorkStep,
                                      workpiece: WorkPiece):
        """Called when ops generation is finished for a workpiece."""
        # Final redraw is triggered by the last _on_ops_chunk_available call's
        # queue_draw. No extra action needed here unless we add UI
        # indicators for processing state (e.g., hide a spinner).
        pass


class LaserDotElement(CanvasElement):
    """
    Draws a simple red dot.
    """
    def __init__(self, radius_mm, **kwargs):
        self.radius_mm = radius_mm
        super().__init__(0,
                         0,
                         2*radius_mm,
                         2*radius_mm,
                         visible=True,
                         selectable=False,
                         **kwargs)

    def render(self, clip):
        super().render(clip)
        if not self.parent:
            return

        self.clear_surface()
        pixels_per_mm_x, _ = self.get_pixels_per_mm()
        ctx = cairo.Context(self.surface)
        ctx.set_hairline(True)
        ctx.set_source_rgb(.9, 0, 0)
        radius = self.width_mm/2*pixels_per_mm_x
        ctx.arc(radius, radius, radius-1, 0., 2*math.pi)
        ctx.fill()


class WorkSurface(Canvas):
    """
    The WorkSurface displays a grid area with WorkPieces and
    WorkPieceOpsElements according to real world dimensions.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.show_travel_moves = False
        self.workpiece_elements = CanvasElement(
            *self.root.rect(),
            selectable=False
        )
        self.root.add(self.workpiece_elements)
        self.laser_dot = LaserDotElement(1)
        self.set_laser_dot_position(0, 0)
        self.root.add(self.laser_dot)
        self.grid_size = 10  # in mm
        self.update()

    def set_show_travel_moves(self, show: bool):
        """Sets whether to display travel moves and triggers a redraw."""
        if self.show_travel_moves != show:
            self.show_travel_moves = show
            # Mark elements dirty that depend on this setting
            for elem in self.find_by_type(WorkStepElement):
                elem.dirty = True
            self.queue_draw()

    def set_size(self, width_mm, height_mm):
        self.root.set_size(width_mm, height_mm)
        for elem in self.find_by_type(WorkStepElement):
            elem.set_size(width_mm, height_mm)
        self.update()

    def update(self):
        self.aspect_ratio = self.root.width_mm/self.root.height_mm
        self.workpiece_elements.set_size(self.root.width_mm,
                                         self.root.height_mm)
        self.root.allocate()
        self.queue_draw()

    def add_workstep(self, workstep):
        """
        Adds the workstep, but only if it does not yet exist.
        Also adds each of the WorkPieces, but only if they
        do not exist.
        """
        # Add or find the WorkStep.
        if not self.find_by_data(workstep):
            elem = WorkStepElement(workstep,
                                   *self.root.rect(),
                                   canvas=self,
                                   parent=self.root)
            self.add(elem)
            workstep.changed.connect(self.on_workstep_changed)
        self.queue_draw()

    def set_laser_dot_visible(self, visible=True):
        self.laser_dot.set_visible(visible)
        self.queue_draw()

    def set_laser_dot_position(self, x_mm, y_mm):
        height_mm = self.size()[1]
        dot_radius_mm = self.laser_dot.radius_mm
        self.laser_dot.set_pos(x_mm-dot_radius_mm,
                               height_mm-y_mm-dot_radius_mm)
        self.queue_draw()

    def on_workstep_changed(self, workstep, **kwargs):
        elem = self.find_by_data(workstep)
        if not elem:
            return
        elem.set_visible(workstep.visible)
        self.queue_draw()

    def add_workpiece(self, workpiece):
        """
        Adds a workpiece.
        """
        if self.workpiece_elements.find_by_data(workpiece):
            self.queue_draw()
            return
        width_mm, height_mm = workpiece.get_default_size()
        elem = WorkPieceElement(workpiece,
                                self.root.width_mm/2-width_mm/2,
                                self.root.height_mm/2-height_mm/2,
                                width_mm,
                                height_mm)
        self.workpiece_elements.add(elem)
        self.queue_draw()

    def clear_workpieces(self):
        self.workpiece_elements.clear()
        self.queue_draw()

    def clear(self):
        self.root.clear()
        self.queue_draw()

    def find_by_type(self, thetype):
        return [c for c in self.root.children if isinstance(c, thetype)]

    def set_workpieces_visible(self, visible=True):
        self.workpiece_elements.set_visible(visible)
        self.queue_draw()

    def do_snapshot(self, snapshot):
        logger.debug("WorkSurface: do_snapshot called (actual redraw)")
        # Create a Cairo context for the snapshot
        width, height = self.get_width(), self.get_height()
        bounds = Graphene.Rect().init(0, 0, width, height)
        ctx = snapshot.append_cairo(bounds)

        self.pixels_per_mm_x = width/self.root.width_mm
        self.pixels_per_mm_y = height/self.root.height_mm
        self._draw_grid(ctx, width, height)

        # The tree of elements in the canvas looks like this:
        # root (CanvasElement)
        #   workpieces (CanvasElement)
        #     workpiece (WorkPieceElement)
        #     ...       (WorkPieceElement)
        #   workstep    (WorkStepElement)
        #   ...         (WorkStepElement)
        # When a workpiece moves or is resized, we need to ensure
        # that the worksteps update in sync with them.
        # For now, to achieve that we force worksteps to update
        # always, by marking them dirty.
        for elem in self.find_by_type(WorkStepElement):
            elem.dirty = True

        super().do_snapshot(snapshot)

    def _draw_grid(self, ctx, width, height):
        """
        Draw scales on the X and Y axes.
        """
        # Draw vertical lines
        for x in range(0, int(self.root.width_mm)+1, self.grid_size):
            x_px = x*self.pixels_per_mm_x
            ctx.move_to(x_px, 0)
            ctx.line_to(x_px, height)
            ctx.set_source_rgb(.9, .9, .9)
            ctx.stroke()

        # Draw horizontal lines
        for y in range(int(self.root.height_mm), -1, -self.grid_size):
            y_px = y*self.pixels_per_mm_y
            ctx.move_to(0, y_px)
            ctx.line_to(width, y_px)
            ctx.set_source_rgb(.9, .9, .9)
            ctx.stroke()

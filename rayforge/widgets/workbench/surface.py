import math
import logging
from blinker import Signal
from typing import Optional, Tuple, cast, Dict, List
from gi.repository import Graphene, Gdk, Gtk  # type: ignore
from ...models.doc import Doc
from ...models.workpiece import WorkPiece
from ...models.machine import Machine
from ...undo import SetterCommand, ListItemCommand
from ..canvas import Canvas, CanvasElement
from ..canvas.selection import MultiSelectionGroup
from .axis import AxisRenderer
from .dotelem import DotElement
from .workstepelem import WorkStepElement
from .workpieceelem import WorkPieceElement
from .cameraelem import CameraImageElement


logger = logging.getLogger(__name__)


class WorkSurface(Canvas):
    """
    The WorkSurface displays a grid area with WorkPieces and
    WorkPieceOpsElements according to real world dimensions.
    """

    # The minimum allowed zoom level, relative to the "fit-to-view" size
    # (zoom=1.0). 0.1 means you can zoom out until the view is 10% of its
    # "fit" size.
    MIN_ZOOM_FACTOR = 0.1

    # The maximum allowed pixel density when zooming in.
    MAX_PIXELS_PER_MM = 100.0

    def __init__(
        self, doc: Doc, machine: Machine, cam_visible: bool = False, **kwargs
    ):
        logger.debug("WorkSurface.__init__ called")
        super().__init__(**kwargs)
        self.doc = doc
        self.machine = machine
        self.zoom_level = 1.0
        self._show_travel_moves = False
        self.width_mm, self.height_mm = machine.dimensions
        self.pixels_per_mm_x = 0.0
        self.pixels_per_mm_y = 0.0
        self._cam_visible = cam_visible
        self._transform_start_states: Dict[CanvasElement, dict] = {}

        # The root element itself should not clip, allowing its children
        # (like _workpiece_elements) to draw outside its bounds.
        self.root.clip = False

        self._axis_renderer = AxisRenderer(
            width_mm=self.width_mm,
            height_mm=self.height_mm,
            zoom_level=self.zoom_level,
        )
        self.root.background = 0.8, 0.8, 0.8, 0.1  # light gray background

        # This container for workpieces should not clip its children.
        self._workpiece_elements = CanvasElement(
            0, 0, 0, 0, selectable=False, clip=False
        )
        self.root.add(self._workpiece_elements)

        # DotElement size will be set in pixels by WorkSurface
        # Initialize with zero size, size and position will be set in
        # do_size_allocate
        self._laser_dot = DotElement(0, 0, 0, 0)
        self.root.add(self._laser_dot)

        # Add scroll event controller for zoom
        self._scroll_controller = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        self._scroll_controller.connect("scroll", self.on_scroll)
        self.add_controller(self._scroll_controller)

        # Add middle click gesture for panning
        self._pan_gesture = Gtk.GestureDrag.new()
        self._pan_gesture.set_button(Gdk.BUTTON_MIDDLE)
        self._pan_gesture.connect("drag-begin", self.on_pan_begin)
        self._pan_gesture.connect("drag-update", self.on_pan_update)
        self._pan_gesture.connect("drag-end", self.on_pan_end)
        self.add_controller(self._pan_gesture)
        self._pan_start = 0, 0

        # This is hacky, but what to do: The EventControllerScroll provides
        # no access to any mouse position, and there is no easy way to
        # get the mouse position in Gtk4. So I have to store it here and
        # track the motion event...
        self._mouse_pos = 0, 0

        # Signals for clipboard and duplication operations
        self.cut_requested = Signal()
        self.copy_requested = Signal()
        self.paste_requested = Signal()
        self.duplicate_requested = Signal()

        # Connect to undo/redo signals from the canvas
        self.move_begin.connect(self._on_any_transform_begin)
        self.move_end.connect(self._on_move_end)
        self.resize_begin.connect(self._on_any_transform_begin)
        self.resize_end.connect(self._on_resize_end)
        self.rotate_begin.connect(self._on_any_transform_begin)
        self.rotate_end.connect(self._on_rotate_end)
        self.elements_deleted.connect(self._on_elements_deleted)

        # Add CameraImageElements for each camera
        self.machine.changed.connect(self._on_machine_changed)
        self._on_machine_changed(machine)

        # Connect to the history manager's changed signal to sync the view
        # globally, which is necessary for undo/redo actions triggered
        # outside of this widget.
        self.doc.history_manager.changed.connect(self._on_history_changed)

    def _on_history_changed(self, sender, **kwargs):
        """
        Called when the undo/redo history changes. This handler acts as a
        synchronizer to fix state timing issues.
        """
        # Rebuild the entire selection state.
        selected_elements = self.get_selected_elements()
        self._update_selection_state(selected_elements)
        self.queue_draw()

    def _on_any_transform_begin(self, sender, elements: List[CanvasElement]):
        self._transform_start_states.clear()
        for element in elements:
            if not isinstance(element.data, WorkPiece):
                continue
            workpiece: WorkPiece = element.data
            self._transform_start_states[element] = {
                "pos": workpiece.pos,
                "size": workpiece.size,
                "angle": workpiece.angle,
            }

    def _on_move_end(self, sender, elements: List[CanvasElement]):
        history = self.doc.history_manager
        with history.transaction(_("Move workpiece(s)")) as t:
            for element in elements:
                if (
                    not isinstance(element.data, WorkPiece)
                    or element not in self._transform_start_states
                ):
                    continue
                workpiece: WorkPiece = element.data
                start_state = self._transform_start_states[element]
                if workpiece.pos and start_state["pos"] != workpiece.pos:
                    t.add(SetterCommand(
                        workpiece,
                        "set_pos",
                        workpiece.pos,
                        start_state["pos"],
                    ))

        self._transform_start_states.clear()

    def _on_rotate_end(self, sender, elements: List[CanvasElement]):
        history = self.doc.history_manager
        with history.transaction(_("Rotate workpiece(s)")) as t:
            for element in elements:
                if (
                    not isinstance(element.data, WorkPiece)
                    or element not in self._transform_start_states
                ):
                    continue
                workpiece: WorkPiece = element.data
                start_state = self._transform_start_states[element]
                if start_state["angle"] != workpiece.angle:
                    t.add(SetterCommand(
                        workpiece,
                        "set_angle",
                        (workpiece.angle,),
                        (start_state["angle"],),
                    ))

        self._transform_start_states.clear()

    def _on_resize_end(self, sender, elements: List[CanvasElement]):
        history = self.doc.history_manager
        with history.transaction(_("Resize workpiece(s)")) as t:
            for element in elements:
                if (
                    not isinstance(element.data, WorkPiece)
                    or element not in self._transform_start_states
                ):
                    continue
                workpiece: WorkPiece = element.data
                start_state = self._transform_start_states[element]
                if workpiece.pos and start_state["pos"] != workpiece.pos:
                    t.add(
                        SetterCommand(
                            workpiece,
                            "set_pos",
                            workpiece.pos,
                            start_state["pos"],
                        )
                    )
                if workpiece.size and start_state["size"] != workpiece.size:
                    t.add(
                        SetterCommand(
                            workpiece,
                            "set_size",
                            workpiece.size,
                            start_state["size"],
                        )
                    )

        self._transform_start_states.clear()

    def _on_elements_deleted(self, sender, elements: List[CanvasElement]):
        workpieces_to_delete = [
            elem.data for elem in elements if isinstance(elem.data, WorkPiece)
        ]

        if not workpieces_to_delete:
            return

        history = self.doc.history_manager
        with history.transaction(_("Delete workpiece(s)")) as t:
            for wp in workpieces_to_delete:
                cmd = ListItemCommand(
                    owner_obj=self.doc,
                    item=wp,
                    undo_command="add_workpiece",
                    redo_command="remove_workpiece",
                    name=_("Delete workpiece"),
                )
                t.add(cmd)

    def on_button_press(self, gesture, n_press: int, x: int, y: int):
        # First, let the parent Canvas handle the event to determine if a
        # resize is starting and to set self._active_elem and self._resizing.
        super().on_button_press(gesture, n_press, x, y)

        # If a resize operation has started on a WorkPieceElement, hide the
        # corresponding ops elements to improve performance.
        if self._resizing and (self._active_elem or self._selection_group):
            elements_in_transform = self.get_selected_elements()
            for element in elements_in_transform:
                if not isinstance(element.data, WorkPiece):
                    continue
                workpiece_data = element.data
                for step_elem in self.find_by_type(WorkStepElement):
                    ops_elem = step_elem.find_by_data(workpiece_data)
                    if ops_elem:
                        ops_elem.set_visible(False)

    def on_button_release(self, gesture, x: float, y: float):
        # Before the parent class resets the resizing state, check if a resize
        # was in progress on a WorkPieceElement.
        workpieces_to_update = []
        if self._resizing and (self._active_elem or self._selection_group):
            elements_in_transform = self.get_selected_elements()
            for element in elements_in_transform:
                if isinstance(element.data, WorkPiece):
                    workpieces_to_update.append(element.data)

        # Let the parent class finish the drag/resize operation.
        super().on_button_release(gesture, x, y)

        # If a resize has just finished, make the ops visible again and
        # trigger a re-allocation and re-render to reflect the new size.
        if workpieces_to_update:
            for workpiece in workpieces_to_update:
                for step_elem in self.find_by_type(WorkStepElement):
                    ops_elem = step_elem.find_by_data(workpiece)
                    if ops_elem:
                        ops_elem.set_visible(True)

    def set_pan(self, pan_x_mm: float, pan_y_mm: float):
        """Sets the pan position in mm and updates the axis renderer."""
        self._axis_renderer.set_pan_x_mm(pan_x_mm)
        self._axis_renderer.set_pan_y_mm(pan_y_mm)
        self._recalculate_sizes()
        self.queue_draw()

    def _get_base_pixels_per_mm(self) -> Tuple[float, float]:
        """Calculates the pixels/mm for a zoom level of 1.0 (fit-to-view)."""
        width, height_pixels = self.get_width(), self.get_height()
        if not all([width, height_pixels, self.width_mm, self.height_mm]):
            return 1.0, 1.0  # Avoid division by zero at startup

        y_axis_pixels = self._axis_renderer.get_y_axis_width()
        x_axis_height = self._axis_renderer.get_x_axis_height()
        right_margin = math.ceil(y_axis_pixels / 2)
        top_margin = math.ceil(x_axis_height / 2)
        content_width_px = width - y_axis_pixels - right_margin
        content_height_px = height_pixels - x_axis_height - top_margin

        base_ppm_x = (
            content_width_px / self.width_mm if self.width_mm > 0 else 0
        )
        base_ppm_y = (
            content_height_px / self.height_mm if self.height_mm > 0 else 0
        )
        return base_ppm_x, base_ppm_y

    def set_zoom(self, zoom_level: float):
        """
        Sets the zoom level and updates the axis renderer.
        The caller is responsible for ensuring the zoom_level is clamped.
        """
        self.zoom_level = zoom_level
        self._axis_renderer.set_zoom(self.zoom_level)
        self.root.mark_dirty(recursive=True)
        self.do_size_allocate(self.get_width(), self.get_height(), 0)
        self.queue_draw()

    def set_size(self, width_mm: float, height_mm: float):
        """
        Sets the real-world size of the work surface in mm
        and updates related properties.
        """
        self.width_mm = width_mm
        self.height_mm = height_mm
        self._axis_renderer.set_width_mm(self.width_mm)
        self._axis_renderer.set_height_mm(self.height_mm)
        self.queue_draw()

    def get_size(self) -> Tuple[float, float]:
        """Returns the size of the work surface in mm."""
        return self.width_mm, self.height_mm

    def on_motion(self, gesture, x: int, y: int):
        self._mouse_pos = x, y
        return super().on_motion(gesture, x, y)

    def pixel_to_mm(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Converts pixel coordinates to real-world mm."""
        height_pixels = self.get_height()
        y_axis_pixels = self._axis_renderer.get_y_axis_width()
        x_axis_height = self._axis_renderer.get_x_axis_height()
        top_margin = math.ceil(x_axis_height / 2)
        x_mm = self._axis_renderer.pan_x_mm + (
            (x_px - y_axis_pixels) / self.pixels_per_mm_x
        )
        y_mm = (
            self._axis_renderer.pan_y_mm
            + (height_pixels - y_px - top_margin) / self.pixels_per_mm_y
        )
        return x_mm, y_mm

    def on_scroll(self, controller, dx: float, dy: float):
        """Handles the scroll event for zoom."""
        zoom_speed = 0.1

        # 1. Calculate a desired new zoom level based on scroll direction
        if dy > 0:  # Scroll down - zoom out
            desired_zoom = self.zoom_level * (1 - zoom_speed)
        else:  # Scroll up - zoom in
            desired_zoom = self.zoom_level * (1 + zoom_speed)

        # 2. Get the base "fit-to-view" pixel density (for zoom = 1.0)
        base_ppm_x, base_ppm_y = self._get_base_pixels_per_mm()
        if base_ppm_x <= 0 or base_ppm_y <= 0:
            return  # Cannot calculate zoom limits yet (e.g., at startup)

        # Use the smaller base density for consistent limit calculations
        base_ppm = min(base_ppm_x, base_ppm_y)

        # 3. Calculate the pixel density limits
        # The minimum density is based on our zoom factor.
        min_ppm = base_ppm * self.MIN_ZOOM_FACTOR
        # The maximum density is a fixed constant.
        max_ppm = self.MAX_PIXELS_PER_MM

        # 4. Calculate the target density and clamp it within our limits
        target_ppm = base_ppm * desired_zoom
        clamped_ppm = max(min_ppm, min(target_ppm, max_ppm))

        # 5. Convert the valid, clamped density back into a final zoom level
        final_zoom = clamped_ppm / base_ppm

        # If the zoom level is already at its limit and won't change, do
        # nothing.
        if abs(final_zoom - self.zoom_level) < 1e-9:
            return

        # 6. Calculate pan adjustment to zoom around the mouse cursor
        mouse_x_px, mouse_y_px = self._mouse_pos
        focus_x_mm, focus_y_mm = self.pixel_to_mm(mouse_x_px, mouse_y_px)

        newpixels_per_mm_x = base_ppm_x * final_zoom
        newpixels_per_mm_y = base_ppm_y * final_zoom

        height_pixels = self.get_height()
        y_axis_pixels = self._axis_renderer.get_y_axis_width()
        x_axis_height = self._axis_renderer.get_x_axis_height()
        top_margin = math.ceil(x_axis_height / 2)

        if newpixels_per_mm_x > 0 and newpixels_per_mm_y > 0:
            new_pan_x_mm = (
                focus_x_mm - (mouse_x_px - y_axis_pixels) / newpixels_per_mm_x
            )
            new_pan_y_mm = (
                focus_y_mm
                - (height_pixels - mouse_y_px - top_margin)
                / newpixels_per_mm_y
            )
            self.set_pan(new_pan_x_mm, new_pan_y_mm)

        # 7. Apply the final, clamped zoom level.
        self.set_zoom(final_zoom)

    def _recalculate_sizes(self):
        origin_x, origin_y = self._axis_renderer.get_origin()
        content_width, content_height = self._axis_renderer.get_content_size()

        # Set the root element's size directly in pixels
        self.root.set_pos(origin_x, origin_y - content_height)
        self.root.set_size(content_width, content_height)

        # Update WorkSurface's internal pixel dimensions based on content area
        self.pixels_per_mm_x, self.pixels_per_mm_y = (
            self._axis_renderer.get_pixels_per_mm()
        )

        # Update the workpiece element group and WorkStepElement group sizes:
        # they should always match root group size
        content_width, content_height = self._axis_renderer.get_content_size()
        self._workpiece_elements.set_size(content_width, content_height)
        for elem in self.find_by_type(WorkStepElement):
            elem.set_size(content_width, content_height)

        # Update CameraImageElement sizes
        for elem in self.find_by_type(CameraImageElement):
            elem.set_size(content_width, content_height)

        # Update laser dot size based on new pixel dimensions and its mm radius
        dot_radius_mm = self._laser_dot.radius_mm
        dot_diameter_px = round(2 * dot_radius_mm * self.pixels_per_mm_x)
        self._laser_dot.set_size(dot_diameter_px, dot_diameter_px)

        # Re-position laser dot based on new pixel dimensions
        current_dot_pos_px = self._laser_dot.pos_abs()
        current_dot_pos_mm = self._laser_dot.pixel_to_mm(*current_dot_pos_px)
        self.set_laser_dot_position(*current_dot_pos_mm)

    def do_size_allocate(self, width: int, height: int, baseline: int):
        """Handles canvas size allocation in pixels."""
        # Calculate grid bounds using AxisRenderer
        self._axis_renderer.set_width_px(width)
        self._axis_renderer.set_height_px(height)
        self._recalculate_sizes()
        self.root.allocate()

    def set_show_travel_moves(self, show: bool):
        """Sets whether to display travel moves and triggers re-rendering."""
        if self._show_travel_moves != show:
            self._show_travel_moves = show
            # Propagate the change to all existing WorkStepElements
            for elem in self.find_by_type(WorkStepElement):
                elem = cast(WorkStepElement, elem)
                elem.set_show_travel_moves(show)

    def update_from_doc(self, doc: Doc):
        self.doc = doc

        # Remove anything from the canvas that no longer exists.
        for elem in self.find_by_type(WorkStepElement):
            if elem.data not in doc.workplan:
                elem.remove()
        for elem in self.find_by_type(WorkPieceElement):
            if elem.data not in doc:
                elem.remove()

        # Add any new elements.
        for workpiece in doc.workpieces:
            self.add_workpiece(workpiece)
        for workstep in doc.workplan:
            self.add_workstep(workstep)

    def add_workstep(self, workstep):
        """
        Adds the workstep, but only if it does not yet exist.
        Also adds each of the WorkPieces, but only if they
        do not exist.
        """
        # Add or find the WorkStep.
        elem = cast(WorkStepElement, self.find_by_data(workstep))
        if not elem:
            # WorkStepElement should cover the entire canvas area in pixels
            elem = WorkStepElement(
                workstep,
                0,  # x_px
                0,  # y_px
                self.root.width,  # width_px
                self.root.height,  # height_px
                canvas=self,
                parent=self.root,
                show_travel_moves=self._show_travel_moves,
            )
            self.add(elem)
            workstep.changed.connect(self.on_workstep_changed)
        self.queue_draw()

        # Ensure WorkPieceOpsElements are created for each WorkPiece
        for workpiece in workstep.workpieces():
            elem.add_workpiece(workpiece)

    def set_laser_dot_visible(self, visible=True):
        self._laser_dot.set_visible(visible)
        self.queue_draw()

    def set_laser_dot_position(self, x_mm, y_mm):
        """Sets the laser dot position in real-world mm."""
        # LaserDotElement is sized to represent the dot diameter in pixels.
        # Its position should be the top-left corner of its bounding box.
        # We want the center of the dot to be at (x_px, y_px).
        x_px, y_px = self._laser_dot.mm_to_pixel(x_mm, y_mm)
        dot_width_px = self._laser_dot.width
        self._laser_dot.set_pos(
            round(x_px - dot_width_px / 2), round(y_px - dot_width_px / 2)
        )
        self.queue_draw()

    def on_workstep_changed(self, workstep, **kwargs):
        elem = self.find_by_data(workstep)
        if not elem:
            return
        elem.set_visible(workstep.visible)
        self.queue_draw()

    def add_workpiece(self, workpiece: WorkPiece):
        """
        Adds a workpiece to the canvas.
        If the workpiece does not have a position and size, it calculates a
        sensible default (scaled to fit and centered). Otherwise, it uses
        the existing properties.
        """
        if self._workpiece_elements.find_by_data(workpiece):
            self.queue_draw()
            return

        # If the workpiece is new (e.g., from a file import) and has no
        # position or size, calculate defaults.
        if workpiece.pos is None or workpiece.size is None:
            wswidth_mm, wsheight_mm = self.get_size()
            wp_width_nat_mm, wp_height_nat_mm = workpiece.get_default_size(
                wswidth_mm, wsheight_mm
            )

            # Determine the size to use in mm, scaling down if necessary to fit
            width_mm = wp_width_nat_mm
            height_mm = wp_height_nat_mm
            if width_mm > wswidth_mm or height_mm > wsheight_mm:
                scale_w = wswidth_mm / width_mm if width_mm > 0 else 1
                scale_h = wsheight_mm / height_mm if height_mm > 0 else 1
                scale = min(scale_w, scale_h)
                width_mm *= scale
                height_mm *= scale

            # Set the workpiece's size and centered position in mm
            workpiece.set_size(width_mm, height_mm)
            x_mm = (wswidth_mm - width_mm) / 2
            y_mm = (wsheight_mm - height_mm) / 2
            workpiece.set_pos(x_mm, y_mm)

        # Now that the workpiece is guaranteed to have a pos and size,
        # create its canvas element representation.
        elem = WorkPieceElement(
            workpiece, canvas=self, parent=self._workpiece_elements
        )
        self._workpiece_elements.add(elem)
        self.queue_draw()

    def clear_workpieces(self):
        self._workpiece_elements.remove_all()
        self.queue_draw()
        self.active_element_changed.send(self, element=None)

    def remove_all(self):
        # Clear all children except the fixed ones
        # (_workpiece_elements, _laser_dot)
        children_to_remove = [
            c
            for c in self.root.children
            if c not in [self._workpiece_elements, self._laser_dot]
        ]
        for child in children_to_remove:
            child.remove()
        # Clear children of _workpiece_elements
        self._workpiece_elements.remove_all()
        self.queue_draw()

    def find_by_type(self, thetype):
        """
        Search recursively through the root's children
        """
        return self.root.find_by_type(thetype)

    def set_workpieces_visible(self, visible=True):
        self._workpiece_elements.set_visible(visible)
        self.queue_draw()

    def set_camera_image_visibility(self, visible: bool):
        self._cam_visible = visible
        for elem in self.find_by_type(CameraImageElement):
            elem.set_visible(visible)
        self.queue_draw()

    def _on_machine_changed(self, machine, **kwargs):
        logger.debug("WorkSurface: Machine changed, updating camera elements.")
        # Get current camera elements on the canvas
        current_camera_elements = {}
        for elem in self.find_by_type(CameraImageElement):
            elem = cast(CameraImageElement, elem)
            current_camera_elements[elem.camera] = elem

        # Add new camera elements
        for camera in self.machine.cameras:
            if camera not in current_camera_elements:
                camera_image_elem = CameraImageElement(camera)
                camera_image_elem.set_visible(self._cam_visible)
                self.root.insert(0, camera_image_elem)
                logger.debug(
                    f"Added CameraImageElement for camera {camera.name}"
                )

        # Remove camera elements that no longer exist in the machine
        cameras_in_machine = {camera for camera in self.machine.cameras}
        for camera_instance, elem in list(current_camera_elements.items()):
            if camera_instance not in cameras_in_machine:
                elem.remove()
                logger.debug(
                    "Removed CameraImageElement for camera "
                    f"{camera_instance.name}"
                )

    def do_snapshot(self, snapshot):
        # Create a Cairo context for the snapshot
        width, height = self.get_width(), self.get_height()
        bounds = Graphene.Rect().init(0, 0, width, height)
        ctx = snapshot.append_cairo(bounds)

        # Draw grid, axis, and labels first, so they are in the background.
        self._axis_renderer.draw_grid(ctx)
        self._axis_renderer.draw_axes_and_labels(ctx)

        # Use the parent Canvas's recursive rendering.
        super().do_snapshot(snapshot)

    def on_key_pressed(
        self, controller, keyval: int, keycode: int, state: Gdk.ModifierType
    ) -> bool:
        """Handles key press events for the work surface."""
        # Reset pan and zoom with '1'
        if keyval == Gdk.KEY_1:
            self.set_pan(0.0, 0.0)
            self.set_zoom(1.0)
            return True  # Event handled

        # Handle clipboard and duplication
        is_ctrl = bool(state & Gdk.ModifierType.CONTROL_MASK)
        if is_ctrl:
            selected_wps = self.get_selected_workpieces()
            if keyval == Gdk.KEY_x:
                if selected_wps:
                    self.cut_requested.send(self, workpieces=selected_wps)
                    return True
            elif keyval == Gdk.KEY_c:
                if selected_wps:
                    self.copy_requested.send(self, workpieces=selected_wps)
                    return True
            elif keyval == Gdk.KEY_v:
                self.paste_requested.send(self)
                return True
            elif keyval == Gdk.KEY_d:
                if selected_wps:
                    self.duplicate_requested.send(
                        self, workpieces=selected_wps
                    )
                    return True
            elif keyval == Gdk.KEY_a:
                # Select all workpieces
                all_workpieces = self.doc.workpieces
                if all_workpieces:
                    self.select_workpieces(all_workpieces)

        # Handle arrow key movement for selected workpieces
        is_shift = bool(state & Gdk.ModifierType.SHIFT_MASK)

        move_amount_mm = 1.0
        if is_shift:
            move_amount_mm *= 10
        elif is_ctrl:
            move_amount_mm *= 0.1

        move_amount_x_mm = 0.0
        move_amount_y_mm = 0.0

        if keyval == Gdk.KEY_Up:
            move_amount_y_mm = move_amount_mm
        elif keyval == Gdk.KEY_Down:
            move_amount_y_mm = -move_amount_mm
        elif keyval == Gdk.KEY_Left:
            move_amount_x_mm = -move_amount_mm
        elif keyval == Gdk.KEY_Right:
            move_amount_x_mm = move_amount_mm

        if move_amount_x_mm != 0 or move_amount_y_mm != 0:
            selected_wps = self.get_selected_workpieces()
            if not selected_wps:
                return True  # Consume event but do nothing

            history = self.doc.history_manager
            with history.transaction(_("Move the workpiece")) as t:
                for wp in selected_wps:
                    old_pos = wp.pos
                    if old_pos:
                        new_pos = (
                            old_pos[0] + move_amount_x_mm,
                            old_pos[1] + move_amount_y_mm,
                        )
                        cmd = SetterCommand(
                            wp, "set_pos", new_args=new_pos, old_args=old_pos
                        )
                        t.execute(cmd)
            return True

        # Propagate to parent Canvas for its default behavior (e.g., Shift/
        # Ctrl)
        return super().on_key_pressed(controller, keyval, keycode, state)

    def on_pan_begin(self, gesture, x, y):
        self._pan_start = (
            self._axis_renderer.pan_x_mm,
            self._axis_renderer.pan_y_mm,
        )

    def on_pan_update(self, gesture, x, y):
        # Calculate pan offset based on drag delta
        offset = gesture.get_offset()
        new_pan_x_mm = self._pan_start[0] - offset.x / self.pixels_per_mm_x
        new_pan_y_mm = self._pan_start[1] + offset.y / self.pixels_per_mm_y
        self.set_pan(new_pan_x_mm, new_pan_y_mm)

    def on_pan_end(self, gesture, x, y):
        pass

    def get_active_workpiece(self) -> Optional[WorkPiece]:
        active_elem = self.get_active_element()
        if active_elem and isinstance(active_elem.data, WorkPiece):
            return active_elem.data
        return None

    def get_selected_workpieces(self) -> List[WorkPiece]:
        selected_workpieces = []
        for elem in self.get_selected_elements():
            if isinstance(elem.data, WorkPiece):
                selected_workpieces.append(elem.data)
        return selected_workpieces

    def _update_selection_state(
        self, newly_selected_elements: List[CanvasElement]
    ):
        """Helper to unify selection state updates."""
        if len(newly_selected_elements) > 1:
            self._active_elem = None
            self._selection_group = MultiSelectionGroup(
                newly_selected_elements, self
            )
        elif newly_selected_elements:
            self._active_elem = newly_selected_elements[0]
            self._selection_group = None
        else:
            self._active_elem = None
            self._selection_group = None

        self.active_element_changed.send(self, element=self._active_elem)
        self.queue_draw()

    def select_workpieces(self, workpieces_to_select: List[WorkPiece]):
        """
        Clears the current selection and selects the canvas elements
        corresponding to the given list of WorkPiece objects.
        """
        self.root.unselect_all()
        uids_to_select = {wp.uid for wp in workpieces_to_select}
        newly_selected_elements = []

        # Iterate through the canvas elements to find the ones to select
        for elem in self.find_by_type(WorkPieceElement):
            if (
                elem.data
                and hasattr(elem.data, "uid")
                and elem.data.uid in uids_to_select
            ):
                elem.selected = True
                newly_selected_elements.append(elem)

        self._update_selection_state(newly_selected_elements)

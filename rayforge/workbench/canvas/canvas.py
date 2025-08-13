from __future__ import annotations
import math
from typing import Any, Generator, List, Tuple, Optional, Set, Union
import cairo
from gi.repository import Gtk, Gdk, Graphene  # type: ignore
from blinker import Signal
from .element import CanvasElement
from .region import ElementRegion
from .cursor import get_cursor_for_region
from .selection import MultiSelectionGroup
from ...core.matrix import Matrix


class Canvas(Gtk.DrawingArea):
    """
    An interactive drawing area that manages and renders `CanvasElement`
    objects.

    It handles user interactions like clicking, dragging, resizing, and
    rotating elements, as well as selection management (single, multi,
    and framing).
    """

    BASE_HANDLE_SIZE = 20.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = CanvasElement(
            0.0,
            0.0,
            0.0,  # Initial size is 0, set in do_size_allocate
            0.0,  # Initial size is 0, set in do_size_allocate
            canvas=self,
            parent=self,
        )
        self._active_elem: Optional[CanvasElement] = None

        # Stores the state of an element or group at the start of a transform
        self._active_origin: Optional[
            Union[
                Tuple[float, float, float, float],  # Group bbox (x,y,w,h)
                Tuple[float, float, float, float],  # Legacy move rect
            ]
        ] = None
        # Stores the initial transform of a single element being transformed
        self._initial_transform: Optional[Matrix] = None
        self._initial_world_transform: Optional[Matrix] = None

        self._setup_interactions()

        # --- Interaction State ---
        self._hovered_elem: Optional[CanvasElement] = None
        self._hovered_region: ElementRegion = ElementRegion.NONE
        self._active_region: ElementRegion = ElementRegion.NONE
        self._selection_group: Optional[MultiSelectionGroup] = None
        self._framing_selection: bool = False
        self._selection_frame_rect: Optional[
            Tuple[float, float, float, float]
        ] = None
        self._selection_before_framing: Set[CanvasElement] = set()
        self._group_hovered: bool = False
        self._last_mouse_x: float = 0.0
        self._last_mouse_y: float = 0.0
        self._resizing: bool = False
        self._moving: bool = False
        self._rotating: bool = False

        # --- Rotation State ---
        self._drag_start_angle: float = 0.0

        # --- Signals ---
        self.move_begin = Signal()
        self.move_end = Signal()
        self.resize_begin = Signal()
        self.resize_end = Signal()
        self.rotate_begin = Signal()
        self.rotate_end = Signal()
        self.elements_deleted = Signal()
        self.selection_changed = Signal()
        self.active_element_changed = Signal()
        self.elem_removed = Signal()

    def add(self, elem: CanvasElement):
        """Adds a top-level element to the canvas."""
        self.root.add(elem)

    def remove(self, elem: CanvasElement):
        """Removes a top-level element from the canvas."""
        self.root.remove_child(elem)

    def find_by_data(self, data: Any) -> Optional[CanvasElement]:
        """
        Finds the first element with matching data in the canvas.
        """
        return self.root.find_by_data(data)

    def find_by_type(
        self, thetype: Any
    ) -> Generator[CanvasElement, None, None]:
        """
        Finds all elements of a given type in the canvas.
        """
        return self.root.find_by_type(thetype)

    def size(self) -> Tuple[float, float]:
        """Gets the (width, height) of the canvas."""
        return self.root.size()

    def _setup_interactions(self):
        """Initializes and attaches all GTK event controllers."""
        self._click_gesture = Gtk.GestureClick()
        self._click_gesture.connect("pressed", self.on_button_press)
        self._click_gesture.connect("released", self.on_click_released)
        self.add_controller(self._click_gesture)

        self._motion_controller = Gtk.EventControllerMotion()
        self._motion_controller.connect("motion", self.on_motion)
        self._motion_controller.connect("leave", self.on_motion_leave)
        self.add_controller(self._motion_controller)

        self._drag_gesture = Gtk.GestureDrag()
        self._drag_gesture.connect("drag-update", self.on_mouse_drag)
        self._drag_gesture.connect("drag-end", self.on_button_release)
        self.add_controller(self._drag_gesture)

        self._key_controller = Gtk.EventControllerKey.new()
        self._key_controller.connect("key-pressed", self.on_key_pressed)
        self._key_controller.connect("key-released", self.on_key_released)
        self.add_controller(self._key_controller)
        self._shift_pressed: bool = False
        self._ctrl_pressed: bool = False
        self.set_focusable(True)
        self.grab_focus()

    def do_size_allocate(self, width: int, height: int, baseline: int):
        """GTK handler for when the widget's size changes."""
        self.root.set_size(float(width), float(height))
        self.root.allocate()

    def render(self, ctx: cairo.Context):
        """
        Renders the canvas content onto a given cairo context.
        This orchestrates the drawing of all elements and overlays.
        """
        # Start the recursive rendering from the root element.
        self.root.render(ctx)

        # Draw the selection frame if we are in framing mode.
        if self._framing_selection and self._selection_frame_rect:
            ctx.save()
            x, y, w, h = self._selection_frame_rect
            # A semi-transparent blue fill
            ctx.set_source_rgba(0.2, 0.5, 0.8, 0.3)
            ctx.rectangle(x, y, w, h)
            ctx.fill_preserve()
            # A solid blue, dashed border
            ctx.set_source_rgb(0.2, 0.5, 0.8)
            ctx.set_line_width(1)
            ctx.set_dash((4, 4))
            ctx.stroke()
            ctx.restore()

        # Draw selection handles on top of everything.
        self._render_selection(ctx, self.root)

    def do_snapshot(self, snapshot):
        """GTK4 snapshot-based drawing handler."""
        width, height = self.get_width(), self.get_height()
        bounds = Graphene.Rect().init(0, 0, width, height)
        ctx = snapshot.append_cairo(bounds)
        self.render(ctx)

    def _render_selection(self, ctx: cairo.Context, elem: CanvasElement):
        """
        Renders selection frames and handles by dispatching to specialized
        helpers for single or multi-selection.
        """
        is_multi_select = self._selection_group is not None

        if elem.selected and not is_multi_select:
            self._render_single_selection(ctx, elem)

        for child in elem.children:
            self._render_selection(ctx, child)

        if elem is self.root and self._selection_group:
            self._render_multi_selection(ctx, self._selection_group)

    def _render_single_selection(
        self, ctx: cairo.Context, elem: CanvasElement
    ):
        """
        Draws the selection frame for a single element by applying its
        world transformation matrix, ensuring perfect alignment.
        """
        ctx.save()

        world_transform = elem.get_world_transform()
        m = world_transform.m
        cairo_matrix = cairo.Matrix(
            m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2]
        )
        ctx.transform(cairo_matrix)

        # Calculate scale factors from the matrix to compensate line
        # widths, dash patterns, and handle sizes, so they appear
        # consistent regardless of the element's scale.
        sx = math.hypot(m[0, 0], m[1, 0])
        sy = math.hypot(m[0, 1], m[1, 1])
        avg_scale = (sx + sy) / 2.0 if sx > 1e-6 and sy > 1e-6 else 1.0

        ctx.set_line_width(1.0 / avg_scale)
        ctx.set_dash([d / avg_scale for d in (5, 5)])

        ctx.set_source_rgb(0.4, 0.4, 0.4)
        ctx.rectangle(0, 0, elem.width, elem.height)
        ctx.stroke()
        ctx.set_dash([])

        if not (self._moving or self._resizing or self._rotating):
            x, y = self._last_mouse_x, self._last_mouse_y
            is_hovered = elem.check_region_hit(x, y) != ElementRegion.NONE

            self._render_selection_handles(
                ctx,
                target=elem,
                abs_x=0,
                abs_y=0,
                is_fully_hovered=is_hovered,
                specific_hovered_region=self._hovered_region,
                # Pass both sx and sy to correctly calculate handle shapes
                # for non-uniform scaling.
                scale_compensation_factor=(sx, sy),
            )

        ctx.restore()

    def _render_multi_selection(
        self, ctx: cairo.Context, group: MultiSelectionGroup
    ):
        """
        Draws the selection frame and handles for a multi-selection group.
        """
        group._calculate_bounding_box()
        abs_x, abs_y, w, h = group.x, group.y, group.width, group.height

        ctx.save()
        # Draw the dashed selection box (group frame is not rotated).
        ctx.set_source_rgb(0.4, 0.4, 0.4)
        ctx.set_dash((5, 5))
        ctx.set_line_width(1)
        ctx.rectangle(abs_x, abs_y, w, h)
        ctx.stroke()
        ctx.set_dash([])

        # Draw handles if not currently transforming.
        if not (self._moving or self._resizing or self._rotating):
            self._render_selection_handles(
                ctx,
                target=group,
                abs_x=abs_x,
                abs_y=abs_y,
                is_fully_hovered=self._group_hovered,
                specific_hovered_region=self._hovered_region,
                scale_compensation_factor=1.0,  # No scale for group box
            )
        ctx.restore()

    def _render_selection_handles(
        self,
        ctx: cairo.Context,
        target: Union[CanvasElement, MultiSelectionGroup],
        abs_x: float,
        abs_y: float,
        is_fully_hovered: bool,
        specific_hovered_region: ElementRegion,
        scale_compensation_factor: Union[float, Tuple[float, float]],
    ):
        """
        Generic helper to draw interactive handles for a target, which can
        be a single element or a multi-selection group.
        """
        ctx.set_source_rgba(0.2, 0.5, 0.8, 0.7)

        # Draw corner and rotation handles if mouse is over the target.
        if is_fully_hovered:
            handle_regions = [
                ElementRegion.ROTATION_HANDLE,
                ElementRegion.TOP_LEFT,
                ElementRegion.TOP_RIGHT,
                ElementRegion.BOTTOM_LEFT,
                ElementRegion.BOTTOM_RIGHT,
            ]
            for region in handle_regions:
                rx, ry, rw, rh = target.get_region_rect(
                    region,
                    self.BASE_HANDLE_SIZE,
                    scale_compensation_factor,
                )
                if rw > 0 and rh > 0:
                    # Draw the line for the rotation handle.
                    if region == ElementRegion.ROTATION_HANDLE:
                        ctx.save()
                        ctx.set_source_rgba(0.4, 0.4, 0.4, 0.9)
                        # For line width, a single average scale is fine.
                        avg_scale = 1.0
                        if isinstance(scale_compensation_factor, tuple):
                            sx, sy = scale_compensation_factor
                            if sx > 1e-6 and sy > 1e-6:
                                avg_scale = (sx + sy) / 2.0
                        else:
                            avg_scale = scale_compensation_factor
                        ctx.set_line_width(1.0 / avg_scale)
                        ctx.move_to(abs_x + target.width / 2, abs_y + ry + rh)
                        ctx.line_to(abs_x + target.width / 2, abs_y)
                        ctx.stroke()
                        ctx.restore()

                    ctx.rectangle(abs_x + rx, abs_y + ry, rw, rh)
                    ctx.fill()

        # Draw edge handles only when the mouse is directly over them.
        edge_regions = [
            ElementRegion.TOP_MIDDLE,
            ElementRegion.BOTTOM_MIDDLE,
            ElementRegion.MIDDLE_LEFT,
            ElementRegion.MIDDLE_RIGHT,
        ]
        if specific_hovered_region in edge_regions:
            rx, ry, rw, rh = target.get_region_rect(
                specific_hovered_region,
                self.BASE_HANDLE_SIZE,
                scale_compensation_factor,
            )
            if rw > 0 and rh > 0:
                ctx.rectangle(abs_x + rx, abs_y + ry, rw, rh)
                ctx.fill()

    def _update_hover_state(self, x: float, y: float) -> bool:
        """
        Updates the hover state based on cursor position.

        This is the single source of truth for the interactive region. It
        checks for hits in a specific order:
        1. Resize/rotation handles on the current selection.
        2. Body of any selectable element under the cursor.

        Returns:
            True if the hover state changed and a redraw is needed.
        """
        selected_elems = self.get_selected_elements()
        is_multi_select = len(selected_elems) > 1
        new_hovered_region = ElementRegion.NONE
        new_hovered_elem = None

        # Priority 1: Check for handle hits on the current selection.
        target: Optional[Union[CanvasElement, MultiSelectionGroup]] = None
        if is_multi_select:
            target = self._selection_group
        elif selected_elems:
            target = selected_elems[0]

        if target:
            region = target.check_region_hit(x, y)
            if region not in [ElementRegion.NONE, ElementRegion.BODY]:
                new_hovered_region = region
                if isinstance(target, CanvasElement):
                    new_hovered_elem = target

        # Priority 2: If no handles were hit, find the element body.
        if new_hovered_region == ElementRegion.NONE:
            hit_elem = self.root.get_elem_hit(x, y, selectable=True)
            if hit_elem and hit_elem is not self.root:
                new_hovered_region = ElementRegion.BODY
                new_hovered_elem = hit_elem

        # Compare new state with old to see if a redraw is needed.
        needs_redraw = (
            self._hovered_region != new_hovered_region
            or self._hovered_elem is not new_hovered_elem
        )
        self._hovered_region = new_hovered_region
        self._hovered_elem = new_hovered_elem

        # Update the group hover flag.
        new_group_hovered = (
            self._selection_group is not None
            and self._selection_group.check_region_hit(x, y)
            != ElementRegion.NONE
        )
        if self._group_hovered != new_group_hovered:
            self._group_hovered = new_group_hovered
            needs_redraw = True

        return needs_redraw

    def on_button_press(self, gesture, n_press: int, x: float, y: float):
        """
        Handles the start of a click or drag operation.

        This method determines the user's intent based on what was
        clicked (element, handle, or background) and modifier keys. It
        manages selection changes and initiates move, resize, rotate, or
        framing operations.
        """
        self.grab_focus()
        self._update_hover_state(x, y)

        self._active_region = self._hovered_region
        hit = self._hovered_elem
        self._framing_selection = False
        selection_changed = False

        # --- Selection Logic ---
        if self._active_region in [ElementRegion.NONE, ElementRegion.BODY]:
            if hit is None:  # Clicked on background: start framing.
                self._framing_selection = True
                if self._shift_pressed:
                    self._selection_before_framing = set(
                        self.get_selected_elements()
                    )
                else:
                    if self.get_selected_elements():
                        selection_changed = True
                    self.root.unselect_all()
                    self._selection_before_framing = set()
                self._active_elem = None
            elif hit:  # Clicked an element.
                if not self._shift_pressed:
                    if not hit.selected:
                        self.root.unselect_all()
                        selection_changed = True
                        hit.selected = True
                else:  # Shift-click toggles selection.
                    hit.selected = not hit.selected
                    selection_changed = True
                self._active_elem = hit

        if self._framing_selection:
            self._moving, self._resizing, self._rotating = False, False, False
            if selection_changed:
                self._finalize_selection_state()
            self.queue_draw()
            return

        if selection_changed:
            self._finalize_selection_state()

        # --- Transform Logic ---
        selected_elements = self.get_selected_elements()
        target = self._selection_group or self._active_elem

        if not target:
            return

        # Start a transform action based on the active region.
        if self._active_region == ElementRegion.BODY and hit:
            self._moving = True
            self.move_begin.send(self, elements=selected_elements)
        elif self._active_region == ElementRegion.ROTATION_HANDLE:
            self._rotating = True
            self.rotate_begin.send(self, elements=selected_elements)
            self._start_rotation(target, x, y)
        elif self._active_region != ElementRegion.NONE:  # Any other handle
            self._resizing = True
            self.resize_begin.send(self, elements=selected_elements)

        # Store initial state for the transform.
        if isinstance(target, MultiSelectionGroup):
            self._active_origin = target._bounding_box
            target.store_initial_states()
        elif isinstance(target, CanvasElement):
            # For single-element transforms, store the matrices.
            self._initial_transform = target.transform.copy()
            self._initial_world_transform = target.get_world_transform().copy()
            # Also store legacy properties for move operation.
            self._active_origin = target.rect()

        self.queue_draw()

    def on_motion(self, gesture, x: float, y: float):
        """
        Handles mouse movement, updating hover state and cursor icon.
        """
        self._last_mouse_x = x
        self._last_mouse_y = y
        if not (self._moving or self._resizing or self._rotating):
            if self._update_hover_state(x, y):
                self.queue_draw()

        if self._moving:
            self.set_cursor(Gdk.Cursor.new_from_name("move"))
            return

        # Determine cursor rotation based on selection.
        cursor_angle = 0.0
        selected_elems = self.get_selected_elements()
        if self._selection_group:
            cursor_angle = self._selection_group.angle
        elif selected_elems:
            cursor_angle = selected_elems[0].get_world_angle()

        cursor = get_cursor_for_region(self._hovered_region, cursor_angle)
        self.set_cursor(cursor)

    def on_motion_leave(self, controller):
        """Resets hover state when the mouse leaves the canvas."""
        self._last_mouse_x, self._last_mouse_y = -1.0, -1.0  # Out of bounds
        if (
            self._hovered_elem is None
            and self._hovered_region == ElementRegion.NONE
        ):
            return

        self._hovered_elem = None
        self._group_hovered = False
        self._hovered_region = ElementRegion.NONE
        self.queue_draw()
        self.set_cursor(Gdk.Cursor.new_from_name("default"))

    def _move_active_element(self, offset_x: float, offset_y: float):
        """Moves a single selected element using a matrix-native approach."""
        if not self._active_elem or not self._initial_world_transform:
            return

        # Create a delta translation matrix in world space
        delta_translation = Matrix.translation(offset_x, offset_y)

        # Apply this delta to the initial world transform
        new_world_transform = delta_translation @ self._initial_world_transform

        # Convert back to the element's local space
        parent_inv_world = Matrix.identity()
        if isinstance(self._active_elem.parent, CanvasElement):
            parent_inv_world = (
                self._active_elem.parent.get_world_transform().invert()
            )

        new_local_transform = parent_inv_world @ new_world_transform

        # Set the final transform, preserving all components
        self._active_elem.set_transform(new_local_transform)

    def on_mouse_drag(self, gesture, offset_x: float, offset_y: float):
        """
        Handles an active drag, dispatching to transform-specific methods.
        """
        if self._framing_selection:
            ok, start_x, start_y = self._drag_gesture.get_start_point()
            if not ok:
                return
            x1, y1 = start_x, start_y
            x2, y2 = start_x + offset_x, start_y + offset_y
            self._selection_frame_rect = (
                min(x1, x2),
                min(y1, y2),
                abs(x1 - x2),
                abs(y1 - y2),
            )
            self._update_framing_selection()  # Update selection live
            self.queue_draw()
            return

        if self._selection_group:
            if self._moving:
                self._selection_group.apply_move(offset_x, offset_y)
            elif self._resizing:
                self._apply_group_resize(offset_x, offset_y)
            elif self._rotating:
                self._rotate_selection_group(offset_x, offset_y)
            self.queue_draw()
        elif self._active_elem:
            if self._moving:
                self._move_active_element(offset_x, offset_y)
            elif self._resizing:
                self._resize_active_element(offset_x, offset_y)
            elif self._rotating:
                ok, start_x, start_y = self._drag_gesture.get_start_point()
                if not ok:
                    return
                current_x = start_x + offset_x
                current_y = start_y + offset_y
                self._rotate_active_element(current_x, current_y)
            self.queue_draw()

    def _apply_group_resize(self, offset_x: float, offset_y: float):
        """Calculates new group bounding box based on the drag offset."""
        if (
            not self._selection_group
            or not isinstance(self._active_origin, tuple)
            or len(self._active_origin) != 4
        ):
            return

        orig_x, orig_y, orig_w, orig_h = self._active_origin
        min_size = 20.0

        # Determine which handles are being dragged.
        is_left = self._active_region in {
            ElementRegion.TOP_LEFT,
            ElementRegion.MIDDLE_LEFT,
            ElementRegion.BOTTOM_LEFT,
        }
        is_right = self._active_region in {
            ElementRegion.TOP_RIGHT,
            ElementRegion.MIDDLE_RIGHT,
            ElementRegion.BOTTOM_RIGHT,
        }
        is_top = self._active_region in {
            ElementRegion.TOP_LEFT,
            ElementRegion.TOP_MIDDLE,
            ElementRegion.TOP_RIGHT,
        }
        is_bottom = self._active_region in {
            ElementRegion.BOTTOM_LEFT,
            ElementRegion.BOTTOM_MIDDLE,
            ElementRegion.BOTTOM_RIGHT,
        }

        if self._ctrl_pressed:  # Center-out resize
            dw, dh = 0.0, 0.0
            if is_left:
                dw = -offset_x
            elif is_right:
                dw = offset_x
            if is_top:
                dh = -offset_y
            elif is_bottom:
                dh = offset_y

            dw *= 2
            dh *= 2

            if self._shift_pressed and orig_w > 0 and orig_h > 0:
                aspect = orig_w / orig_h
                if abs(offset_x) > abs(offset_y):
                    dh = dw / aspect
                else:
                    dw = dh * aspect

            new_w, new_h = orig_w + dw, orig_h + dh
            new_x = orig_x - (new_w - orig_w) / 2
            new_y = orig_y - (new_h - orig_h) / 2
        else:  # Anchor-based resize
            new_x, new_y, new_w, new_h = orig_x, orig_y, orig_w, orig_h
            if is_left:
                new_x, new_w = orig_x + offset_x, orig_w - offset_x
            elif is_right:
                new_w = orig_w + offset_x
            if is_top:
                new_y, new_h = orig_y + offset_y, orig_h - offset_y
            elif is_bottom:
                new_h = orig_h + offset_y

            if self._shift_pressed and orig_w > 0 and orig_h > 0:
                aspect = orig_w / orig_h
                dw, dh = new_w - orig_w, new_h - orig_h
                is_corner = (is_left or is_right) and (is_top or is_bottom)

                if (is_corner and abs(dw) > abs(dh) * aspect) or (
                    not is_corner and (is_left or is_right)
                ):
                    new_h = new_w / aspect
                else:
                    new_w = new_h * aspect

                if is_left:
                    new_x = orig_x + orig_w - new_w
                if is_top:
                    new_y = orig_y + orig_h - new_h

        new_w, new_h = max(new_w, min_size), max(new_h, min_size)
        new_box = (new_x, new_y, new_w, new_h)
        self._selection_group.apply_resize(new_box, self._active_origin)

    def _start_rotation(
        self,
        target: Union[CanvasElement, MultiSelectionGroup],
        x: float,
        y: float,
    ):
        """Stores the initial state for a rotation operation."""
        is_group = isinstance(target, MultiSelectionGroup)

        center_x, center_y = (
            target.center if is_group else target.get_world_center()
        )
        self._drag_start_angle = math.degrees(
            math.atan2(y - center_y, x - center_x)
        )

    def _rotate_active_element(self, current_x: float, current_y: float):
        """
        Rotates a single element using a matrix-native approach to preserve
        all existing transformations, including shear.
        """
        if not self._active_elem or not self._initial_world_transform:
            return

        # 1. Use the initial world transform to find the stable center point.
        elem_center_world = self._initial_world_transform.transform_point(
            (self._active_elem.width / 2, self._active_elem.height / 2)
        )

        # 2. Calculate the angle of the current mouse position.
        current_angle = math.degrees(
            math.atan2(
                current_y - elem_center_world[1],
                current_x - elem_center_world[0],
            )
        )

        # 3. Find the change in angle since the drag started.
        angle_diff = current_angle - self._drag_start_angle

        # 4. Create a delta rotation matrix around the world-space center.
        delta_rotation = Matrix.rotation(angle_diff, center=elem_center_world)

        # 5. Apply this delta to the element's initial world state.
        new_world_transform = delta_rotation @ self._initial_world_transform

        # 6. Convert the new world transform back to a local transform.
        parent_inv_world = Matrix.identity()
        if isinstance(self._active_elem.parent, CanvasElement):
            parent_inv_world = (
                self._active_elem.parent.get_world_transform().invert()
            )
        new_local_transform = parent_inv_world @ new_world_transform

        # 7. Set the new matrix directly. This preserves shear and
        # avoids jumps.
        self._active_elem.set_transform(new_local_transform)

    def _rotate_selection_group(self, offset_x: float, offset_y: float):
        """Rotates the entire selection group based on cursor drag."""
        if not self._selection_group:
            return
        ok, start_x, start_y = self._drag_gesture.get_start_point()
        if not ok:
            return

        current_x, current_y = start_x + offset_x, start_y + offset_y
        center_x, center_y = self._selection_group.initial_center
        current_angle = math.degrees(
            math.atan2(current_y - center_y, current_x - center_x)
        )
        angle_diff = current_angle - self._drag_start_angle
        self._selection_group.apply_rotate(angle_diff)

    def _resize_active_element(self, offset_x: float, offset_y: float):
        """
        Resizes a single element using a fully matrix-native approach that
        prevents unwanted shear by scaling in the element's local space.
        """
        if (
            not self._active_elem
            or not self._initial_transform
            or not self._initial_world_transform
        ):
            return

        min_size = 20.0
        base_w, base_h = self._active_elem.width, self._active_elem.height

        # 1. Determine which handles are being dragged.
        is_left = self._active_region in {
            ElementRegion.TOP_LEFT,
            ElementRegion.MIDDLE_LEFT,
            ElementRegion.BOTTOM_LEFT,
        }
        is_right = self._active_region in {
            ElementRegion.TOP_RIGHT,
            ElementRegion.MIDDLE_RIGHT,
            ElementRegion.BOTTOM_RIGHT,
        }
        is_top = self._active_region in {
            ElementRegion.TOP_LEFT,
            ElementRegion.TOP_MIDDLE,
            ElementRegion.TOP_RIGHT,
        }
        is_bottom = self._active_region in {
            ElementRegion.BOTTOM_LEFT,
            ElementRegion.BOTTOM_MIDDLE,
            ElementRegion.BOTTOM_RIGHT,
        }

        # 2. Transform the screen-space drag offset into the element's
        #    initial local coordinate system (unrotated frame of reference).
        initial_world_no_trans = self._initial_world_transform.copy()
        initial_world_no_trans.m[0, 2] = initial_world_no_trans.m[1, 2] = 0.0
        inv_rot_scale = initial_world_no_trans.invert()
        local_delta_x, local_delta_y = inv_rot_scale.transform_vector(
            (offset_x, offset_y)
        )

        # 3. Calculate desired change in size (dw, dh) in local space.
        dw, dh = 0.0, 0.0
        if is_left:
            dw = -local_delta_x
        elif is_right:
            dw = local_delta_x
        if is_top:
            dh = -local_delta_y
        elif is_bottom:
            dh = local_delta_y

        if self._ctrl_pressed:
            dw *= 2.0
            dh *= 2.0

        # 4. Handle aspect ratio constraint (Shift key).
        if self._shift_pressed and base_w > 0 and base_h > 0:
            aspect = base_w / base_h
            is_corner = (is_left or is_right) and (is_top or is_bottom)
            if is_corner:
                if abs(dw) * aspect > abs(dh):
                    dh = dw / aspect
                else:
                    dw = dh * aspect
            elif is_left or is_right:
                dh = dw / aspect
            elif is_top or is_bottom:
                dw = dh * aspect

        # 5. Calculate new size and the scale factors to apply.
        new_w = max(min_size, base_w + dw)
        new_h = max(min_size, base_h + dh)
        scale_x = new_w / base_w if base_w > 0 else 1.0
        scale_y = new_h / base_h if base_h > 0 else 1.0

        # 6. Define the fixed anchor point in the element's LOCAL GEOMETRY.
        anchor_norm_x = (
            0.5 if not (is_left or is_right) else (1.0 if is_left else 0.0)
        )
        anchor_norm_y = (
            0.5 if not (is_top or is_bottom) else (1.0 if is_top else 0.0)
        )
        if self._ctrl_pressed:
            anchor_norm_x, anchor_norm_y = 0.5, 0.5

        anchor_local_geom = (anchor_norm_x * base_w, anchor_norm_y * base_h)

        # 7. Construct the delta transform: a scale around the LOCAL anchor.
        t_to_origin = Matrix.translation(
            -anchor_local_geom[0], -anchor_local_geom[1]
        )
        m_scale = Matrix.scale(scale_x, scale_y)
        t_from_origin = Matrix.translation(
            anchor_local_geom[0], anchor_local_geom[1]
        )
        delta_transform_local = t_from_origin @ m_scale @ t_to_origin

        # 8. Apply this local delta to the element's initial local transform.
        new_local_transform = self._initial_transform @ delta_transform_local

        # 9. Apply the new, complete transform matrix to the element.
        self._active_elem.set_transform(new_local_transform)

    def on_button_release(self, gesture, x: float, y: float):
        """Handles the end of a drag operation, finalizing transforms."""
        if self._framing_selection:
            self._framing_selection = False
            self._selection_frame_rect = None
            self._selection_before_framing.clear()
            self._finalize_selection_state()
            return

        if not (self._moving or self._resizing or self._rotating):
            return

        elements = self.get_selected_elements()
        if self._moving:
            self.move_end.send(self, elements=elements)
        elif self._resizing:
            self.resize_end.send(self, elements=elements)
            for elem in elements:
                elem.trigger_update()
        elif self._rotating:
            self.rotate_end.send(self, elements=elements)

        if self._selection_group:
            self._selection_group._calculate_bounding_box()
            self._active_origin = self._selection_group._bounding_box

        self._resizing, self._moving, self._rotating = False, False, False
        self._active_region = ElementRegion.NONE
        self._initial_transform = None
        self._initial_world_transform = None

    def on_click_released(self, gesture, n_press: int, x: float, y: float):
        """
        Handles the completion of a click that did not become a drag.
        """
        if self._framing_selection:
            self._framing_selection = False
            self._selection_frame_rect = None
            self._selection_before_framing.clear()
            self._finalize_selection_state()

    def _finalize_selection_state(self):
        """
        Updates selection state after an operation.

        This centralizes the logic for updating the active element, the
        multi-selection group, and firing necessary signals.
        """
        selected = self.get_selected_elements()

        if self._active_elem and self._active_elem not in selected:
            self._active_elem = None
        if not self._active_elem and selected:
            self._active_elem = selected[-1]

        if len(selected) > 1:
            if not self._selection_group or set(
                self._selection_group.elements
            ) != set(selected):
                self._selection_group = MultiSelectionGroup(selected, self)
        else:
            self._selection_group = None

        self.active_element_changed.send(self, element=self._active_elem)
        self.selection_changed.send(
            self, elements=selected, active_element=self._active_elem
        )
        self.queue_draw()

    def _get_element_world_bbox(self, elem: CanvasElement) -> Graphene.Rect:
        """
        Calculates the axis-aligned bounding box of an element in world
        coordinates, accounting for all transformations.
        """
        world_transform = elem.get_world_transform()
        w, h = elem.width, elem.height

        # Transform all four local corners to world space.
        local_corners = [(0, 0), (w, 0), (w, h), (0, h)]
        world_corners = [
            world_transform.transform_point(p) for p in local_corners
        ]

        # Find the min/max of the transformed corner coordinates.
        min_x = min(p[0] for p in world_corners)
        min_y = min(p[1] for p in world_corners)
        max_x = max(p[0] for p in world_corners)
        max_y = max(p[1] for p in world_corners)

        return Graphene.Rect().init(min_x, min_y, max_x - min_x, max_y - min_y)

    def _update_framing_selection(self):
        """
        Updates element selection based on the rubber-band frame.
        """
        if not self._selection_frame_rect:
            return

        frame_x, frame_y, frame_w, frame_h = self._selection_frame_rect

        # Avoid selection changes from a simple click (zero-area frame).
        if frame_w < 2 and frame_h < 2:
            return

        selection_rect = Graphene.Rect().init(
            frame_x, frame_y, frame_w, frame_h
        )
        selection_changed = False

        for elem in self.root.get_all_children_recursive():
            if elem.selectable:
                elem_bbox = self._get_element_world_bbox(elem)
                intersects = selection_rect.intersection(elem_bbox)[0]

                # Select if it intersects or was part of the initial set
                # in shift-mode.
                newly_selected = (
                    elem in self._selection_before_framing
                ) or intersects
                if elem.selected != newly_selected:
                    elem.selected = newly_selected
                    selection_changed = True

        if selection_changed:
            self._finalize_selection_state()

    def on_key_pressed(
        self, controller, keyval: int, keycode: int, state: Gdk.ModifierType
    ) -> bool:
        """Handles key press events for modifiers and actions."""
        if keyval in (Gdk.KEY_Shift_L, Gdk.KEY_Shift_R):
            self._shift_pressed = True
            return True
        elif keyval in (Gdk.KEY_Control_L, Gdk.KEY_Control_R):
            self._ctrl_pressed = True
            return True
        elif keyval == Gdk.KEY_Delete:
            selected_elements = list(self.root.get_selected())
            if selected_elements:
                self.elements_deleted.send(self, elements=selected_elements)
                self.root.remove_selected()
                self._finalize_selection_state()
            return True
        return False

    def on_key_released(
        self, controller, keyval: int, keycode: int, state: Gdk.ModifierType
    ):
        """Handles key release events for modifiers."""
        if keyval in (Gdk.KEY_Shift_L, Gdk.KEY_Shift_R):
            self._shift_pressed = False
        elif keyval in (Gdk.KEY_Control_L, Gdk.KEY_Control_R):
            self._ctrl_pressed = False

    def get_active_element(self) -> Optional[CanvasElement]:
        """Returns the currently active element, if any."""
        return self._active_elem

    def get_selected_elements(self) -> List[CanvasElement]:
        """Returns a list of all currently selected elements."""
        return list(self.root.get_selected())

    def unselect_all(self):
        """Deselects all elements on the canvas."""
        # Do nothing if there's no selection to clear, to avoid
        # unnecessary state changes and signal emissions.
        if not self.get_selected_elements():
            return

        self.root.unselect_all()
        self._finalize_selection_state()

    def dump(self):
        """Prints a representation of the entire element hierarchy."""
        self.root.dump()

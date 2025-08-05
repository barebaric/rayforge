from __future__ import annotations
import logging
import math
from typing import Any, Generator, List, Tuple, Optional, Set
import cairo
from gi.repository import Gtk, Gdk, Graphene  # type: ignore
from blinker import Signal
from .element import CanvasElement
from .region import ElementRegion
from .cursor import get_cursor_for_region
from .selection import MultiSelectionGroup


class Canvas(Gtk.DrawingArea):
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
        self._active_origin: Optional[Tuple[float, float, float, float]] = None
        self._setup_interactions()

        # Interaction state
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

        # Rotation state
        self._original_elem_angle: float = 0.0
        self._drag_start_angle: float = 0.0

        # Signals
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
        self.root.add(elem)

    def remove(self, elem: CanvasElement):
        # The root element's remove method handles removing from its children
        self.root.remove_child(elem)

    def find_by_data(self, data: Any) -> Optional[CanvasElement]:
        """
        Returns the CanvasElement with the given data, or None if none
        was found.
        """
        return self.root.find_by_data(data)

    def find_by_type(
        self, thetype: Any
    ) -> Generator[CanvasElement, None, None]:
        """
        Returns the CanvasElements with the given type.
        """
        return self.root.find_by_type(thetype)

    def size(self) -> Tuple[float, float]:
        return self.root.size()

    def _setup_interactions(self):
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
        self._resizing: bool = False
        self._moving: bool = False
        self._rotating: bool = False

        self._key_controller = Gtk.EventControllerKey.new()
        self._key_controller.connect("key-pressed", self.on_key_pressed)
        self._key_controller.connect("key-released", self.on_key_released)
        self.add_controller(self._key_controller)
        self._shift_pressed: bool = False
        self._ctrl_pressed: bool = False
        self.set_focusable(True)
        self.grab_focus()

    def do_size_allocate(self, width: int, height: int, baseline: int):
        self.root.set_size(float(width), float(height))
        self.root.allocate()

    def render(self, ctx: cairo.Context):
        """
        Renders the canvas content onto a given cairo context.
        This is the main drawing logic, separated for extensibility.
        """
        # Start the recursive rendering process from the root element.
        self.root.render(ctx)

        # Draw the selection frame if we are in framing mode
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
        width, height = self.get_width(), self.get_height()
        bounds = Graphene.Rect().init(0, 0, width, height)
        ctx = snapshot.append_cairo(bounds)
        self.render(ctx)

    def _render_selection(self, ctx: cairo.Context, elem: CanvasElement):
        """
        Renders selection frames and handles by dispatching to specialized
        helpers.
        """
        is_multi_select = self._selection_group is not None

        # 1. Draw selection for a single, selected element
        if elem.selected and not is_multi_select:
            self._render_single_selection(ctx, elem)

        # 2. Recurse for all children
        for child in elem.children:
            self._render_selection(ctx, child)

        # 3. Draw the multi-selection group frame (only at the root level)
        if elem is self.root and self._selection_group:
            self._render_multi_selection(ctx, self._selection_group)

    def _render_single_selection(
        self, ctx: cairo.Context, elem: CanvasElement
    ):
        """Draws the selection frame and handles for a single element."""
        ctx.save()

        abs_x, abs_y = elem.pos_abs()
        angle = elem.get_angle()

        # Apply rotation to the context
        if angle != 0:
            ctx.translate(abs_x + elem.width / 2, abs_y + elem.height / 2)
            ctx.rotate(math.radians(angle))
            ctx.translate(
                -(abs_x + elem.width / 2), -(abs_y + elem.height / 2)
            )

        # Draw the dashed selection box
        ctx.set_source_rgb(0.4, 0.4, 0.4)
        ctx.set_dash((5, 5))
        ctx.set_line_width(1)
        ctx.rectangle(abs_x, abs_y, elem.width, elem.height)
        ctx.stroke()
        ctx.set_dash([])

        # Draw handles if not currently transforming
        if not (self._moving or self._resizing or self._rotating):
            x, y = self._last_mouse_x, self._last_mouse_y
            is_hovered = elem.check_region_hit(x, y) != ElementRegion.NONE
            self._render_selection_handles(
                ctx,
                target=elem,
                abs_x=abs_x,
                abs_y=abs_y,
                is_fully_hovered=is_hovered,
                specific_hovered_region=self._hovered_region,
            )
        ctx.restore()

    def _render_multi_selection(
        self, ctx: cairo.Context, group: MultiSelectionGroup
    ):
        """
        Draws the selection frame and handles for a multi-selection group.
        """
        # Ensure the bounding box is up-to-date
        group._calculate_bounding_box()
        abs_x, abs_y, w, h = group.x, group.y, group.width, group.height

        ctx.save()
        # Draw the dashed selection box (no rotation for the group frame)
        ctx.set_source_rgb(0.4, 0.4, 0.4)
        ctx.set_dash((5, 5))
        ctx.set_line_width(1)
        ctx.rectangle(abs_x, abs_y, w, h)
        ctx.stroke()
        ctx.set_dash([])

        # Draw handles if not currently transforming
        if not (self._moving or self._resizing or self._rotating):
            self._render_selection_handles(
                ctx,
                target=group,
                abs_x=abs_x,
                abs_y=abs_y,
                is_fully_hovered=self._group_hovered,
                specific_hovered_region=self._hovered_region,
            )
        ctx.restore()

    def _render_selection_handles(
        self,
        ctx: cairo.Context,
        target: CanvasElement | MultiSelectionGroup,
        abs_x: float,
        abs_y: float,
        is_fully_hovered: bool,
        specific_hovered_region: ElementRegion,
    ):
        """
        A generic helper to draw interactive handles for a target
        (either a CanvasElement or a MultiSelectionGroup).
        """
        ctx.set_source_rgba(0.2, 0.5, 0.8, 0.7)

        # Draw corner and rotation handles if the mouse is over the
        # element/group
        if is_fully_hovered:
            handle_regions = [
                ElementRegion.ROTATION_HANDLE,
                ElementRegion.TOP_LEFT,
                ElementRegion.TOP_RIGHT,
                ElementRegion.BOTTOM_LEFT,
                ElementRegion.BOTTOM_RIGHT,
            ]
            for region in handle_regions:
                rx, ry, rw, rh = target.get_region_rect(region)
                if rw > 0 and rh > 0:
                    # Special case: draw the line for the rotation handle
                    if region == ElementRegion.ROTATION_HANDLE:
                        ctx.save()
                        ctx.set_source_rgba(0.4, 0.4, 0.4, 0.9)
                        ctx.set_line_width(1)
                        ctx.move_to(abs_x + target.width / 2, abs_y + ry + rh)
                        ctx.line_to(abs_x + target.width / 2, abs_y)
                        ctx.stroke()
                        ctx.restore()

                    ctx.rectangle(abs_x + rx, abs_y + ry, rw, rh)
                    ctx.fill()

        # Draw edge handles only when the mouse is directly over them
        edge_regions = [
            ElementRegion.TOP_MIDDLE,
            ElementRegion.BOTTOM_MIDDLE,
            ElementRegion.MIDDLE_LEFT,
            ElementRegion.MIDDLE_RIGHT,
        ]
        if specific_hovered_region in edge_regions:
            rx, ry, rw, rh = target.get_region_rect(specific_hovered_region)
            if rw > 0 and rh > 0:
                ctx.rectangle(abs_x + rx, abs_y + ry, rw, rh)
                ctx.fill()

    def _update_hover_state(self, x: float, y: float) -> bool:
        """
        Updates hover state and returns True if a redraw is needed.
        This is the single source of truth for the interactive region.
        """
        selected_elems = self.get_selected_elements()
        is_multi_select = len(selected_elems) > 1

        # Use temporary variables to determine the new state.
        new_hovered_region = ElementRegion.NONE
        new_hovered_elem = None

        # Priority 1: Check for handle hits on the current selection.
        if is_multi_select:
            if self._selection_group:
                region = self._selection_group.check_region_hit(x, y)
                if region not in [ElementRegion.NONE, ElementRegion.BODY]:
                    new_hovered_region = region
                    # For multi-select, there's no single hovered element.
        elif selected_elems:
            # Check for handles on the single selected element.
            sel_elem = selected_elems[0]
            region = sel_elem.check_region_hit(x, y)
            if region not in [ElementRegion.NONE, ElementRegion.BODY]:
                new_hovered_region = region
                new_hovered_elem = sel_elem  # We hit a handle of this element.

        # Priority 2: If no handles were hit, find which element body is under
        # the cursor.
        if new_hovered_region == ElementRegion.NONE:
            hit_elem = self.root.get_elem_hit(
                x - self.root.x, y - self.root.y, selectable=True
            )
            if hit_elem and hit_elem is not self.root:
                new_hovered_region = ElementRegion.BODY
                new_hovered_elem = hit_elem

        # Now, compare the new state with the old and determine if a redraw is
        # needed.
        needs_redraw = False
        if self._hovered_region != new_hovered_region:
            self._hovered_region = new_hovered_region
            needs_redraw = True

        if self._hovered_elem is not new_hovered_elem:
            self._hovered_elem = new_hovered_elem
            needs_redraw = True

        # Finally, update the group hover flag.
        new_group_hovered = False
        if self._selection_group:
            if (
                self._selection_group.check_region_hit(x, y)
                != ElementRegion.NONE
            ):
                new_group_hovered = True

        if self._group_hovered != new_group_hovered:
            self._group_hovered = new_group_hovered
            needs_redraw = True

        return needs_redraw

    def on_button_press(self, gesture, n_press: int, x: float, y: float):
        self.grab_focus()
        self._update_hover_state(x, y)

        self._active_region = self._hovered_region
        hit = self._hovered_elem
        self._framing_selection = False
        selection_changed = False

        # Logic for selection change
        if self._active_region in [ElementRegion.NONE, ElementRegion.BODY]:
            if hit is None:  # Clicked on background, prepare for framing
                self._framing_selection = True
                if self._shift_pressed:
                    # Store existing selection for additive mode
                    self._selection_before_framing = set(
                        self.get_selected_elements()
                    )
                else:
                    # Clear selection for standard framing
                    if self.get_selected_elements():
                        selection_changed = True
                    self.root.unselect_all()
                    self._selection_before_framing = set()
                self._active_elem = None  # No active element when framing
            elif hit:  # Clicked an element
                if not self._shift_pressed:
                    # If not already the sole selected item, change selection.
                    if not hit.selected:
                        self.root.unselect_all()
                        selection_changed = True
                        hit.selected = True
                else:  # Shift-click
                    hit.selected = not hit.selected
                    selection_changed = True
                # The clicked item is the new primary/active item.
                self._active_elem = hit

        # If framing, we don't start a transform.
        if self._framing_selection:
            self._moving, self._resizing, self._rotating = False, False, False
            self._active_origin = None
            if selection_changed:
                self._finalize_selection_state()
            self.queue_draw()
            return

        # Update selection state and start transform if needed
        if selection_changed:
            self._finalize_selection_state()

        selected_elements = self.get_selected_elements()

        # Logic for starting a transform action
        if self._active_region == ElementRegion.BODY and hit:
            self._moving, self._resizing, self._rotating = (True, False, False)
            self.move_begin.send(self, elements=selected_elements)
        elif self._active_region == ElementRegion.ROTATION_HANDLE:
            self._moving, self._resizing, self._rotating = False, False, True
            self.rotate_begin.send(self, elements=selected_elements)
            target = self._selection_group or self._active_elem
            if target:
                self._start_rotation(target, x, y)
        elif self._active_region not in [
            ElementRegion.NONE,
            ElementRegion.BODY,
        ]:
            self._moving, self._resizing, self._rotating = False, True, False
            self.resize_begin.send(self, elements=selected_elements)

        if self._selection_group:
            self._active_origin = self._selection_group._bounding_box
            self._selection_group.store_initial_states()
        elif self._active_elem:
            self._active_origin = self._active_elem.rect()

        self.queue_draw()

    def on_motion(self, gesture, x: float, y: float):
        self._last_mouse_x = x
        self._last_mouse_y = y
        if not (self._moving or self._resizing or self._rotating):
            if self._update_hover_state(x, y):
                self.queue_draw()

        if self._moving:
            self.set_cursor(Gdk.Cursor.new_from_name("move"))
            return

        cursor_angle = 0.0
        selected_elems = self.get_selected_elements()
        if self._selection_group:
            cursor_angle = self._selection_group.angle
        elif selected_elems:
            # Use the single selected element for cursor angle
            cursor_angle = selected_elems[0].get_angle()

        cursor = get_cursor_for_region(self._hovered_region, cursor_angle)
        self.set_cursor(cursor)

    def on_motion_leave(self, controller):
        """
        Called when the pointer leaves the canvas. Resets hover state to
        prevent sticky hover effects.
        """
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
        cursor = Gdk.Cursor.new_from_name("default")
        self.set_cursor(cursor)

    def on_mouse_drag(self, gesture, offset_x: float, offset_y: float):
        if self._framing_selection:
            ok, start_x, start_y = self._drag_gesture.get_start_point()
            if not ok:
                return

            x1, y1 = start_x, start_y
            x2, y2 = start_x + offset_x, start_y + offset_y

            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x1 - x2)
            h = abs(y1 - y2)

            self._selection_frame_rect = (x, y, w, h)
            self._update_framing_selection()  # Update selection live
            self.queue_draw()
            return

        if not self._active_origin:
            return

        if self._selection_group:
            if self._moving:
                self._selection_group.apply_move(offset_x, offset_y)
                self.queue_draw()
            elif self._resizing:
                self._apply_group_resize(offset_x, offset_y)
                self.queue_draw()
            elif self._rotating:
                self._rotate_selection_group(offset_x, offset_y)
                self.queue_draw()
        elif self._active_elem:
            if self._moving:
                elem_start_x, elem_start_y, _, _ = self._active_origin
                self._active_elem.set_pos(
                    elem_start_x + offset_x, elem_start_y + offset_y
                )
                self.queue_draw()
            elif self._resizing:
                self._resize_active_element(offset_x, offset_y)
            elif self._rotating:
                ok, start_x, start_y = self._drag_gesture.get_start_point()
                if not ok:
                    return
                current_x, current_y = start_x + offset_x, start_y + offset_y
                self._rotate_active_element(current_x, current_y)

    def _apply_group_resize(self, offset_x: float, offset_y: float):
        """Calculates new group bbox based on the drag offset."""
        if not self._selection_group or not self._active_origin:
            return

        orig_x, orig_y, orig_w, orig_h = self._active_origin
        min_size = 20.0

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

        if self._ctrl_pressed:
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
                is_corner = (is_left or is_right) and (is_top or is_bottom)
                if is_corner and abs(offset_x) > abs(offset_y):
                    dh = dw / aspect
                elif is_corner:
                    dw = dh * aspect
                elif is_left or is_right:
                    dh = dw / aspect
                else:
                    dw = dh * aspect

            new_w, new_h = orig_w + dw, orig_h + dh
            new_x = orig_x - (new_w - orig_w) / 2
            new_y = orig_y - (new_h - orig_h) / 2
        else:  # Default anchor-based resize
            new_x, new_y, new_w, new_h = orig_x, orig_y, orig_w, orig_h
            if is_left:
                new_x = orig_x + offset_x
                new_w = orig_w - offset_x
            elif is_right:
                new_w = orig_w + offset_x

            if is_top:
                new_y = orig_y + offset_y
                new_h = orig_h - offset_y
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
        self, target: CanvasElement | MultiSelectionGroup, x: float, y: float
    ):
        """Stores initial state for a rotation operation."""
        self._original_elem_angle = (
            target.angle
            if isinstance(target, MultiSelectionGroup)
            else target.get_angle()
        )
        if isinstance(target, MultiSelectionGroup):
            center_x, center_y = target.center
        else:
            abs_x, abs_y = target.pos_abs()
            center_x = abs_x + target.width / 2
            center_y = abs_y + target.height / 2
        self._drag_start_angle = math.degrees(
            math.atan2(y - center_y, x - center_x)
        )

    def _rotate_active_element(self, current_x: float, current_y: float):
        """Handles the logic for rotating an element based on drag delta."""
        if not self._active_elem:
            return

        abs_x, abs_y = self._active_elem.pos_abs()
        center_x = abs_x + self._active_elem.width / 2
        center_y = abs_y + self._active_elem.height / 2

        current_angle = math.degrees(
            math.atan2(current_y - center_y, current_x - center_x)
        )

        angle_diff = current_angle - self._drag_start_angle
        new_angle = self._original_elem_angle + angle_diff

        self._active_elem.set_angle(new_angle)
        self.queue_draw()

    def _rotate_selection_group(self, offset_x: float, offset_y: float):
        """Handles logic for rotating the entire selection group."""
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
        self.queue_draw()

    def _resize_active_element(self, offset_x: float, offset_y: float):
        """
        Handles the logic for resizing a (potentially rotated) element,
        supporting aspect ratio lock (Shift) and resize-from-center (Ctrl).
        """
        if not self._active_elem or not self._active_origin:
            return

        start_x, start_y, start_w, start_h = self._active_origin
        min_size = 20.0
        angle_deg = self._active_elem.get_angle()

        # 1. Transform drag delta into the element's local coordinate system
        angle_rad = math.radians(-angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        local_delta_x = offset_x * cos_a - offset_y * sin_a
        local_delta_y = offset_x * sin_a + offset_y * cos_a

        # 2. Determine which edges/corners are being dragged
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

        # 3. Calculate initial change in width/height (dw, dh)
        dw, dh = 0.0, 0.0
        if is_left:
            dw = -local_delta_x
        elif is_right:
            dw = local_delta_x
        if is_top:
            dh = -local_delta_y
        elif is_bottom:
            dh = local_delta_y

        # If Ctrl is pressed, resize from the center by doubling the change
        if self._ctrl_pressed:
            dw *= 2.0
            dh *= 2.0

        # 4. Handle aspect ratio constraint if Shift is pressed
        if self._shift_pressed and start_w > 0 and start_h > 0:
            aspect = start_w / start_h
            is_corner = (is_left or is_right) and (is_top or is_bottom)

            if is_corner:
                # For corners, use the larger delta's axis to drive the resize
                if abs(local_delta_x) > abs(local_delta_y):
                    dh = dw / aspect
                else:
                    dw = dh * aspect
            elif is_left or is_right:  # Horizontal edge drag
                dh = dw / aspect
            elif is_top or is_bottom:  # Vertical edge drag
                dw = dh * aspect

        # 5. Calculate new size, enforce minimums, and re-check aspect ratio
        new_w, new_h = float(start_w) + dw, float(start_h) + dh

        clamped_w, clamped_h = max(new_w, min_size), max(new_h, min_size)
        if self._shift_pressed and start_w > 0 and start_h > 0:
            aspect = start_w / start_h
            if clamped_w != new_w:  # Width was clamped
                clamped_h = clamped_w / aspect
            if clamped_h != new_h:  # Height was clamped (takes precedence)
                clamped_w = clamped_h * aspect
        new_w, new_h = clamped_w, clamped_h

        # 6. Calculate final change in size and how the center shifts
        dw = new_w - start_w
        dh = new_h - start_h
        center_dx_local, center_dy_local = 0.0, 0.0

        # If Ctrl is NOT pressed, shift center to keep opposite side anchored.
        # If Ctrl IS pressed, center does not shift (remains 0).
        if not self._ctrl_pressed:
            if is_left:
                center_dx_local = -dw / 2
            elif is_right:
                center_dx_local = dw / 2
            if is_top:
                center_dy_local = -dh / 2
            elif is_bottom:
                center_dy_local = dh / 2

        # 7. Transform the center shift back to the canvas coordinate system
        angle_rad_fwd = math.radians(angle_deg)
        cos_a_fwd, sin_a_fwd = math.cos(angle_rad_fwd), math.sin(angle_rad_fwd)
        center_dx_canvas = (
            center_dx_local * cos_a_fwd - center_dy_local * sin_a_fwd
        )
        center_dy_canvas = (
            center_dx_local * sin_a_fwd + center_dy_local * cos_a_fwd
        )

        # 8. Calculate new top-left position based on the (shifted) center
        old_center_x = start_x + start_w / 2
        old_center_y = start_y + start_h / 2
        new_center_x = old_center_x + center_dx_canvas
        new_center_y = old_center_y + center_dy_canvas
        new_x = new_center_x - new_w / 2
        new_y = new_center_y - new_h / 2

        # 9. Apply changes
        self._active_elem.set_pos(new_x, new_y)
        self._active_elem.set_size(new_w, new_h)

    def on_button_release(self, gesture, x: float, y: float):
        if self._framing_selection:
            # The selection is already live, so we just clean up and finalize.
            self._framing_selection = False
            self._selection_frame_rect = None
            self._selection_before_framing.clear()
            self._finalize_selection_state()
            return

        elements = self.get_selected_elements()
        if not (self._moving or self._resizing or self._rotating):
            return

        if self._moving:
            self.move_end.send(self, elements=elements)
        elif self._resizing:
            self.resize_end.send(self, elements=elements)
            for elem in elements:
                elem.trigger_update()
        elif self._rotating:
            self.rotate_end.send(self, elements=elements)

        if self._active_elem:
            self._active_origin = self._active_elem.rect()
        elif self._selection_group:
            self._selection_group._calculate_bounding_box()
            self._active_origin = self._selection_group._bounding_box

        self._resizing = False
        self._moving = False
        self._rotating = False
        self._active_region = ElementRegion.NONE

    def on_click_released(self, gesture, n_press: int, x: float, y: float):
        """
        Called when a click is completed without being turned into a drag.
        This resets the framing selection state.
        """
        if self._framing_selection:
            self._framing_selection = False
            self._selection_frame_rect = None
            self._selection_before_framing.clear()
            # The selection was already cleared in on_button_press.
            # We just need to update the final state.
            self._finalize_selection_state()

    def _finalize_selection_state(self):
        """
        Updates _active_elem and _selection_group based on the current
        selection, then queues a redraw and emits the changed signal.
        This is called after a selection operation is complete.
        """
        selected_elements = self.get_selected_elements()

        # If the active element is no longer in the selection, clear it.
        if self._active_elem and self._active_elem not in selected_elements:
            self._active_elem = None

        # If there's a selection but no active element (e.g., after framing,
        # or after deselecting the active element), pick one.
        if not self._active_elem and selected_elements:
            self._active_elem = selected_elements[-1]

        # Update the multi-selection group
        if len(selected_elements) > 1:
            # Avoid re-creating group if elements are the same
            if not self._selection_group or set(
                self._selection_group.elements
            ) != set(selected_elements):
                self._selection_group = MultiSelectionGroup(
                    selected_elements, self
                )
        else:
            self._selection_group = None

        # Emit signals with rich context
        self.active_element_changed.send(self, element=self._active_elem)
        self.selection_changed.send(
            self, elements=selected_elements, active_element=self._active_elem
        )
        self.queue_draw()

    def _get_element_world_bbox(self, elem: CanvasElement) -> Graphene.Rect:
        """
        Calculates the axis-aligned bounding box of an element in world
        coordinates, accounting for its rotation.
        """
        abs_x, abs_y, w, h = elem.rect_abs()
        angle_rad = math.radians(elem.get_angle())

        if angle_rad == 0:
            return Graphene.Rect().init(abs_x, abs_y, w, h)

        center_x, center_y = abs_x + w / 2, abs_y + h / 2
        corners_rel = [
            (-w / 2, -h / 2),
            (w / 2, -h / 2),
            (w / 2, h / 2),
            (-w / 2, h / 2),
        ]
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for rel_x, rel_y in corners_rel:
            rot_x = rel_x * cos_a - rel_y * sin_a
            rot_y = rel_x * sin_a + rel_y * cos_a
            abs_corner_x, abs_corner_y = center_x + rot_x, center_y + rot_y
            min_x, min_y = min(min_x, abs_corner_x), min(min_y, abs_corner_y)
            max_x, max_y = max(max_x, abs_corner_x), max(max_y, abs_corner_y)

        return Graphene.Rect().init(min_x, min_y, max_x - min_x, max_y - min_y)

    def _update_framing_selection(self):
        """
        Updates the selection state of all elements based on the current
        selection frame. Called during a drag operation.
        """
        if not self._selection_frame_rect:
            return

        frame_x, frame_y, frame_w, frame_h = self._selection_frame_rect

        # FIX: Do not perform selection for a zero-area frame, which happens
        # on the initial click before a drag starts. This prevents the
        # immediate re-selection of a just-deselected element.
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

                # An element is selected if it was selected before
                # (in shift mode) OR if it currently intersects the frame.
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
        if keyval == Gdk.KEY_Shift_L or keyval == Gdk.KEY_Shift_R:
            self._shift_pressed = True
            return True
        elif keyval == Gdk.KEY_Control_L or keyval == Gdk.KEY_Control_R:
            self._ctrl_pressed = True
            return True
        elif keyval == Gdk.KEY_Delete:
            selected_elements = list(self.root.get_selected())
            if selected_elements:
                self.elements_deleted.send(self, elements=selected_elements)
            return True
        return False

    def on_key_released(
        self, controller, keyval: int, keycode: int, state: Gdk.ModifierType
    ):
        if keyval == Gdk.KEY_Shift_L or keyval == Gdk.KEY_Shift_R:
            self._shift_pressed = False
        elif keyval == Gdk.KEY_Control_L or keyval == Gdk.KEY_Control_R:
            self._ctrl_pressed = False

    def get_active_element(self) -> Optional[CanvasElement]:
        return self._active_elem

    def get_selected_elements(self) -> List[CanvasElement]:
        return list(self.root.get_selected())

    def dump(self):
        """Prints a representation of the entire element hierarchy."""
        self.root.dump()


if __name__ == "__main__":
    # To see debug logs
    logging.basicConfig(level=logging.DEBUG)

    class CanvasApp(Gtk.Application):
        def __init__(self):
            super().__init__(application_id="com.example.CanvasApp")

        def do_activate(self):
            win = Gtk.ApplicationWindow(application=self)
            win.set_default_size(800, 800)

            canvas = Canvas()
            win.set_child(canvas)

            group = CanvasElement(50, 50, 400, 300, background=(0, 1, 1, 1))
            group.add(
                CanvasElement(
                    50, 50, 200, 150, background=(0, 0, 1, 1), selectable=False
                )
            )
            # Buffered element to test threaded updates
            group.add(
                CanvasElement(
                    100, 100, 150, 150, background=(0, 1, 0, 1), buffered=True
                )
            )
            group.add(
                CanvasElement(50, 100, 250, 250, background=(1, 0, 1, 1))
            )
            canvas.add(group)
            win.present()

    app = CanvasApp()
    app.run([])

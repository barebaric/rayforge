from __future__ import annotations
import logging
import math
from typing import Any, Generator, List, Tuple, Optional
import cairo
from gi.repository import Gtk, Gdk, Graphene  # type: ignore
from blinker import Signal
from .element import CanvasElement, ElementRegion
from .cursor import get_rotated_cursor


class Canvas(Gtk.DrawingArea):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = CanvasElement(
            0,
            0,
            0,  # Initial size is 0, set in do_size_allocate
            0,  # Initial size is 0, set in do_size_allocate
            canvas=self,
            parent=self,
        )
        self.active_elem: Optional[CanvasElement] = None
        self.active_origin: Optional[Tuple[int, int, int, int]] = None
        self.active_element_changed = Signal()
        self._setup_interactions()

        # Interaction state
        self.hovered_elem: Optional[CanvasElement] = None
        self.hovered_region: ElementRegion = ElementRegion.NONE
        self.active_region: ElementRegion = ElementRegion.NONE

        # Rotation state
        self.original_elem_angle: float = 0.0
        self.drag_start_angle: float = 0.0

        # Cache for custom-rendered cursors to avoid recreating them
        self._cursor_cache: dict[int, Gdk.Cursor] = {}

        # Signals for undo/redo integration
        self.move_begin = Signal()
        self.move_end = Signal()
        self.resize_begin = Signal()
        self.resize_end = Signal()
        self.rotate_begin = Signal()
        self.rotate_end = Signal()
        self.elements_deleted = Signal()

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

    def size(self) -> Tuple[int, int]:
        return self.root.size()

    def _setup_interactions(self):
        self.click_gesture = Gtk.GestureClick()
        self.click_gesture.connect("pressed", self.on_button_press)
        self.add_controller(self.click_gesture)

        self.motion_controller = Gtk.EventControllerMotion()
        self.motion_controller.connect("motion", self.on_motion)
        self.motion_controller.connect("leave", self.on_motion_leave)
        self.add_controller(self.motion_controller)

        self.drag_gesture = Gtk.GestureDrag()
        self.drag_gesture.connect("drag-update", self.on_mouse_drag)
        self.drag_gesture.connect("drag-end", self.on_button_release)
        self.add_controller(self.drag_gesture)
        self.resizing: bool = False
        self.moving: bool = False
        self.rotating: bool = False

        self.key_controller = Gtk.EventControllerKey.new()
        self.key_controller.connect("key-pressed", self.on_key_pressed)
        self.key_controller.connect("key-released", self.on_key_released)
        self.add_controller(self.key_controller)
        self.shift_pressed: bool = False
        self.ctrl_pressed: bool = False
        self.set_focusable(True)
        self.grab_focus()

        self.elem_removed = Signal()

    def do_size_allocate(self, width: int, height: int, baseline: int):
        self.root.set_size(width, height)
        self.root.allocate()

    def render(self, ctx: cairo.Context):
        """
        Renders the canvas content onto a given cairo context.
        This is the main drawing logic, separated for extensibility.
        """
        # Start the recursive rendering process from the root element.
        self.root.render(ctx)

        # Draw selection handles on top of everything.
        self._render_selection(ctx, self.root)

    def do_snapshot(self, snapshot):
        width, height = self.get_width(), self.get_height()
        bounds = Graphene.Rect().init(0, 0, width, height)
        ctx = snapshot.append_cairo(bounds)
        self.render(ctx)

    def _render_selection(self, ctx, elem: CanvasElement):
        if elem.selected:
            abs_x, abs_y = elem.pos_abs()
            ctx.save()

            # Apply rotation transform for the selection handles
            if elem.get_angle() != 0:
                ctx.translate(abs_x + elem.width / 2, abs_y + elem.height / 2)
                ctx.rotate(math.radians(elem.get_angle()))
                ctx.translate(
                    -(abs_x + elem.width / 2), -(abs_y + elem.height / 2)
                )

            # Draw dashed selection rectangle
            ctx.set_source_rgb(0.4, 0.4, 0.4)
            ctx.set_dash((5, 5))
            ctx.set_line_width(1)
            ctx.rectangle(abs_x, abs_y, elem.width, elem.height)
            ctx.stroke()

            # Don't draw hover/resize handles while moving an element.
            if self.moving or self.resizing or self.rotating:
                ctx.restore()
                return

            # Prepare to draw handles
            ctx.set_source_rgba(0.2, 0.5, 0.8, 0.7)  # A nice blue
            ctx.set_dash([])  # Solid line for handles

            # Rotation handle is visible on hover
            if elem.hovered:
                rx, ry, rw, rh = elem.get_region_rect(
                    ElementRegion.ROTATION_HANDLE
                )
                if rw > 0 and rh > 0:
                    ctx.save()
                    ctx.set_source_rgba(0.4, 0.4, 0.4, 0.9)
                    ctx.set_line_width(1)
                    ctx.move_to(abs_x + elem.width / 2, abs_y + ry + rh)
                    ctx.line_to(abs_x + elem.width / 2, abs_y)
                    ctx.stroke()
                    ctx.restore()
                    ctx.rectangle(abs_x + rx, abs_y + ry, rw, rh)
                    ctx.fill()

            # Corner handles are visible on hover of the whole element
            if elem.hovered:
                corner_regions = [
                    ElementRegion.TOP_LEFT,
                    ElementRegion.TOP_RIGHT,
                    ElementRegion.BOTTOM_LEFT,
                    ElementRegion.BOTTOM_RIGHT,
                ]
                for region in corner_regions:
                    rx, ry, rw, rh = elem.get_region_rect(region)
                    if rw > 0 and rh > 0:
                        ctx.rectangle(abs_x + rx, abs_y + ry, rw, rh)
                        ctx.fill()

            # Edge handles are only visible when hovering that specific region
            edge_regions = [
                ElementRegion.TOP_MIDDLE,
                ElementRegion.BOTTOM_MIDDLE,
                ElementRegion.MIDDLE_LEFT,
                ElementRegion.MIDDLE_RIGHT,
            ]
            if (
                self.hovered_elem == elem
                and self.hovered_region in edge_regions
            ):
                rx, ry, rw, rh = elem.get_region_rect(self.hovered_region)
                if rw > 0 and rh > 0:
                    ctx.rectangle(abs_x + rx, abs_y + ry, rw, rh)
                    ctx.fill()

            ctx.restore()

        # Recursively render children
        for child in elem.children:
            self._render_selection(ctx, child)

    def _update_hover_state(self, x: int, y: int) -> bool:
        """Updates hover state and returns True if a redraw is needed."""
        needs_redraw = False

        # Priority 1: Check for handle hits on the selected element.
        # This allows grabbing handles that are outside the element's body.
        final_hovered_elem = None
        new_hovered_region = ElementRegion.NONE

        selected_elem = self.active_elem
        if selected_elem and selected_elem.selected:
            elem_x, elem_y = selected_elem.pos_abs()
            local_x, local_y = x - elem_x, y - elem_y

            # Un-rotate the hover point to check against un-rotated handles
            if selected_elem.get_angle() != 0:
                angle_rad = math.radians(-selected_elem.get_angle())
                center_x, center_y = (
                    selected_elem.width / 2,
                    selected_elem.height / 2,
                )
                cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                rot_x = (
                    center_x
                    + (local_x - center_x) * cos_a
                    - (local_y - center_y) * sin_a
                )
                rot_y = (
                    center_y
                    + (local_x - center_x) * sin_a
                    + (local_y - center_y) * cos_a
                )
                local_x, local_y = rot_x, rot_y

            region = selected_elem.check_region_hit(int(local_x), int(local_y))

            # If a handle is hit, it takes priority
            if region not in [ElementRegion.NONE, ElementRegion.BODY]:
                new_hovered_region = region
                final_hovered_elem = selected_elem

        # Priority 2: If no handle was hit, check for body hits on any element.
        if not final_hovered_elem:
            body_hit_elem = self.root.get_elem_hit(
                x - self.root.x, y - self.root.y, selectable=True
            )
            if body_hit_elem is self.root:
                body_hit_elem = None

            if body_hit_elem:
                final_hovered_elem = body_hit_elem
                new_hovered_region = ElementRegion.BODY

        # Update the visual hover state on the element
        if self.hovered_elem != final_hovered_elem:
            if self.hovered_elem:
                self.hovered_elem.hovered = False
            self.hovered_elem = final_hovered_elem
            if self.hovered_elem:
                self.hovered_elem.hovered = True
            needs_redraw = True

        # Update the hovered region for cursor changes and drag state
        if self.hovered_region != new_hovered_region:
            self.hovered_region = new_hovered_region
            needs_redraw = True

        return needs_redraw

    def on_button_press(self, gesture, n_press: int, x: int, y: int):
        self.grab_focus()
        # The hover state now correctly identifies handle-hovers
        self._update_hover_state(x, y)
        hit = self.hovered_elem

        # Before changing selection state, check if the hit element
        # was already selected.
        was_already_selected = hit.selected if hit else False

        # If we didn't hit an already-selected element, unselect all
        if not (hit and was_already_selected):
            self.root.unselect_all()

        if hit and hit != self.root:
            hit.selected = True
            self.active_elem = hit
            self.active_origin = hit.rect()

            # The hovered_region is now reliable for determining the action
            if was_already_selected:
                self.active_region = self.hovered_region
            else:
                self.active_region = ElementRegion.BODY

            if self.active_region == ElementRegion.BODY:
                self.moving = True
                self.resizing = False
                self.rotating = False
                self.move_begin.send(self, element=hit)
                # Bring to front logic
                if hit.parent and isinstance(hit.parent, CanvasElement):
                    parent_children = hit.parent.children
                    if hit in parent_children:
                        parent_children.remove(hit)
                        parent_children.append(hit)
                        hit.parent.mark_dirty()
            elif self.active_region == ElementRegion.ROTATION_HANDLE:
                self.resizing = False
                self.moving = False
                self.rotating = True
                self.rotate_begin.send(self, element=hit)
                self._start_rotation(hit, x, y)
            elif self.active_region != ElementRegion.NONE:
                self.resizing = True
                self.moving = False
                self.rotating = False
                self.resize_begin.send(self, element=hit)
            else:
                self.active_elem = None

        else:
            self.active_elem = None
            self.resizing = False
            self.moving = False
            self.rotating = False
            self.active_region = ElementRegion.NONE

        self.queue_draw()
        self.active_element_changed.send(self, element=self.active_elem)

    def on_motion(self, gesture, x: int, y: int):
        if self._update_hover_state(x, y):
            self.queue_draw()

        cursor = None
        if self.hovered_region == ElementRegion.BODY:
            cursor = Gdk.Cursor.new_from_name("move")
        elif self.hovered_region == ElementRegion.ROTATION_HANDLE:
            cursor = Gdk.Cursor.new_from_name("crosshair")
        elif self.hovered_region != ElementRegion.NONE and self.hovered_elem:
            # For resize handles, create a custom rotated cursor
            region_angles = {
                ElementRegion.MIDDLE_RIGHT: 0,
                ElementRegion.TOP_RIGHT: 45,
                ElementRegion.TOP_MIDDLE: 90,
                ElementRegion.TOP_LEFT: 135,
                ElementRegion.MIDDLE_LEFT: 180,
                ElementRegion.BOTTOM_LEFT: 225,
                ElementRegion.BOTTOM_MIDDLE: 270,
                ElementRegion.BOTTOM_RIGHT: 315,
            }
            # The direction of scaling is perpendicular to the handle's angle
            base_angle = region_angles.get(self.hovered_region, 0)
            elem_angle = self.hovered_elem.get_angle()
            cursor_angle = base_angle - elem_angle
            cursor = get_rotated_cursor(cursor_angle)
        else:
            cursor = Gdk.Cursor.new_from_name("default")

        self.set_cursor(cursor)

    def on_motion_leave(self, controller):
        """
        Called when the pointer leaves the canvas. Resets hover state to
        prevent sticky hover effects.
        """
        # If nothing is hovered, there's nothing to do.
        if not self.hovered_elem and self.hovered_region == ElementRegion.NONE:
            return

        if self.hovered_elem:
            self.hovered_elem.hovered = False
            self.hovered_elem = None

        self.hovered_region = ElementRegion.NONE

        self.queue_draw()

        # Reset the cursor to the default.
        cursor = Gdk.Cursor.new_from_name("default")
        self.set_cursor(cursor)

    def on_mouse_drag(self, gesture, x: int, y: int):
        if not self.active_elem or not self.active_origin:
            return

        delta_x, delta_y = x, y

        if self.moving:
            start_x, start_y, _, _ = self.active_origin
            self.active_elem.set_pos(start_x + delta_x, start_y + delta_y)
            self.queue_draw()
        elif self.resizing:
            self._resize_active_element(delta_x, delta_y)
        elif self.rotating:
            self._rotate_active_element(delta_x, delta_y)

    def _start_rotation(self, elem: CanvasElement, x: int, y: int):
        """Stores initial state for a rotation operation."""
        self.original_elem_angle = elem.get_angle()
        abs_x, abs_y = elem.pos_abs()
        center_x = abs_x + elem.width / 2
        center_y = abs_y + elem.height / 2
        self.drag_start_angle = math.degrees(
            math.atan2(y - center_y, x - center_x)
        )

    def _rotate_active_element(self, delta_x: int, delta_y: int):
        """Handles the logic for rotating an element based on drag delta."""
        if not self.active_elem:
            return

        ok, start_x, start_y = self.drag_gesture.get_start_point()
        if not ok:
            return

        current_x, current_y = start_x + delta_x, start_y + delta_y

        abs_x, abs_y = self.active_elem.pos_abs()
        center_x = abs_x + self.active_elem.width / 2
        center_y = abs_y + self.active_elem.height / 2

        current_angle = math.degrees(
            math.atan2(current_y - center_y, current_x - center_x)
        )

        angle_diff = current_angle - self.drag_start_angle
        new_angle = self.original_elem_angle + angle_diff

        self.active_elem.set_angle(new_angle)
        self.queue_draw()

    def _resize_active_element(self, delta_x: int, delta_y: int):
        """
        Handles the logic for resizing a (potentially rotated) element,
        supporting aspect ratio lock (Shift) and resize-from-center (Ctrl).
        """
        if not self.active_elem or not self.active_origin:
            return

        start_x, start_y, start_w, start_h = self.active_origin
        min_size = 20
        angle_deg = self.active_elem.get_angle()

        # 1. Transform drag delta into the element's local coordinate system
        angle_rad = math.radians(-angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        local_delta_x = delta_x * cos_a - delta_y * sin_a
        local_delta_y = delta_x * sin_a + delta_y * cos_a

        # 2. Determine which edges/corners are being dragged
        is_left = self.active_region in {
            ElementRegion.TOP_LEFT,
            ElementRegion.MIDDLE_LEFT,
            ElementRegion.BOTTOM_LEFT,
        }
        is_right = self.active_region in {
            ElementRegion.TOP_RIGHT,
            ElementRegion.MIDDLE_RIGHT,
            ElementRegion.BOTTOM_RIGHT,
        }
        is_top = self.active_region in {
            ElementRegion.TOP_LEFT,
            ElementRegion.TOP_MIDDLE,
            ElementRegion.TOP_RIGHT,
        }
        is_bottom = self.active_region in {
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
        if self.ctrl_pressed:
            dw *= 2.0
            dh *= 2.0

        # 4. Handle aspect ratio constraint if Shift is pressed
        if self.shift_pressed and start_w > 0 and start_h > 0:
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
        if self.shift_pressed and start_w > 0 and start_h > 0:
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
        if not self.ctrl_pressed:
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
        self.active_elem.set_pos(round(new_x), round(new_y))
        self.active_elem.set_size(round(new_w), round(new_h))

    def on_button_release(self, gesture, x: float, y: float):
        if self.active_elem:
            if self.moving:
                self.move_end.send(self, element=self.active_elem)
            elif self.resizing:
                self.resize_end.send(self, element=self.active_elem)
                # Trigger a final high-quality render after resize is complete
                self.active_elem.trigger_update()
            elif self.rotating:
                self.rotate_end.send(self, element=self.active_elem)

        if self.active_elem:
            self.active_origin = self.active_elem.rect()

        self.resizing = False
        self.moving = False
        self.rotating = False
        self.active_region = ElementRegion.NONE

    def on_key_pressed(
        self, controller, keyval: int, keycode: int, state: Gdk.ModifierType
    ) -> bool:
        if keyval == Gdk.KEY_Shift_L or keyval == Gdk.KEY_Shift_R:
            self.shift_pressed = True
            return True
        elif keyval == Gdk.KEY_Control_L or keyval == Gdk.KEY_Control_R:
            self.ctrl_pressed = True
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
            self.shift_pressed = False
        elif keyval == Gdk.KEY_Control_L or keyval == Gdk.KEY_Control_R:
            self.ctrl_pressed = False

    def get_active_element(self) -> Optional[CanvasElement]:
        return self.active_elem

    def get_selected_elements(self) -> List[CanvasElement]:
        return list(self.root.get_selected())


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

            group = CanvasElement(50, 50, 400, 300,
                                  background=(0, 1, 1, 1))
            group.add(CanvasElement(50, 50, 200, 150,
                                    background=(0, 0, 1, 1),
                                    selectable=False))
            # Buffered element to test threaded updates
            group.add(CanvasElement(100, 100, 150, 150,
                                    background=(0, 1, 0, 1),
                                    buffered=True))
            group.add(CanvasElement(50, 100, 250, 250,
                                    background=(1, 0, 1, 1)))
            canvas.add(group)
            win.present()

    app = CanvasApp()
    app.run([])

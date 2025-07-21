from __future__ import annotations
import logging
from typing import Any, Generator, List, Tuple, Optional
import cairo
from gi.repository import Gtk, Gdk, Graphene  # type: ignore
from blinker import Signal
from .canvaselem import CanvasElement, ElementRegion


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
        self.add_controller(self.motion_controller)

        self.drag_gesture = Gtk.GestureDrag()
        self.drag_gesture.connect("drag-update", self.on_mouse_drag)
        self.drag_gesture.connect("drag-end", self.on_button_release)
        self.add_controller(self.drag_gesture)
        self.resizing: bool = False
        self.moving: bool = False

        self.key_controller = Gtk.EventControllerKey.new()
        self.key_controller.connect("key-pressed", self.on_key_pressed)
        self.key_controller.connect("key-released", self.on_key_released)
        self.add_controller(self.key_controller)
        self.shift_pressed: bool = False
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

            # Draw dashed selection rectangle
            ctx.set_source_rgb(0.4, 0.4, 0.4)
            ctx.set_dash((5, 5))
            ctx.set_line_width(1)
            ctx.rectangle(abs_x, abs_y, elem.width, elem.height)
            ctx.stroke()

            # Prepare to draw handles
            ctx.set_source_rgba(0.2, 0.5, 0.8, 0.7)  # A nice blue
            ctx.set_dash([])  # Solid line for handles

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

        # Find element under cursor, but ignore the root element
        hit_elem = self.root.get_elem_hit(
            x - self.root.x, y - self.root.y, selectable=True
        )
        if hit_elem is self.root:
            hit_elem = None

        # Check for change in hovered element
        if self.hovered_elem != hit_elem:
            if self.hovered_elem:
                self.hovered_elem.hovered = False
            self.hovered_elem = hit_elem
            if self.hovered_elem:
                self.hovered_elem.hovered = True
            needs_redraw = True

        # Check for change in hovered region on the current hovered element
        new_hovered_region = ElementRegion.NONE
        if self.hovered_elem and self.hovered_elem.selected:
            elem_x, elem_y = self.hovered_elem.pos_abs()
            local_x, local_y = x - elem_x, y - elem_y
            new_hovered_region = self.hovered_elem.check_region_hit(
                local_x, local_y
            )

        if self.hovered_region != new_hovered_region:
            self.hovered_region = new_hovered_region
            needs_redraw = True

        return needs_redraw

    def on_button_press(self, gesture, n_press: int, x: int, y: int):
        self.grab_focus()
        hit = self.root.get_elem_hit(
            x - self.root.x, y - self.root.y, selectable=True
        )

        # Before changing selection state, check if the hit element
        # was already selected.
        was_already_selected = hit.selected if hit else False

        self.root.unselect_all()

        if hit and hit != self.root:
            hit.selected = True
            self.active_elem = hit
            self.active_origin = hit.rect()

            # If the element was not selected before this click, the action
            # should always be a "move", regardless of the hit region.
            # Otherwise, if it was already selected, check for resize handles.
            if was_already_selected:
                elem_x, elem_y = hit.pos_abs()
                local_x, local_y = x - elem_x, y - elem_y
                self.active_region = hit.check_region_hit(local_x, local_y)
            else:
                self.active_region = ElementRegion.BODY

            if self.active_region == ElementRegion.BODY:
                self.moving = True
                self.resizing = False
                # Bring to front logic
                if hit.parent and isinstance(hit.parent, CanvasElement):
                    parent_children = hit.parent.children
                    if hit in parent_children:
                        parent_children.remove(hit)
                        parent_children.append(hit)
                        hit.parent.mark_dirty()
            elif self.active_region != ElementRegion.NONE:
                self.resizing = True
                self.moving = False
            else:
                self.active_elem = None

        else:
            self.active_elem = None
            self.resizing = False
            self.moving = False
            self.active_region = ElementRegion.NONE

        self._update_hover_state(x, y)
        self.queue_draw()
        self.active_element_changed.send(self, element=self.active_elem)

    def on_motion(self, gesture, x: int, y: int):
        if self._update_hover_state(x, y):
            self.queue_draw()

        cursor_map = {
            ElementRegion.TOP_LEFT: "nw-resize",
            ElementRegion.BOTTOM_RIGHT: "se-resize",
            ElementRegion.TOP_RIGHT: "ne-resize",
            ElementRegion.BOTTOM_LEFT: "sw-resize",
            ElementRegion.TOP_MIDDLE: "n-resize",
            ElementRegion.BOTTOM_MIDDLE: "s-resize",
            ElementRegion.MIDDLE_LEFT: "w-resize",
            ElementRegion.MIDDLE_RIGHT: "e-resize",
            ElementRegion.BODY: "move",
            ElementRegion.NONE: "default",
        }
        cursor_name = cursor_map.get(self.hovered_region, "default")
        cursor = Gdk.Cursor.new_from_name(cursor_name)
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

    def _resize_active_element(self, delta_x: int, delta_y: int):
        """Handles the logic for resizing an element based on drag delta."""
        if not self.active_elem or not self.active_origin:
            return

        start_x, start_y, start_w, start_h = self.active_origin
        min_size = self.active_elem.handle_size * 2

        # Determine which edges are being dragged
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

        # 1. Calculate new rect based on the dragged handle
        new_x, new_y, new_w, new_h = (
            float(start_x),
            float(start_y),
            float(start_w),
            float(start_h),
        )
        if is_left:
            new_w, new_x = start_w - delta_x, start_x + delta_x
        elif is_right:
            new_w = start_w + delta_x

        if is_top:
            new_h, new_y = start_h - delta_y, start_y + delta_y
        elif is_bottom:
            new_h = start_h + delta_y

        # 2. Enforce minimum size
        if new_w < min_size:
            if is_left:
                new_x = start_x + start_w - min_size
            new_w = min_size
        if new_h < min_size:
            if is_top:
                new_y = start_y + start_h - min_size
            new_h = min_size

        # 3. Handle aspect ratio
        if self.shift_pressed and start_w > 0 and start_h > 0:
            aspect = start_w / start_h
            rect = new_x, new_y, new_w, new_h
            start_rect = start_x, start_y, start_w, start_h
            delta = delta_x, delta_y
            new_x, new_y, new_w, new_h = self._constrain_to_aspect_ratio(
                rect, start_rect, delta, aspect
            )

        # 4. Apply changes
        self.active_elem.set_pos(round(new_x), round(new_y))
        self.active_elem.set_size(round(new_w), round(new_h))

    def _constrain_to_aspect_ratio(
        self,
        rect: Tuple[float, float, float, float],
        start_rect: Tuple[int, int, int, int],
        delta: Tuple[int, int],
        aspect: float,
    ) -> Tuple[float, float, float, float]:
        """Adjusts rectangle to maintain aspect ratio during resize."""
        new_x, new_y, new_w, new_h = rect
        start_x, start_y, start_w, start_h = start_rect
        delta_x, delta_y = delta

        # Determine resize type
        is_corner = self.active_region in {
            ElementRegion.TOP_LEFT,
            ElementRegion.TOP_RIGHT,
            ElementRegion.BOTTOM_LEFT,
            ElementRegion.BOTTOM_RIGHT,
        }
        is_horiz_edge = self.active_region in {
            ElementRegion.MIDDLE_LEFT,
            ElementRegion.MIDDLE_RIGHT,
        }
        is_vert_edge = self.active_region in {
            ElementRegion.TOP_MIDDLE,
            ElementRegion.BOTTOM_MIDDLE,
        }

        # Adjust dimensions based on the dominant mouse movement for corners
        if is_corner:
            if abs(delta_x) > abs(delta_y):
                new_h = new_w / aspect
            else:
                new_w = new_h * aspect
        elif is_horiz_edge:
            new_h = new_w / aspect
        elif is_vert_edge:
            new_w = new_h * aspect

        # Recalculate position based on new size and which handles are dragged
        is_left = self.active_region in {
            ElementRegion.TOP_LEFT,
            ElementRegion.MIDDLE_LEFT,
            ElementRegion.BOTTOM_LEFT,
        }
        is_top = self.active_region in {
            ElementRegion.TOP_LEFT,
            ElementRegion.TOP_MIDDLE,
            ElementRegion.TOP_RIGHT,
        }

        if is_left:
            new_x = start_x + start_w - new_w
        if is_top:
            new_y = start_y + start_h - new_h

        # Center the resize for edge drags
        if is_horiz_edge:
            new_y = start_y + (start_h - new_h) / 2
        if is_vert_edge:
            new_x = start_x + (start_w - new_w) / 2

        return new_x, new_y, new_w, new_h

    def on_button_release(self, gesture, x: float, y: float):
        if self.active_elem and self.resizing:
            # Trigger a final high-quality render after resize is complete
            self.active_elem.trigger_update()

        self.resizing = False
        self.moving = False
        self.active_region = ElementRegion.NONE

    def on_key_pressed(
        self, controller, keyval: int, keycode: int, state: Gdk.ModifierType
    ) -> bool:
        if keyval == Gdk.KEY_Shift_L or keyval == Gdk.KEY_Shift_R:
            self.shift_pressed = True
            return True
        elif keyval == Gdk.KEY_Delete:
            self.root.remove_selected()
            self.active_elem = None
            self.active_origin = None
            self.queue_draw()
            self.active_element_changed.send(self, element=None)
            return True
        return False

    def on_key_released(
        self, controller, keyval: int, keycode: int, state: Gdk.ModifierType
    ):
        if keyval == Gdk.KEY_Shift_L or keyval == Gdk.KEY_Shift_R:
            self.shift_pressed = False

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

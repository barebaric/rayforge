from __future__ import annotations
import logging
from typing import Any, Generator, List, Tuple, Optional, Union
import cairo
from gi.repository import Gtk, Gdk, Graphene, GLib  # type: ignore
from copy import deepcopy
from blinker import Signal
import asyncio
from ..task import task_mgr, CancelledError


logger = logging.getLogger(__name__)


class CanvasElement:
    def __init__(self,
                 x: int,
                 y: int,
                 width: int,
                 height: int,
                 selected: bool = False,
                 selectable: bool = True,
                 visible: bool = True,
                 background: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 canvas: Optional[Canvas] = None,
                 parent: Optional[Union[Canvas, CanvasElement]] = None,
                 data: Any = None,
                 clip: bool = True,
                 buffered: bool = False,
                 debounce_ms: int = 50):
        logger.debug(
            f"CanvasElement.__init__: x={x}, y={y}, width={width}, "
            f"height={height}"
        )

        self.x: int = x  # Relative to parent (or canvas if top-level)
        self.y: int = y  # Relative to parent (or canvas if top-level)
        self.width: int = width  # Width in pixels
        self.height: int = height  # Height in pixels
        self.selected: bool = selected
        self.selectable: bool = selectable
        self.visible: bool = visible
        self.surface: Optional[cairo.ImageSurface] = None
        self.canvas: Optional[Canvas] = canvas
        self.parent: Optional[Union[Canvas, CanvasElement]] = parent
        self.children: List[CanvasElement] = []
        self.background: Tuple[float, float, float, float] = background
        self.data: Any = data
        self.dirty: bool = True
        self.clip: bool = clip
        self.buffered: bool = buffered
        self.debounce_ms: int = debounce_ms
        self._debounce_timer_id: Optional[int] = None
        self._update_task: Optional[asyncio.Task] = None

    def mark_dirty(self, ancestors: bool = True, recursive: bool = False):
        self.dirty = True
        if ancestors and isinstance(self.parent, CanvasElement):
            self.parent.mark_dirty(ancestors=ancestors)
        if recursive:
            for child in self.children:
                child.mark_dirty(ancestors=False, recursive=True)

    def copy(self) -> CanvasElement:
        return deepcopy(self)

    def add(self, elem: CanvasElement):
        self.children.append(elem)
        elem.canvas = self.canvas
        elem.parent = self
        elem.allocate()
        self.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()

    def insert(self, index: int, elem: CanvasElement):
        self.children.insert(index, elem)
        elem.canvas = self.canvas
        elem.parent = self
        elem.allocate()
        self.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()

    def set_visible(self, visible: bool = True):
        self.visible = visible
        self.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()

    def find_by_data(self, data: Any) -> Optional[CanvasElement]:
        if data == self.data:
            return self
        for child in self.children:
            result = child.find_by_data(data)
            if result:
                return result
        return None

    def find_by_type(
        self, thetype: Any
    ) -> Generator[CanvasElement, None, None]:
        # Searches itself and children recursively
        if isinstance(self, thetype):
            yield self
        for child in self.children[:]:
            result = child.find_by_type(thetype)
            for elem in result:
                yield elem

    def data_by_type(self, thetype: Any) -> Generator[Any, None, None]:
        for elem in self.find_by_type(thetype):
            yield elem.data

    def remove_all(self):
        """Removes all children"""
        children = self.children
        self.children = []
        if self.canvas is not None:
            for child in children:
                self.canvas.elem_removed.send(self, child=child)
        self.mark_dirty()

    def remove(self):
        assert self.parent is not None
        self.parent.remove_child(self)

    def remove_child(self, elem: CanvasElement):
        """
        Not recursive.
        """
        for child in self.children[:]:
            if child == elem:
                self.children.remove(child)
                if self.canvas:
                    self.canvas.elem_removed.send(self, child=child)
        self.mark_dirty()

    def get_selected(self):
        if self.selected:
            yield self
        for child in self.children[:]:
            result = child.get_selected()
            for elem in result:
                yield elem

    def get_selected_data(self):
        for elem in self.get_selected():
            yield elem.data

    def remove_selected(self):
        for child in self.children[:]:
            if child.selected:
                self.children.remove(child)
                if self.canvas:
                    self.canvas.elem_removed.send(self, child=child)
            else:
                child.remove_selected()
        self.mark_dirty()

    def unselect_all(self):
        for child in self.children:
            child.unselect_all()
        self.selected = False
        self.mark_dirty()

    def set_pos(self, x: int, y: int):
        if self.x != x or self.y != y:
            self.x, self.y = x, y
            if isinstance(self.parent, CanvasElement):
                self.parent.mark_dirty()

    def pos(self) -> Tuple[int, int]:
        return self.x, self.y

    def pos_abs(self) -> Tuple[int, int]:
        parent_x, parent_y = 0, 0
        if isinstance(self.parent, CanvasElement):
            parent_x, parent_y = self.parent.pos_abs()
        return self.x+parent_x, self.y+parent_y

    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def set_size(self, width: int, height: int):
        width = int(width)
        height = int(height)
        if width != self.width or height != self.height:
            self.width, self.height = width, height
            self.allocate()
            self.mark_dirty()
            self.trigger_update()
            if self.canvas:
                self.canvas.queue_draw()

    def rect(self) -> Tuple[int, int, int, int]:
        """returns x, y, width, height"""
        return self.x, self.y, self.width, self.height

    def rect_abs(self) -> Tuple[int, int, int, int]:
        x, y = self.pos_abs()
        return x, y, self.width, self.height

    def get_aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0  # Avoid division by zero
        return self.width / self.height

    def allocate(self, force: bool = False):
        for child in self.children:
            child.allocate(force)

        # If not buffered, there is no surface to allocate.
        if not self.buffered:
            self.surface = None
            return

        # If the size didn't change, do nothing.
        if self.surface is not None and not force and \
                self.surface.get_width() == self.width and \
                self.surface.get_height() == self.height:
            return

        if self.width > 0 and self.height > 0:
            # The surface is now created by the update process.
            # We can trigger an update here to generate it.
            self.trigger_update()
        else:
            self.surface = None  # Cannot create surface with zero size

    def render(self, ctx: cairo.Context):
        """
        Renders the element and its children to the given Cairo context.
        This method handles visibility, transformation, clipping, and
        recursive rendering of children.
        """
        if not self.visible:
            return

        ctx.save()
        # Translate to the element's local coordinates
        ctx.translate(self.x, self.y)

        # Apply clipping if enabled
        if self.clip:
            ctx.rectangle(0, 0, self.width, self.height)
            ctx.clip()

        # Draw the element's own content (background, buffered surface, etc.)
        self.draw(ctx)

        # Recursively render children
        for child in self.children:
            child.render(ctx)

        ctx.restore()

    def draw(self, ctx: cairo.Context):
        """
        Draws the element's content to the given context. For buffered
        elements, it paints the internal surface. For unbuffered elements,
        it just paints the background. Subclasses should override this
        for unbuffered custom drawing.
        """
        if not self.buffered or not self.surface:
            # Unbuffered: just draw the background. Subclasses will draw
            # on top.
            ctx.set_source_rgba(*self.background)
            ctx.rectangle(0, 0, self.width, self.height)
            ctx.fill()
            return

        # Draw the element by scaling its current surface. This is fast and
        # provides responsive feedback.
        source_w, source_h = (
            self.surface.get_width(),
            self.surface.get_height(),
        )
        if source_w <= 0 or source_h <= 0:
            return

        scale_x, scale_y = self.width / source_w, self.height / source_h

        ctx.save()
        ctx.scale(scale_x, scale_y)
        ctx.set_source_surface(self.surface, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint()
        ctx.restore()

    def clear_surface(self):
        """
        Clears the internal surface, if it exists. This is useful for
        buffered elements to reset their content.
        """
        if self.surface:
            ctx = cairo.Context(self.surface)
            ctx.set_source_rgba(*self.background)
            ctx.set_operator(cairo.OPERATOR_SOURCE)
            ctx.paint()
            self.mark_dirty()

    def trigger_update(self):
        """
        Schedules the async update() method to be run, with debouncing.
        """
        if not self.buffered:
            return

        # If a debouncer is already scheduled, cancel it to restart the timer.
        if self._debounce_timer_id is not None:
            GLib.source_remove(self._debounce_timer_id)
            self._debounce_timer_id = None

        if self.debounce_ms <= 0:
            self._start_update_task()
        else:
            self._debounce_timer_id = GLib.timeout_add(
                self.debounce_ms, self._start_update_task
            )

    def _start_update_task(self) -> bool:
        """Starts the asyncio update task. This is called by the debouncer."""
        self._debounce_timer_id = None

        if self._update_task and not self._update_task.done():
            self._update_task.cancel()

        key = (f"element_update_{self.__class__.__name__}", id(self))
        self._update_task = task_mgr.add_coroutine(self.update(), key=key)

        return False  # For GLib.timeout_add, run only once

    async def update(self):
        """
        The core async orchestrator for rendering a buffered element. It runs
        `render_to_surface` in a thread and applies the result on the main
        GTK thread. Subclasses should override `render_to_surface`, not this.
        """
        render_width, render_height = self.width, self.height
        if render_width <= 0 or render_height <= 0:
            return

        try:
            new_surface = await asyncio.to_thread(
                self.render_to_surface, render_width, render_height
            )
            self.surface = new_surface
            self.mark_dirty(ancestors=True)
            if self.canvas:
                self.canvas.queue_draw()
        except CancelledError:
            logger.debug(f"Update for {self.__class__.__name__} cancelled.")
        except Exception as e:
            logger.error(
                f"Error during update for {self.__class__.__name__}: {e}",
                exc_info=True
            )

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Performs the rendering to a new surface. This method is run in a
        background thread and should be overridden by subclasses for custom,
        long-running drawing logic.
        """
        if width <= 0 or height <= 0:
            return None

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        ctx.set_source_rgba(*self.background)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.paint()
        return surface

    def has_dirty_children(self) -> bool:
        if self.dirty:
            return True
        for child in self.children:
            if child.has_dirty_children():
                return True
        return False

    def get_elem_hit(
        self, x: float, y: float, selectable: bool = False
    ) -> Optional[CanvasElement]:
        """
        Check if the point (x, y) hits this elem or any of its children.
        If selectable is True, only selectable elems are considered.
        Coordinates are relative to the current element's top-left.
        """
        # Check children (child-to-parent order)
        for child in reversed(self.children):
            # Translate the coordinates to the child's local coordinate system
            child_x = x - child.x
            child_y = y - child.y
            hit = child.get_elem_hit(child_x, child_y, selectable)
            if hit:
                return hit

        if selectable and not self.selectable:
            return None

        # Check if the point is within the elem's bounds
        if 0 <= x <= self.width and 0 <= y <= self.height:
            return self

        return None

    def get_position_in_ancestor(
        self, ancestor: Union[Canvas, CanvasElement]
    ) -> Tuple[float, float]:
        """
        Calculates and returns the (x, y) pixel position of the current element
        relative to the top-left corner of the specified ancestor.
        """
        if self == ancestor:
            return 0.0, 0.0

        current: CanvasElement = self
        pos_x, pos_y = 0.0, 0.0
        while current.parent is not None and current.parent != ancestor:
            pos_x += current.x
            pos_y += current.y
            if not isinstance(current.parent, CanvasElement):
                raise ValueError(
                    "Ancestor is not in the element's parent chain")
            current = current.parent

        if current.parent != ancestor:
            # This should not happen if ancestor is in the parent chain
            raise ValueError("Ancestor is not in the element's parent chain")

        # Add the position relative to the direct parent (which is the
        # ancestor)
        pos_x += current.x
        pos_y += current.y
        return pos_x, pos_y

    def dump(self, indent: int = 0):
        print("  "*indent + self.__class__.__name__ + ':')
        print("  "*(indent+1) + "Visible:", self.visible)
        print("  "*(indent+1) + "Dirty:", self.dirty)
        print("  "*(indent+1) + "Dirty (recurs.):", self.has_dirty_children())
        print("  "*(indent+1) + "Size:", self.rect())
        for child in self.children:
            child.dump(indent+1)


class Canvas(Gtk.DrawingArea):
    def __init__(
        self, width_mm: float = 100, height_mm: float = 100, **kwargs
    ):
        super().__init__(**kwargs)
        self.root = CanvasElement(
            0,
            0,
            0,  # Initial size is 0, set in do_size_allocate
            0,  # Initial size is 0, set in do_size_allocate
            canvas=self,
            parent=self,
        )
        self.handle_size: int = 12   # Resize handle size
        self.active_elem: Optional[CanvasElement] = None
        self.active_origin: Optional[Tuple[int, int, int, int]] = None
        self.active_element_changed = Signal()
        self._setup_interactions()

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
            ctx.set_source_rgb(.4, .4, .4)
            ctx.set_dash((5, 5))
            ctx.rectangle(abs_x, abs_y, elem.width, elem.height)
            ctx.stroke()
            ctx.restore()

        # Draw resize handle
        if elem == self.active_elem:
            abs_x, abs_y = elem.pos_abs()
            ctx.save()
            ctx.set_source_rgb(.4, .4, .4)
            ctx.set_line_width(1)
            handle_x = abs_x + elem.width
            handle_y = abs_y + elem.height
            ctx.rectangle(handle_x - self.handle_size / 2,
                          handle_y - self.handle_size / 2,
                          self.handle_size,
                          self.handle_size)
            ctx.stroke()
            ctx.restore()

        # Recursively render children
        for child in elem.children:
            self._render_selection(ctx, child)

    def get_elem_handle_hit(
        self,
        elem: CanvasElement,
        x: float,
        y: float,
        selectable: bool = True,
    ) -> Optional[CanvasElement]:
        """
        Check if the point (x, y) hits the resize handle of this elem or
        any of its children.
        Coordinates are relative to the canvas's top-left.
        """
        # Translate the hit coordinates to the element's local coordinate
        # system
        elem_x, elem_y = elem.get_position_in_ancestor(self)
        local_x = x - elem_x
        local_y = y - elem_y

        for child in elem.children:
            # Pass the hit coordinates relative to the current element
            hit = self.get_elem_handle_hit(child,
                                           x,
                                           y,
                                           selectable=True)
            if hit:
                return hit

        if selectable and not elem.selectable:
            return None
        if not elem.selected:
            return None

        # Check if the point is within the elem's handle bounds (in local
        # coordinates)
        handle_x1 = elem.width - self.handle_size / 2
        handle_x2 = handle_x1 + self.handle_size
        handle_y1 = elem.height - self.handle_size/2
        handle_y2 = handle_y1 + self.handle_size

        if (
            handle_x1 <= local_x <= handle_x2
            and handle_y1 <= local_y <= handle_y2
        ):
            return elem
        return None

    def on_button_press(self, gesture, n_press: int, x: int, y: int):
        self.grab_focus()
        # Check whether the resize handle was clicked.
        # x and y are in canvas pixel coordinates
        hit = self.get_elem_handle_hit(self.root, x, y, selectable=True)
        self.root.unselect_all()

        if hit and hit != self.root:
            hit.selected = True
            self.resizing = True
            self.moving = False  # Ensure moving is false when resizing
            self.active_elem = hit
            self.active_origin = hit.rect()
            self.queue_draw()
            self.active_element_changed.send(self, element=self.active_elem)
            return

        # Check whether the element body was clicked.
        # Translate the hit coordinates to the root element's local
        # coordinate system (which is the canvas's)
        hit = self.root.get_elem_hit(
            x - self.root.x, y - self.root.y, selectable=True)
        if hit and hit != self.root:
            hit.selected = True
            self.moving = True
            self.resizing = False  # Ensure resizing is false when moving
            self.active_elem = hit
            self.active_origin = hit.rect()

            # Move the hit element to the end of its parent's children list
            if hit.parent and isinstance(hit.parent, CanvasElement):
                parent_children = hit.parent.children
                if hit in parent_children:
                    parent_children.remove(hit)
                    parent_children.append(hit)
                    hit.parent.mark_dirty()  # Mark parent dirty as child
                    # order changed

            self.queue_draw()
            self.active_element_changed.send(self, element=self.active_elem)
            return

        self.active_elem = None
        self.resizing = False
        self.moving = False
        self.queue_draw()
        self.active_element_changed.send(self, element=None)

    def on_motion(self, gesture, x: int, y: int):
        # x and y are already in canvas pixel coordinates
        hit = self.get_elem_handle_hit(self.root, x, y, selectable=True)
        if hit:
            cursor_name = "se-resize"
        else:
            cursor_name = "default"
        cursor = Gdk.Cursor.new_from_name(cursor_name)
        self.set_cursor(cursor)

    def on_mouse_drag(self, gesture, x: int, y: int):
        if self.active_elem is None or self.active_origin is None:
            return

        start_x, start_y, start_w, start_h = self.active_origin
        # x and y are the delta from the press point
        delta_x = x
        delta_y = y

        if self.moving:
            self.active_elem.set_pos(start_x + delta_x,
                                     start_y + delta_y)

        if self.resizing:
            new_w = max(self.handle_size, start_w + delta_x)
            # Ensure the new size does not exceed the parent's bounds
            if isinstance(self.active_elem.parent, CanvasElement):
                new_w = min(
                    new_w, self.active_elem.parent.width - self.active_elem.x)

            if self.shift_pressed:
                aspect = start_w / start_h if start_h != 0 else 1
                new_h = round(new_w / aspect)
            else:
                new_h = max(self.handle_size, start_h + delta_y)
                # Ensure the new size does not exceed the parent's
                # bounds
                if isinstance(self.active_elem.parent, CanvasElement):
                    new_h = min(
                        new_h,
                        self.active_elem.parent.height - self.active_elem.y,
                    )

            self.active_elem.set_size(new_w, new_h)
            self.active_elem.allocate()  # Reallocate surface for new size

        self.queue_draw()

    def on_button_release(self, gesture, x: float, y: float):
        self.resizing = False
        self.moving = False

    def on_key_pressed(self, controller, keyval: int, keycode: int,
                       state: Gdk.ModifierType):
        if keyval == Gdk.KEY_Shift_L or keyval == Gdk.KEY_Shift_R:
            self.shift_pressed = True
        elif keyval == Gdk.KEY_Delete:
            self.root.remove_selected()
            self.active_elem = None
            self.active_origin = None
            self.queue_draw()
            self.active_element_changed.send(self, element=None)

    def on_key_released(self, controller, keyval: int, keycode: int,
                        state: Gdk.ModifierType):
        if keyval == Gdk.KEY_Shift_L or keyval == Gdk.KEY_Shift_R:
            self.shift_pressed = False

    def get_active_element(self) -> Optional[CanvasElement]:
        return self.active_elem

    def get_selected_elements(self) -> List[CanvasElement]:
        return list(self.root.get_selected())


if __name__ == "__main__":
    class CanvasApp(Gtk.Application):
        def __init__(self):
            super().__init__(application_id="com.example.CanvasApp")

        def do_activate(self):
            win = Gtk.ApplicationWindow(application=self)
            win.set_default_size(800, 800)
            # Canvas size is now in pixels
            canvas = Canvas(800, 800)
            win.set_child(canvas)
            # Element sizes and positions are now in pixels
            group = CanvasElement(50, 50, 400, 300,
                                  background=(0, 1, 1, 1))
            group.add(CanvasElement(50, 50, 200, 150,
                                    background=(0, 0, 1, 1),
                                    selectable=False))
            group.add(CanvasElement(100, 100, 150, 150,
                                    background=(0, 1, 0, 1)))
            group.add(CanvasElement(50, 100, 250, 250,
                                    background=(1, 0, 1, 1)))
            canvas.add(group)
            win.present()

    app = CanvasApp()
    app.run([])

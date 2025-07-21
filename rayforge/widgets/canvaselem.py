from __future__ import annotations
import os
import logging
from typing import TYPE_CHECKING, Any, Generator, List, Tuple, Optional, Union
import cairo
from gi.repository import GLib  # type: ignore
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum, auto

# Forward declaration for type hinting
if TYPE_CHECKING:
    from .canvas import Canvas


logger = logging.getLogger(__name__)
max_workers = max(
    1, (os.cpu_count() or 1) - 2
)  # Reserve 2 threads for UI responsiveness


class ElementRegion(Enum):
    NONE = auto()
    BODY = auto()
    TOP_LEFT = auto()
    TOP_MIDDLE = auto()
    TOP_RIGHT = auto()
    MIDDLE_LEFT = auto()
    MIDDLE_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_MIDDLE = auto()
    BOTTOM_RIGHT = auto()


class CanvasElement:
    # A shared thread pool for all element updates.
    _executor = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="CanvasElementWorker"
    )

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        selected: bool = False,
        selectable: bool = True,
        visible: bool = True,
        background: Tuple[float, float, float, float] = (0, 0, 0, 0),
        canvas: Optional["Canvas"] = None,
        parent: Optional[Union["Canvas", CanvasElement]] = None,
        data: Any = None,
        clip: bool = True,
        buffered: bool = False,
        debounce_ms: int = 50,
    ):
        logger.debug(
            f"CanvasElement.__init__: x={x}, y={y}, width={width}, "
            f"height={height}"
        )

        self.x: int = x
        self.y: int = y
        self.width: int = width
        self.height: int = height
        self.selected: bool = selected
        self.selectable: bool = selectable
        self.visible: bool = visible
        self.surface: Optional[cairo.ImageSurface] = None
        self.canvas: Optional["Canvas"] = canvas
        self.parent: Optional[Union["Canvas", CanvasElement]] = parent
        self.children: List[CanvasElement] = []
        self.background: Tuple[float, float, float, float] = background
        self.data: Any = data
        self.dirty: bool = True
        self.clip: bool = clip
        self.buffered: bool = buffered
        self.debounce_ms: int = debounce_ms
        self._debounce_timer_id: Optional[int] = None
        self._update_future: Optional[Future] = None

        # UI interaction state
        self.hovered: bool = False
        self.handle_size: int = 30

    def trigger_update(self):
        """Schedules render_to_surface to be run in the thread pool."""
        if not self.buffered:
            return

        if self._debounce_timer_id is not None:
            GLib.source_remove(self._debounce_timer_id)

        if self.debounce_ms <= 0:
            self._start_update()
        else:
            self._debounce_timer_id = GLib.timeout_add(
                self.debounce_ms, self._start_update
            )

    def _start_update(self) -> bool:
        """Submits the rendering task to the thread pool."""
        self._debounce_timer_id = None

        if self._update_future and not self._update_future.done():
            self._update_future.cancel()

        render_width, render_height = self.width, self.height
        if render_width <= 0 or render_height <= 0:
            return False

        # Submit the thread-safe part to the executor
        self._update_future = self._executor.submit(
            self.render_to_surface, render_width, render_height
        )
        # Add a callback to handle the result on the main thread
        self._update_future.add_done_callback(self._on_update_complete)

        return False  # For GLib.timeout_add, run only once

    def _on_update_complete(self, future: Future):
        """
        Callback executed by the worker thread when render_to_surface is done.
        It schedules the UI update to happen on the main GTK thread.
        """
        if future.cancelled():
            logger.debug(f"Update for {self.__class__.__name__} cancelled.")
            return

        if exc := future.exception():
            logger.error(
                f"Error in background update for {self.__class__.__name__}: "
                f"{exc}",
                exc_info=exc,
            )
            return

        # The result is the new cairo surface
        new_surface = future.result()

        # IMPORTANT: Schedule the UI-modifying part to run on the main thread
        GLib.idle_add(self._apply_surface, new_surface)

    def _apply_surface(
        self, new_surface: Optional[cairo.ImageSurface]
    ) -> bool:
        """Applies the new surface. Runs on the main GTK thread."""
        self.surface = new_surface
        self.mark_dirty(ancestors=True)
        if self.canvas:
            self.canvas.queue_draw()
        # The future is now complete, clear it.
        self._update_future = None
        return False  # Do not call again

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Performs the rendering to a new surface. This method is run in a
        background thread and should be overridden by subclasses for custom,
        long-running drawing logic. It must be thread-safe.
        """
        if width <= 0 or height <= 0:
            return None

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        ctx.set_source_rgba(*self.background)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.paint()
        return surface

    def get_region_rect(
        self, region: ElementRegion
    ) -> Tuple[int, int, int, int]:
        """
        Returns the rectangle (x, y, w, h) for a given region in
        local coordinates.
        """
        w, h = self.width, self.height
        hs = self.handle_size

        # Corner regions are hs x hs squares
        if region == ElementRegion.TOP_LEFT:
            return 0, 0, hs, hs
        if region == ElementRegion.TOP_RIGHT:
            return w - hs, 0, hs, hs
        if region == ElementRegion.BOTTOM_LEFT:
            return 0, h - hs, hs, hs
        if region == ElementRegion.BOTTOM_RIGHT:
            return w - hs, h - hs, hs, hs

        # Edge regions are between the corners
        if region == ElementRegion.TOP_MIDDLE:
            return hs, 0, w - 2 * hs, hs
        if region == ElementRegion.BOTTOM_MIDDLE:
            return hs, h - hs, w - 2 * hs, hs
        if region == ElementRegion.MIDDLE_LEFT:
            return 0, hs, hs, h - 2 * hs
        if region == ElementRegion.MIDDLE_RIGHT:
            return w - hs, hs, hs, h - 2 * hs

        if region == ElementRegion.BODY:
            return 0, 0, w, h

        return 0, 0, 0, 0  # For NONE or other cases

    def check_region_hit(self, x: int, y: int) -> ElementRegion:
        """Checks which region is hit at local coordinates (x, y)."""
        # Check handles first as they are on top of the body
        regions_to_check = [
            ElementRegion.TOP_LEFT,
            ElementRegion.TOP_RIGHT,
            ElementRegion.BOTTOM_LEFT,
            ElementRegion.BOTTOM_RIGHT,
            ElementRegion.TOP_MIDDLE,
            ElementRegion.BOTTOM_MIDDLE,
            ElementRegion.MIDDLE_LEFT,
            ElementRegion.MIDDLE_RIGHT,
        ]
        for region in regions_to_check:
            rx, ry, rw, rh = self.get_region_rect(region)
            # Make sure handle regions are not negative in size
            if rw > 0 and rh > 0 and rx <= x <= rx + rw and ry <= y < ry + rh:
                return region

        # If no handle is hit, check the body
        bx, by, bw, bh = self.get_region_rect(ElementRegion.BODY)
        if bx <= x < bx + bw and by <= y < by + bh:
            return ElementRegion.BODY

        return ElementRegion.NONE

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
        if self.selected:
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
        return self.x + parent_x, self.y + parent_y

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
        if (
            self.surface is not None
            and not force
            and self.surface.get_width() == self.width
            and self.surface.get_height() == self.height
        ):
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
        self, ancestor: Union["Canvas", CanvasElement]
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
                    "Ancestor is not in the element's parent chain"
                )
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
        print("  " * indent + self.__class__.__name__ + ":")
        print("  " * (indent + 1) + "Visible:", self.visible)
        print("  " * (indent + 1) + "Dirty:", self.dirty)
        print(
            "  " * (indent + 1) + "Dirty (recurs.):", self.has_dirty_children()
        )
        print("  " * (indent + 1) + "Size:", self.rect())
        for child in self.children:
            child.dump(indent + 1)

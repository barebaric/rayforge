from __future__ import annotations
import os
import logging
from typing import TYPE_CHECKING, Any, Generator, List, Tuple, Optional, Union
import cairo
from gi.repository import GLib  # type: ignore
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, Future
from .region import ElementRegion, get_region_rect, check_region_hit
from ...core.matrix import Matrix

# Forward declaration for type hinting
if TYPE_CHECKING:
    from .canvas import Canvas


logger = logging.getLogger(__name__)
max_workers = max(
    1, (os.cpu_count() or 1) - 2
)  # Reserve 2 threads for UI responsiveness


class CanvasElement:
    # A shared thread pool for all element updates.
    _executor = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="CanvasElementWorker"
    )

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
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
        angle: float = 0.0,
        pixel_perfect_hit: bool = False,
    ):
        logger.debug(
            f"CanvasElement.__init__: x={x}, y={y}, width={width}, "
            f"height={height}"
        )

        self.x: float = float(x)
        self.y: float = float(y)
        self.width: float = float(width)
        self.height: float = float(height)
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
        self.angle: float = angle
        self.pixel_perfect_hit = pixel_perfect_hit

        # New matrix-based transform properties
        self.local_transform: Matrix = Matrix.identity()
        self._world_transform: Matrix = Matrix.identity()
        self._transform_dirty: bool = True

        if self.pixel_perfect_hit and not self.buffered:
            raise ValueError(
                "pixel perfect hit cannot be used on unbuffered elements"
            )

        # UI interaction state
        self.hovered: bool = False
        self.handle_size: float = 15.0

        # Initial synchronization
        self._rebuild_local_transform()

    def _rebuild_local_transform(self):
        """
        Builds a matrix that ONLY handles transformations within the element's
        local space (e.g., rotation around its center). It does NOT include
        the element's x/y position.
        """
        center_x, center_y = self.width / 2, self.height / 2

        # This matrix now only contains the rotation around the local center.
        self.local_transform = Matrix.rotation(
            self.angle, center=(center_x, center_y)
        )

        # The dirty flag cascade remains correct.
        self.mark_dirty(ancestors=False, recursive=True)

    def get_world_transform(self) -> Matrix:
        """
        Calculates the full world transformation matrix for this element,
        which maps its local (0,0) origin to its final place on the canvas.
        """
        if not self._transform_dirty:
            return self._world_transform

        parent_world_transform = Matrix.identity()
        if isinstance(self.parent, CanvasElement):
            parent_world_transform = self.parent.get_world_transform()

        # The element's own contribution to the transform:
        # Its translation (x,y) from its parent's origin.
        m_trans = Matrix.translation(self.x, self.y)

        # The transformation order must match the render path:
        # Parent -> Translate -> Local (Rotation)
        # With the custom __matmul__, this is written as L @ T @ P.
        self._world_transform = (
            self.local_transform @ m_trans @ parent_world_transform
        )

        self._transform_dirty = False
        return self._world_transform

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

        render_width, render_height = round(self.width), round(self.height)
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
    ) -> Tuple[float, float, float, float]:
        """
        Returns the rectangle (x, y, w, h) for a given region in
        local coordinates by calling the generic utility function.
        """
        return get_region_rect(
            region, self.width, self.height, self.handle_size
        )

    def check_region_hit(self, x_abs: float, y_abs: float) -> ElementRegion:
        """
        Checks which region is hit by transforming the absolute point into
        the element's local coordinate space.
        """
        # 1. Get the full world transform of this element.
        world_transform = self.get_world_transform()

        # 2. Get its inverse to map world points back to local space.
        try:
            inv_world = world_transform.invert()
        except Exception:  # Catch potential numpy LinAlgError
            return ElementRegion.NONE  # Not invertible, cannot be hit.

        # 3. Transform the absolute mouse point into the element's local space.
        local_x, local_y = inv_world.transform_point((x_abs, y_abs))

        # 4. Use the new, simple, local-space checker.
        return check_region_hit(
            local_x, local_y, self.width, self.height, self.handle_size
        )

    def mark_dirty(self, ancestors: bool = True, recursive: bool = False):
        self.dirty = True
        self._transform_dirty = True
        if ancestors and isinstance(self.parent, CanvasElement):
            self.parent.mark_dirty(ancestors=ancestors)
        if recursive:
            for child in self.children:
                child.mark_dirty(ancestors=False, recursive=True)

    def copy(self) -> CanvasElement:
        return deepcopy(self)

    def add(self, elem: CanvasElement):
        if elem.parent:
            elem.parent.remove_child(elem)
        self.children.append(elem)
        elem.canvas = self.canvas
        elem.parent = self
        elem.allocate()
        self.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()

    def insert(self, index: int, elem: CanvasElement):
        if elem.parent:
            elem.parent.remove_child(elem)
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

    def get_all_children_recursive(
        self,
    ) -> Generator[CanvasElement, None, None]:
        """
        Recursively yields all descendant elements.
        """
        for child in self.children:
            yield child
            yield from child.get_all_children_recursive()

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

    def set_pos(self, x: float, y: float):
        if self.x != x or self.y != y:
            self.x, self.y = x, y
            self._rebuild_local_transform()
            if isinstance(self.parent, CanvasElement):
                self.parent.mark_dirty()

    def pos(self) -> Tuple[float, float]:
        return self.x, self.y

    def pos_abs(self) -> Tuple[float, float]:
        """
        Returns the absolute position by extracting it from the world matrix.
        """
        world_transform = self.get_world_transform()
        # The absolute position is the translation component of the matrix.
        return world_transform.get_translation()

    def size(self) -> Tuple[float, float]:
        return self.width, self.height

    def set_size(self, width: float, height: float):
        width = float(width)
        height = float(height)
        if width != self.width or height != self.height:
            self.width, self.height = width, height
            self._rebuild_local_transform()
            self.allocate()
            self.mark_dirty()
            self.trigger_update()
            if self.canvas:
                self.canvas.queue_draw()

    def rect(self) -> Tuple[float, float, float, float]:
        """returns x, y, width, height"""
        return self.x, self.y, self.width, self.height

    def rect_abs(self) -> Tuple[float, float, float, float]:
        x, y = self.pos_abs()
        return x, y, self.width, self.height

    def get_aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0  # Avoid division by zero
        return self.width / self.height

    def set_angle(self, angle: float):
        """Sets the rotation angle in degrees."""
        if self.angle == angle:
            return
        self.angle = angle % 360
        self._rebuild_local_transform()
        if isinstance(self.parent, CanvasElement):
            self.parent.mark_dirty()

    def get_angle(self) -> float:
        """Gets the rotation angle in degrees."""
        return self.angle

    def get_world_angle(self) -> float:
        """Returns the total world angle by decomposing the world matrix."""
        world_transform = self.get_world_transform()
        return world_transform.get_rotation()

    def get_world_center(self) -> Tuple[float, float]:
        """
        Calculates the element's center point in world coordinates,
        accounting for all parent transformations.
        """
        # The local center is simple
        local_center = (self.width / 2, self.height / 2)
        # The world transform correctly maps this local point to world space
        return self.get_world_transform().transform_point(local_center)

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
            and self.surface.get_width() == round(self.width)
            and self.surface.get_height() == round(self.height)
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
        Renders the element using a hybrid approach:
        1. Use cairo to translate to the element's x,y position.
        2. Use the matrix to apply local transformations like rotation.
        """
        if not self.visible:
            return

        ctx.save()

        # 1. Perform the simple translation to the element's position.
        ctx.translate(self.x, self.y)

        # 2. Apply the local transformation matrix (which is just rotation).
        m = self.local_transform.m
        cairo_matrix = cairo.Matrix(
            m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2]
        )
        ctx.transform(cairo_matrix)

        # The rest of the logic operates in the element's simple,
        # un-rotated local space, where its top-left is at (0, 0).
        if self.clip:
            ctx.rectangle(0, 0, self.width, self.height)
            ctx.clip()

        self.draw(ctx)

        # Child rendering now correctly inherits the full transform.
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
        Coordinates are relative to the current element's top-left frame.
        """
        # --- Part 1: Check children first (top-most are checked first) ---
        for child in reversed(self.children):
            # The transformation from child-local space to parent-local
            # space is:
            # 1. Apply the child's local rotation/scale (child.local_transform)
            # 2. Translate by the child's position (child.x, child.y)
            # With our custom matmul, the order is reversed: L @ T
            transform_child_to_parent = (
                child.local_transform @ Matrix.translation(child.x, child.y)
            )

            # To go from parent-local to child-local, we need the inverse.
            transform_parent_to_child = transform_child_to_parent.invert()

            # Transform the hit point (which is in our local space) into the
            # child's local space.
            child_x, child_y = transform_parent_to_child.transform_point(
                (x, y)
            )

            # Recursively call hit-testing on the child with its own local
            # coords.
            hit = child.get_elem_hit(child_x, child_y, selectable)
            if hit:
                return hit

        # --- Part 2: If no children were hit, check this element itself ---
        if selectable and not self.selectable:
            return None

        # The incoming coordinates (x, y) are already in our local,
        # untransformed frame. We just need to check if they fall within our
        # geometry.
        # We use the generic, simplified check_region_hit from region.py.
        hit_region = check_region_hit(
            x, y, self.width, self.height, self.handle_size
        )

        if hit_region == ElementRegion.NONE:
            return None  # Not within this element's bounding box at all

        # If pixel-perfect hit is required, perform the check.
        # This requires transforming the local point to a surface pixel
        # coordinate.
        if self.pixel_perfect_hit:
            if self.is_pixel_opaque(x, y):
                return self
            return None

        return self

    def is_pixel_opaque(self, local_x: float, local_y: float) -> bool:
        """
        Checks if the pixel at the given LOCAL coordinates is opaque on the
        element's surface. Returns True if opaque, False if transparent.
        """
        if not self.buffered or not self.surface:
            # Cannot perform pixel check on non-buffered elements or if the
            # surface doesn't exist. Default to treating it as opaque.
            return True

        surface_w = self.surface.get_width()
        surface_h = self.surface.get_height()

        # Check if the local point is even within the element's bounds.
        if not (0 <= local_x < self.width and 0 <= local_y < self.height):
            return False

        # Scale local coordinates to surface pixel coordinates.
        # This handles cases where the element's size and the buffered
        # surface's resolution are different.
        surface_x = int(local_x * (surface_w / self.width))
        surface_y = int(local_y * (surface_h / self.height))

        # Check if the calculated pixel is within the surface's bounds.
        if not (0 <= surface_x < surface_w and 0 <= surface_y < surface_h):
            return False

        # Read the alpha value from the cairo surface data buffer.
        data = self.surface.get_data()
        stride = self.surface.get_stride()

        # Pixel format is ARGB32, but often stored as BGRA in memory.
        # The alpha channel is the 4th byte in the 4-byte pixel group.
        pixel_offset = surface_y * stride + surface_x * 4
        alpha = data[pixel_offset + 3]

        # Consider any non-zero alpha as a "hit".
        return alpha > 0

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
        """Prints a representation of the element and its children."""
        pad = "  " * indent
        print(f"{pad}{self.__class__.__name__}: (Data: {self.data})")
        print(f"{pad}  Visible: {self.visible}, Selected: {self.selected}")
        print(f"{pad}  Rect: {self.rect()}")
        print(f"{pad}  Angle: {self.angle}, Clip: {self.clip}")
        if self.buffered:
            surface_info = "None"
            if self.surface:
                surface_info = (
                    f"Cairo Surface ({self.surface.get_width()}x"
                    f"{self.surface.get_height()})"
                )
            print(f"{pad}  Buffered: True, Surface: {surface_info}")
        if self.children:
            print(f"{pad}  Children ({len(self.children)}):")
            for child in self.children:
                child.dump(indent + 1)

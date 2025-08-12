from __future__ import annotations
import os
import math
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    List,
    Tuple,
    Optional,
    Union,
)
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
# Reserve 2 threads for UI responsiveness
max_workers = max(1, (os.cpu_count() or 1) - 2)


class CanvasElement:
    """
    The base class for all objects rendered on a Canvas.

    This class provides a hierarchical structure (parent-child),
    matrix-based transformations (translation, rotation, scale),
    asynchronous off-thread rendering for performance ("buffering"),
    and basic UI interaction logic like hit-testing.
    """

    # A shared thread pool for all element background updates.
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
        """
        Initializes a new CanvasElement.

        Args:
            x: The x-coordinate relative to the parent.
            y: The y-coordinate relative to the parent.
            width: The width of the element.
            height: The height of the element.
            selected: The initial selection state.
            selectable: If the element can be selected by the user.
            visible: If the element is drawn.
            background: The background color (r, g, b, a).
            canvas: The root Canvas this element belongs to.
            parent: The parent element in the hierarchy.
            data: Arbitrary user data associated with the element.
            clip: If True, drawing is clipped to the element's
                bounding box.
            buffered: If True, the element is rendered to an
                off-screen surface in a background thread. This is
                ideal for complex, static elements. If False, the
                element is drawn directly on every frame.
            debounce_ms: The delay in milliseconds before a
                background render is triggered after a change.
            angle: The local rotation angle in degrees.
            pixel_perfect_hit: If True (and buffered=True),
                hit-testing will check the transparency of the
                pixel on the element's surface.
        """
        logger.debug(
            f"CanvasElement.__init__: x={x}, y={y}, width={width}, "
            f"height={height}"
        )

        self.x: float = float(x)
        self.y: float = float(y)
        self.width: float = float(width)
        self.height: float = float(height)
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
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

        # Matrix for local scale and rotation.
        self.local_transform: Matrix = Matrix.identity()
        # Cached matrix for the full transform to world space.
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
        Builds the local transformation matrix.

        This matrix handles scale and rotation around the element's
        geometric center. It uses a standard T-S-R-T sequence to ensure
        transformations are applied correctly around the center pivot.
        """
        center_x, center_y = self.width / 2, self.height / 2

        # 1. Translate the element's center to the origin
        t_to_origin = Matrix.translation(-center_x, -center_y)
        # 2. Scale around the origin
        m_scale = Matrix.scale(self.scale_x, self.scale_y)
        # 3. Rotate around the origin
        m_rotate = Matrix.rotation(self.angle)
        # 4. Translate the element back to its original center
        t_back_from_origin = Matrix.translation(center_x, center_y)

        # The final transform is composed from right to left:
        # T_back @ R @ S @ T_to_origin
        self.local_transform = (
            t_back_from_origin @ m_rotate @ m_scale @ t_to_origin
        )

        self.mark_dirty(ancestors=False, recursive=True)

    def get_world_transform(self) -> Matrix:
        """
        Calculates the full world transformation matrix.

        This matrix maps a point from this element's local coordinate
        space to the final canvas (world) coordinate space. It caches
        the result and only recalculates when the transform is "dirty".

        The final matrix is composed as:
        `ParentWorld @ Translate @ Local`

        This means the order of operations to transform a point from
        local space into world space is:
        1. Apply this element's local transform (scale, rotation).
        2. Apply this element's translation (x, y).
        3. Apply parent's world transform.
        """
        if not self._transform_dirty:
            return self._world_transform

        parent_world_transform = Matrix.identity()
        if isinstance(self.parent, CanvasElement):
            parent_world_transform = self.parent.get_world_transform()

        m_trans = Matrix.translation(self.x, self.y)

        # Correct pre-multiplication order:
        # Parent -> Translate -> Local (Rotate @ Scale)
        self._world_transform = (
            parent_world_transform @ m_trans @ self.local_transform
        )

        self._transform_dirty = False
        return self._world_transform

    def trigger_update(self):
        """
        Schedules a background render of the element's surface.

        If called multiple times in quick succession, the calls are
        debounced to prevent excessive updates. This has no effect
        on unbuffered elements.
        """
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
        """
        Submits the rendering task to the background thread pool.
        """
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
        Callback executed when the background render is finished.

        It schedules the final UI update to happen on the main GTK
        thread to ensure thread safety.

        Args:
            future: The Future object from the completed task.
        """
        if future.cancelled():
            logger.debug(f"Update for {self.__class__.__name__} cancelled.")
            return

        if exc := future.exception():
            logger.error(
                f"Error in background update for "
                f"{self.__class__.__name__}: {exc}",
                exc_info=exc,
            )
            return

        # The result is the new cairo surface
        new_surface = future.result()

        # Schedule the UI-modifying part to run on the main thread
        GLib.idle_add(self._apply_surface, new_surface)

    def _apply_surface(
        self, new_surface: Optional[cairo.ImageSurface]
    ) -> bool:
        """
        Applies the newly rendered surface from the background task.

        This method runs on the main GTK thread via `GLib.idle_add`.

        Args:
            new_surface: The new surface to apply, or None.
        """
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
        Performs rendering to a new surface in a background thread.

        Subclasses should override this method for custom, long-running
        drawing logic. It MUST be thread-safe. The base implementation
        simply creates a surface and fills it with the background
        color.

        Args:
            width: The integer width of the surface to create.
            height: The integer height of the surface to create.

        Returns:
            A new `cairo.ImageSurface` or `None` if size is invalid.
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
        self,
        region: ElementRegion,
        base_handle_size: float,
        scale_compensation: Union[float, Tuple[float, float]] = 1.0,
    ) -> Tuple[float, float, float, float]:
        """
        Gets the rect (x, y, w, h) for a region in local coordinates.

        Args:
            region: The `ElementRegion` to query (e.g., a handle).
            base_handle_size: The base pixel size for the handle.
            scale_compensation: The element's visual scale factor.

        Returns:
            A tuple (x, y, width, height) in local coordinates.
        """
        return get_region_rect(
            region,
            self.width,
            self.height,
            base_handle_size,
            scale_compensation,
        )

    def check_region_hit(self, x_abs: float, y_abs: float) -> ElementRegion:
        """
        Checks which region is hit at an absolute canvas position.

        It transforms the absolute point into the element's local
        coordinate space to perform the hit check.

        Args:
            x_abs: The absolute x-coordinate on the canvas.
            y_abs: The absolute y-coordinate on the canvas.

        Returns:
            The `ElementRegion` that was hit (e.g., BODY, HANDLE_SE).
        """
        world_transform = self.get_world_transform()
        try:
            inv_world = world_transform.invert()
        except Exception:
            return ElementRegion.NONE

        local_x, local_y = inv_world.transform_point((x_abs, y_abs))

        # Use the single source of truth from the canvas for handle size.
        # Fallback to a default if the element is not on a canvas.
        base_hit_size = self.canvas.BASE_HANDLE_SIZE if self.canvas else 15.0

        # Calculate the visual scale from the world transform matrix to pass
        # to the hit-testing function. This makes hit-testing scale-aware.
        m = world_transform.m
        sx = math.hypot(m[0, 0], m[1, 0])
        sy = math.hypot(m[0, 1], m[1, 1])

        return check_region_hit(
            local_x, local_y, self.width, self.height, base_hit_size, (sx, sy)
        )

    def mark_dirty(self, ancestors: bool = True, recursive: bool = False):
        """
        Flags the element and its transforms as needing an update.

        Args:
            ancestors: If True, marks all parent elements as dirty.
            recursive: If True, marks all child elements as dirty.
        """
        self.dirty = True
        self._transform_dirty = True
        if ancestors and isinstance(self.parent, CanvasElement):
            self.parent.mark_dirty(ancestors=ancestors)
        if recursive:
            for child in self.children:
                child.mark_dirty(ancestors=False, recursive=True)

    def copy(self) -> CanvasElement:
        """Creates a deep copy of the element."""
        return deepcopy(self)

    def add(self, elem: CanvasElement):
        """
        Adds a child element.

        The element is added to the end of the children list. If the
        element already has a parent, it is removed from it first.

        Args:
            elem: The `CanvasElement` to add.
        """
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
        """
        Inserts a child element at a specific index.

        Args:
            index: The index at which to insert the element.
            elem: The `CanvasElement` to insert.
        """
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
        """Sets the visibility of the element."""
        self.visible = visible
        self.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()

    def find_by_data(self, data: Any) -> Optional[CanvasElement]:
        """
        Finds the first element (self or descendant) with matching data.

        Args:
            data: The data to search for.

        Returns:
            The matching `CanvasElement` or `None`.
        """
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
        """
        Finds all elements (self or descendant) of a given type.

        Args:
            thetype: The class/type to search for.

        Yields:
            Matching `CanvasElement` instances.
        """
        if isinstance(self, thetype):
            yield self
        for child in self.children[:]:
            result = child.find_by_type(thetype)
            for elem in result:
                yield elem

    def data_by_type(self, thetype: Any) -> Generator[Any, None, None]:
        """
        Finds all data from elements of a given type.

        Args:
            thetype: The class/type to search for.

        Yields:
            The `data` attribute of matching elements.
        """
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
        """Removes all children from this element."""
        children = self.children
        self.children = []
        if self.canvas is not None:
            for child in children:
                self.canvas.elem_removed.send(self, child=child)
        self.mark_dirty()

    def remove(self):
        """Removes this element from its parent."""
        assert self.parent is not None
        self.parent.remove_child(self)

    def remove_child(self, elem: CanvasElement):
        """
        Removes a direct child element. This is not recursive.

        Args:
            elem: The child element to remove.
        """
        for child in self.children[:]:
            if child == elem:
                self.children.remove(child)
                if self.canvas:
                    self.canvas.elem_removed.send(self, child=child)
        self.mark_dirty()

    def get_selected(self) -> Generator[CanvasElement, None, None]:
        """Recursively finds and yields all selected elements."""
        if self.selected:
            yield self
        for child in self.children[:]:
            result = child.get_selected()
            for elem in result:
                yield elem

    def get_selected_data(self) -> Generator[Any, None, None]:
        """Recursively finds and yields data of selected elements."""
        for elem in self.get_selected():
            yield elem.data

    def remove_selected(self):
        """Recursively finds and removes all selected elements."""
        for child in self.children[:]:
            if child.selected:
                self.children.remove(child)
                if self.canvas:
                    self.canvas.elem_removed.send(self, child=child)
            else:
                child.remove_selected()
        self.mark_dirty()

    def unselect_all(self):
        """Recursively unselects this element and all descendants."""
        for child in self.children:
            child.unselect_all()
        if self.selected:
            self.selected = False
            self.mark_dirty()

    def set_pos(self, x: float, y: float):
        """
        Sets the element's position relative to its parent.

        Args:
            x: The new x-coordinate.
            y: The new y-coordinate.
        """
        if self.x != x or self.y != y:
            self.x, self.y = x, y
            # Only need to mark the transform as dirty.
            self.mark_dirty()
            if isinstance(self.parent, CanvasElement):
                self.parent.mark_dirty()

    def pos(self) -> Tuple[float, float]:
        """Gets the element's position relative to its parent."""
        return self.x, self.y

    def pos_abs(self) -> Tuple[float, float]:
        """
        Gets the absolute position on the canvas.

        This is calculated by extracting the translation component from
        the element's world transformation matrix.
        """
        world_transform = self.get_world_transform()
        return world_transform.get_translation()

    def size(self) -> Tuple[float, float]:
        """Gets the element's size (width, height)."""
        return self.width, self.height

    def set_size(self, width: float, height: float):
        """
        Sets the element's size.

        This rebuilds the local transform, re-allocates the backing
        surface (if buffered), and triggers a redraw.

        Args:
            width: The new width.
            height: The new height.
        """
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

    def set_scale(self, scale_x: float, scale_y: float):
        """
        Sets the scale factor of the element.

        Args:
            scale_x: The horizontal scale factor.
            scale_y: The vertical scale factor.
        """
        if self.scale_x != scale_x or self.scale_y != scale_y:
            self.scale_x, self.scale_y = scale_x, scale_y
            self._rebuild_local_transform()
            self.mark_dirty()
            if self.canvas:
                self.canvas.queue_draw()

    def rect(self) -> Tuple[float, float, float, float]:
        """
        Gets the local rect (x, y, width, height).
        """
        return self.x, self.y, self.width, self.height

    def rect_abs(self) -> Tuple[float, float, float, float]:
        """
        Gets the absolute rect (x, y, width, height).

        The x and y are the absolute position of the top-left corner.
        The width and height are the element's local size, not the
        size of the transformed bounding box.
        """
        x, y = self.pos_abs()
        return x, y, self.width, self.height

    def get_aspect_ratio(self) -> float:
        """Calculates the width-to-height aspect ratio."""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    def set_angle(self, angle: float):
        """
        Sets the local rotation angle in degrees.

        Args:
            angle: The angle in degrees (0-360).
        """
        if self.angle == angle:
            return
        self.angle = angle % 360
        self._rebuild_local_transform()
        if isinstance(self.parent, CanvasElement):
            self.parent.mark_dirty()

    def get_angle(self) -> float:
        """Gets the local rotation angle in degrees."""
        return self.angle

    def get_world_angle(self) -> float:
        """
        Gets the total rotation angle in world coordinates.

        This is calculated by decomposing the world transformation
        matrix.
        """
        world_transform = self.get_world_transform()
        return world_transform.get_rotation()

    def get_world_center(self) -> Tuple[float, float]:
        """
        Calculates the element's center point in world coordinates.
        """
        local_center = (self.width / 2, self.height / 2)
        return self.get_world_transform().transform_point(local_center)

    def allocate(self, force: bool = False):
        """
        Allocates or re-allocates resources, like the backing surface.

        For buffered elements, this triggers a surface update if the
        element's size has changed or if `force` is True.

        Args:
            force: If True, forces reallocation even if size is same.
        """
        for child in self.children:
            child.allocate(force)

        if not self.buffered:
            self.surface = None
            return

        size_changed = (
            self.surface is None
            or self.surface.get_width() != round(self.width)
            or self.surface.get_height() != round(self.height)
        )

        if not size_changed and not force:
            return

        if self.width > 0 and self.height > 0:
            # Trigger an update to generate the new surface.
            self.trigger_update()
        else:
            self.surface = None

    def render(self, ctx: cairo.Context):
        """
        Renders the element and its children to the cairo context.

        This method applies the element's transformations before
        drawing. It uses a hybrid approach:
        1. `ctx.translate` for the element's x,y position.
        2. `ctx.transform` with a matrix for local scale and rotation.

        This ensures children correctly inherit the full transform.

        Args:
            ctx: The cairo context to draw on.
        """
        if not self.visible:
            return

        ctx.save()

        # 1. Translate to the element's position in parent's frame.
        ctx.translate(self.x, self.y)

        # 2. Apply local scale and rotation via the matrix.
        m = self.local_transform.m
        cairo_matrix = cairo.Matrix(
            m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2]
        )
        ctx.transform(cairo_matrix)

        # The rest of the drawing operates in the element's simple,
        # untransformed local space (top-left at 0,0).
        if self.clip:
            ctx.rectangle(0, 0, self.width, self.height)
            ctx.clip()

        self.draw(ctx)

        for child in self.children:
            child.render(ctx)

        ctx.restore()

    def draw(self, ctx: cairo.Context):
        """
        Draws the element's own content.

        - For buffered elements, this paints the internal surface.
        - For unbuffered elements, this is the hook for subclasses to
          implement custom drawing logic. The base implementation just
          draws the background color.

        Args:
            ctx: The cairo context, already transformed to local space.
        """
        if not self.buffered or not self.surface:
            # Unbuffered: just draw the background.
            ctx.set_source_rgba(*self.background)
            ctx.rectangle(0, 0, self.width, self.height)
            ctx.fill()
            return

        source_w = self.surface.get_width()
        source_h = self.surface.get_height()

        if source_w <= 0 or source_h <= 0:
            return

        # Draw by scaling the current surface. This is fast.
        scale_x = self.width / source_w
        scale_y = self.height / source_h

        ctx.save()
        ctx.scale(scale_x, scale_y)
        ctx.set_source_surface(self.surface, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint()
        ctx.restore()

    def draw_selection_frame(self, ctx: cairo.Context):
        """
        Draws the visual selection frame for this element.
        The cairo context is assumed to be already transformed
        so that (0,0) is the top-left of the element.
        Line width, dash, and color are also assumed to be set
        by the caller (the Canvas).
        """
        ctx.rectangle(0, 0, self.width, self.height)
        ctx.stroke()

    def clear_surface(self):
        """
        Clears the internal surface of a buffered element.
        """
        if self.surface:
            ctx = cairo.Context(self.surface)
            ctx.set_source_rgba(*self.background)
            ctx.set_operator(cairo.OPERATOR_SOURCE)
            ctx.paint()
            self.mark_dirty()

    def has_dirty_children(self) -> bool:
        """Checks if this element or any descendant is dirty."""
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
        Checks for a hit on this element or its children.

        The check is performed recursively, starting with the top-most
        child. The incoming coordinates are relative to this element's
        local, untransformed coordinate space.

        Args:
            x: The x-coordinate in this element's local space.
            y: The y-coordinate in this element's local space.
            selectable: If True, only selectable elements are checked.

        Returns:
            The `CanvasElement` that was hit, or `None`.
        """
        # Part 1: Check children first (top-most are last in list).
        for child in reversed(self.children):
            # The transform from the child's local space to the parent's
            # local space is: Translate -> LocalTransform.
            # This must match the order in get_world_transform and render.
            transform_child_to_parent = (
                Matrix.translation(child.x, child.y) @ child.local_transform
            )
            # Invert to go from parent-local to child-local
            transform_parent_to_child = transform_child_to_parent.invert()

            # Transform the hit point into the child's local space.
            child_x, child_y = transform_parent_to_child.transform_point(
                (x, y)
            )

            # Recursively call hit-testing on the child.
            hit = child.get_elem_hit(child_x, child_y, selectable)
            if hit:
                return hit

        # Part 2: If no children were hit, check this element.
        if selectable and not self.selectable:
            return None

        # Check if hit is within the bounding box.
        hit_region = check_region_hit(
            x, y, self.width, self.height, self.handle_size
        )
        if hit_region == ElementRegion.NONE:
            return None

        # For pixel-perfect hits, check surface transparency.
        if self.pixel_perfect_hit:
            if self.is_pixel_opaque(x, y):
                return self
            return None

        return self

    def is_pixel_opaque(self, local_x: float, local_y: float) -> bool:
        """
        Checks if the pixel at local coordinates is opaque.

        For buffered elements, this reads the alpha channel from the
        backing surface. For unbuffered elements, it returns True.

        Args:
            local_x: The x-coordinate in local space.
            local_y: The y-coordinate in local space.

        Returns:
            True if the pixel is considered a hit (alpha > 0).
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
        surface_x = int(local_x * (surface_w / self.width))
        surface_y = int(local_y * (surface_h / self.height))

        # Check if the calculated pixel is within the surface's bounds.
        if not (0 <= surface_x < surface_w and 0 <= surface_y < surface_h):
            return False

        # Read the alpha value from the cairo surface data buffer.
        data = self.surface.get_data()
        stride = self.surface.get_stride()

        # Format is ARGB32, often BGRA in memory. Alpha is 4th byte.
        pixel_offset = surface_y * stride + surface_x * 4
        alpha = data[pixel_offset + 3]

        # Consider any non-zero alpha as a "hit".
        return alpha > 0

    def get_position_in_ancestor(
        self, ancestor: Union["Canvas", CanvasElement]
    ) -> Tuple[float, float]:
        """
        Calculates the (x, y) position relative to an ancestor.

        This method sums the local `x` and `y` properties up the
        hierarchy. It does NOT account for rotation or scaling.

        Args:
            ancestor: The ancestor to measure relative to.

        Returns:
            The (x, y) position relative to the ancestor.

        Raises:
            ValueError: If the specified ancestor is not in this
                        element's parent chain.
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
            raise ValueError("Ancestor is not in the element's parent chain")

        pos_x += current.x
        pos_y += current.y
        return pos_x, pos_y

    def dump(self, indent: int = 0):
        """Prints a debug representation of the element and children."""
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

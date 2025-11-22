from __future__ import annotations
import os
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    List,
    Tuple,
    Optional,
    Union,
    Set,
)
import cairo
from gi.repository import GLib
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, Future
from .region import ElementRegion, get_region_rect, check_region_hit
from .hittest import check_pixel_hit
from ...core.matrix import Matrix

# Forward declaration for type hinting
if TYPE_CHECKING:
    from .canvas import Canvas


logger = logging.getLogger(__name__)
# Reserve 2 threads for UI responsiveness
max_workers = max(1, (os.cpu_count() or 1) - 2)
# Define a maximum dimension for our rendering buffers to prevent cairo errors.
MAX_BUFFER_DIM = 8192


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
        matrix: Optional[Matrix] = None,
        hit_distance: float = 0.0,
        is_editable: bool = False,
        draggable: bool = False,
        show_selection_frame: bool = True,
        drag_handler_controls_transform: bool = False,
        preserves_selection_on_click: bool = False,
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
                pixel on the element's rendered surface.
            matrix: An optional transformation matrix. If provided,
                it overrides x, y, angle, and scale properties on
                initialization.
            hit_distance: For pixel-perfect hit checks, this adds a
                "fuzzy" radius around the mouse pointer. The distance is
                specified in **screen pixels** and is applied to the
                element's rendered surface. A non-zero value will check a
                circular area for any opaque pixel.
            is_editable: If True, the element can be double-clicked
                to enter a special "edit mode".
            draggable: If True, the element can be moved by dragging its
                body, and its drag behavior can be customized by
                overriding `handle_drag_move`.
            show_selection_frame: If False, the selection frame and
                handles will not be drawn for this element even when it is
                selected. Useful for custom interactive handles.
            drag_handler_controls_transform: If True, the `handle_drag_move`
                method is responsible for updating the element's transform
                itself. If False (default), it should return a constrained
                delta for the canvas to apply.
            preserves_selection_on_click: If True, clicking this element
                will not change the existing selection on the canvas. It will
                only make this element the target for a potential drag.
        """
        logger.debug(
            f"CanvasElement.__init__: x={x}, y={y}, width={width}, "
            f"height={height}"
        )

        # Primitive properties are used for initialization and by methods
        # like set_size that need to rebuild the transform. They are NOT
        # kept in sync with the matrix.
        self.x: float = float(x)
        self.y: float = float(y)
        self.width: float = float(width)
        self.height: float = float(height)
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        self.angle: float = angle

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
        self.pixel_perfect_hit = pixel_perfect_hit
        self.hit_distance: float = hit_distance
        self.is_editable: bool = is_editable
        self.draggable: bool = draggable
        self.show_selection_frame: bool = show_selection_frame
        self.drag_handler_controls_transform = drag_handler_controls_transform
        self.preserves_selection_on_click = preserves_selection_on_click

        # This is the single source of truth for the local GEOMETRIC transform.
        self.transform: Matrix = Matrix.identity()
        # This new matrix handles content orientation relative to the geometry.
        self.content_transform: Matrix = Matrix.identity()
        # Cached matrix for the full transform to world space.
        self._world_transform: Matrix = Matrix.identity()
        self._transform_dirty: bool = True

        # UI interaction state
        self.hovered: bool = False
        self._is_under_interactive_transform: bool = False

        if matrix is not None:
            self.set_transform(matrix)
        else:
            # Initial synchronization from primitive properties on creation
            self._rebuild_transform()

    @property
    def is_hovered(self) -> bool:
        """Returns True if the mouse is currently hovering over the element."""
        return self.hovered

    def _rebuild_transform(self):
        """
        Builds the unified local transform from primitive properties.

        This method should only be used for initialization or by setters
        that are intended to reset an element's shear (like `set_angle`
        or `set_scale`). All interactive transformations should modify
        the matrix directly via `set_transform`.
        """
        center_x, center_y = self.width / 2, self.height / 2
        t_to_origin = Matrix.translation(-center_x, -center_y)
        m_scale = Matrix.scale(self.scale_x, self.scale_y)
        m_rotate = Matrix.rotation(self.angle)
        t_back_from_origin = Matrix.translation(center_x, center_y)
        m_trans = Matrix.translation(self.x, self.y)

        # Build the rotation/scale part
        m_trs = t_back_from_origin @ m_rotate @ m_scale @ t_to_origin
        # Combine with translation to form the final local transform
        self.transform = m_trans @ m_trs

        self.mark_dirty(ancestors=False, recursive=True)

    def _set_transform_silent(self, matrix: Matrix):
        """
        Internal method to set the transform without notifying the parent.
        This is used to break recursion in parent-child update cycles.
        """
        self.transform = matrix
        self.mark_dirty(ancestors=True, recursive=True)

        if self.canvas:
            self.canvas.queue_draw()

    def begin_interactive_transform(self):
        """
        Notifies the element that it is being directly manipulated by the user.
        This is a hint for complex parent elements like ShrinkWrapGroup.
        """
        self._is_under_interactive_transform = True

    def end_interactive_transform(self):
        """
        Notifies the element that direct user manipulation has ended.
        This triggers a final update notification to the parent to allow it
        to consolidate the element's final state.
        """
        self._is_under_interactive_transform = False
        if isinstance(self.parent, CanvasElement):
            self.parent.on_child_transform_changed(self)

    def set_transform(self, matrix: Matrix):
        """
        Sets the element's complete local transform matrix directly and
        notifies the parent of the change.
        """
        self._set_transform_silent(matrix)
        # Notify parent of the change, allowing them to react.
        if isinstance(self.parent, CanvasElement):
            self.parent.on_child_transform_changed(self)

    def on_child_transform_changed(self, child: "CanvasElement"):
        """
        Callback triggered by a child when its transform has changed.
        Subclasses can override this to react, e.g., by updating bounds.
        The base implementation does nothing.
        """
        pass

    def on_child_list_changed(self):
        """
        Hook called when the list of children is modified (add/remove).
        Subclasses can override this to react.
        """
        pass

    def on_attached(self):
        """
        Lifecycle hook called when the element is added to a canvas.
        `self.canvas` is guaranteed to be available. Subclasses can
        override this to connect signals or initialize resources.
        """
        pass

    def on_detached(self):
        """
        Lifecycle hook called before the element is removed from a canvas.
        Subclasses can override this to disconnect signals or clean up.
        """
        pass

    def draw_overlay(self, ctx: cairo.Context):
        """
        Draws a custom overlay in world coordinates (pixel space).
        The cairo context's coordinate system is not transformed.
        Subclasses can override this to draw previews or guides.

        Args:
            ctx: The cairo context in world/pixel space.
        """
        pass

    def draw_edit_overlay(self, ctx: cairo.Context):
        """
        Draws a custom overlay for editing, only called when the element is
        the canvas's `edit_context`. The context is in screen space.

        Args:
            ctx: The cairo context in screen/pixel space.
        """
        pass

    def get_world_transform(self) -> Matrix:
        """
        Calculates the full world transformation matrix.

        This matrix maps a point from this element's local coordinate
        space to the final canvas (world) coordinate space. It caches
        the result and only recalculates when the transform is "dirty".
        """
        if not self._transform_dirty:
            return self._world_transform

        if isinstance(self.parent, CanvasElement):
            parent_world = self.parent.get_world_transform()
            self._world_transform = parent_world @ self.transform
        else:
            # For the root element, the world transform is its own transform.
            self._world_transform = self.transform

        self._transform_dirty = False
        return self._world_transform

    def get_world_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Calculates the element's axis-aligned bounding box in world
        coordinates.
        """
        # The rectangle in an element's local coordinates is defined by its
        # width and height, with its origin at (0, 0).
        local_rect = (0, 0, self.width, self.height)

        # Get the matrix that transforms from local space to world space
        world_transform = self.get_world_transform()

        # Transform the local rectangle to get its world-space bounding box
        return world_transform.transform_rectangle(local_rect)

    def trigger_update(self):
        """
        Schedules a background render of the element's surface and recursively
        triggers updates for all children.

        If called multiple times in quick succession, the calls are
        debounced to prevent excessive updates. For unbuffered elements, this
        method simply passes the update call to its children.
        """
        # If this element is buffered, schedule its own surface update.
        if self.buffered:
            if self._debounce_timer_id is not None:
                GLib.source_remove(self._debounce_timer_id)

            if self.debounce_ms <= 0:
                self._start_update()
            else:
                self._debounce_timer_id = GLib.timeout_add(
                    self.debounce_ms, self._start_update
                )

        # Always recursively trigger updates for all children, as they might
        # be buffered even if this parent element is not.
        for child in self.children:
            child.trigger_update()

    def _start_update(self) -> bool:
        """
        Submits the rendering task to the background thread pool.
        This now calculates the correct pixel dimensions for the buffer.
        """
        self._debounce_timer_id = None

        if self._update_future and not self._update_future.done():
            self._update_future.cancel()

        if not self.canvas:
            return False

        # Calculate the total transformation from this element's local space
        # to the final screen (pixel) space.
        transform_to_screen = (
            self.canvas.view_transform @ self.get_world_transform()
        )

        # The absolute scale of this matrix tells us how many pixels one
        # local unit of the element occupies on screen.
        scale_x, scale_y = transform_to_screen.get_abs_scale()

        # Calculate the required buffer dimensions in pixels.
        render_width = round(self.width * scale_x)
        render_height = round(self.height * scale_y)

        # Clamp the render dimensions to the maximum allowed size
        render_width = min(render_width, MAX_BUFFER_DIM)
        render_height = min(render_height, MAX_BUFFER_DIM)

        if render_width <= 0 or render_height <= 0:
            # Don't try to render to a zero or negative size surface.
            self.surface = None  # Ensure any old surface is cleared
            if self.canvas:
                self.canvas.queue_draw()
            return False

        # Submit the thread-safe part to the executor with correct pixel dims.
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

    def check_region_hit(
        self,
        x_abs: float,
        y_abs: float,
        candidates: Optional[Set[ElementRegion]] = None,
    ) -> ElementRegion:
        """
        Checks which region is hit at an absolute canvas position.

        It transforms the absolute point into the element's local
        coordinate space to perform the hit check.

        Args:
            x_abs: The absolute x-coordinate on the canvas.
            y_abs: The y-coordinate on the canvas.
            candidates: An optional set of regions to limit the check to.

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

        # This MUST match the calculation in Canvas._render_handles_overlay
        # to ensure the hit-test geometry aligns with the rendered geometry.
        if self.canvas:
            transform_to_screen = self.canvas.view_transform @ world_transform
        else:
            transform_to_screen = world_transform

        scale_compensation = transform_to_screen.get_scale()
        return check_region_hit(
            local_x,
            local_y,
            self.width,
            self.height,
            base_hit_size,
            scale_compensation,
            candidates=candidates,
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

    def _attach_to_canvas_recursive(self, canvas: Optional["Canvas"]):
        """
        Recursively sets the canvas for self and all children, and calls
        the on_attached lifecycle hook.
        """
        self.canvas = canvas
        self.on_attached()
        for child in self.children:
            child._attach_to_canvas_recursive(canvas)

    def _detach_from_canvas_recursive(self):
        """
        Recursively calls the on_detached hook and nullifies the canvas
        reference for self and all children.
        """
        self.on_detached()
        for child in self.children:
            child._detach_from_canvas_recursive()
        self.canvas = None

    def _reparent(self, elem: CanvasElement):
        """Removes an element from its current parent before adding it here."""
        if elem.parent:
            # Check parent type to call the correct removal method.
            if isinstance(elem.parent, CanvasElement):
                elem.parent.remove_child(elem)
            elif elem.canvas and isinstance(
                elem.parent, elem.canvas.__class__
            ):
                elem.parent.remove(elem)

    def add(self, elem: CanvasElement):
        """
        Adds a child element.

        The element is added to the end of the children list. If the
        element already has a parent, it is removed from it first.

        Args:
            elem: The `CanvasElement` to add.
        """
        self._reparent(elem)

        self.children.append(elem)
        elem.parent = self
        # Recursively propagate the canvas reference and trigger the
        # on_attached hook for the new child and its descendants.
        elem._attach_to_canvas_recursive(self.canvas)
        elem.allocate()
        self.mark_dirty()
        self.on_child_list_changed()
        if self.canvas:
            self.canvas.queue_draw()

    def insert(self, index: int, elem: CanvasElement):
        """
        Inserts a child element at a specific index.

        Args:
            index: The index at which to insert the element.
            elem: The `CanvasElement` to insert.
        """
        self._reparent(elem)

        self.children.insert(index, elem)
        elem.parent = self
        # Recursively propagate the canvas reference and trigger the
        # on_attached hook for the new child and its descendants.
        elem._attach_to_canvas_recursive(self.canvas)
        elem.allocate()
        self.mark_dirty()
        self.on_child_list_changed()
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
            yield from child.find_by_type(thetype)

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
        children_to_remove = self.children[:]
        self.children.clear()

        for child in children_to_remove:
            child._detach_from_canvas_recursive()
            if self.canvas:
                self.canvas.elem_removed.send(self, child=child)

        self.mark_dirty()
        self.on_child_list_changed()

    def remove(self):
        """Removes this element from its parent."""
        assert self.parent is not None
        # Check parent type to call the correct removal method.
        if isinstance(self.parent, CanvasElement):
            self.parent.remove_child(self)
        elif self.canvas and isinstance(self.parent, self.canvas.__class__):
            self.parent.remove(self)

    def remove_child(self, elem: CanvasElement):
        """
        Removes a direct child element. This is not recursive.

        Args:
            elem: The child element to remove.
        """
        if elem in self.children:
            # Trigger the detach hook before actual removal.
            elem._detach_from_canvas_recursive()
            self.children.remove(elem)
            if self.canvas:
                self.canvas.elem_removed.send(self, child=elem)
            self.mark_dirty()
            self.on_child_list_changed()

    def get_selected(self) -> Generator[CanvasElement, None, None]:
        """Recursively finds and yields all selected elements."""
        if self.selected:
            yield self
        for child in self.children[:]:
            yield from child.get_selected()

    def get_selected_data(self) -> Generator[Any, None, None]:
        """Recursively finds and yields data of selected elements."""
        for elem in self.get_selected():
            yield elem.data

    def remove_selected(self):
        """Recursively finds and removes all selected elements."""
        for child in self.children[:]:
            if child.selected:
                self.remove_child(child)
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
        Sets the element's position relative to its parent. This method
        is now matrix-native and preserves shear, rotation, and scale.
        """
        new_transform = self.transform.set_translation(x, y)
        self.set_transform(new_transform)

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

        This rebuilds the local transform (as the center point changes),
        re-allocates the backing surface, and triggers a redraw.
        """
        width = float(width)
        height = float(height)
        if width != self.width or height != self.height:
            self.width, self.height = width, height
            # Size change affects the center point, so a full rebuild is
            # necessary
            self._rebuild_transform()
            # Use set_transform to apply the change and notify parent
            self.set_transform(self.transform)

    def rect(self) -> Tuple[float, float, float, float]:
        """
        Gets the local rect (x, y, width, height).
        """
        x, y = self.transform.get_translation()
        return x, y, self.width, self.height

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

        This method applies the element's unified local transformation
        matrix to the context before drawing its content and children.

        Args:
            ctx: The cairo context to draw on.
        """
        if not self.visible:
            return

        ctx.save()

        # Apply the entire local transform relative to the parent in one go.
        cairo_matrix = cairo.Matrix(*self.transform.for_cairo())
        ctx.transform(cairo_matrix)

        # The context is now fully transformed. All subsequent drawing happens
        # in the element's untransformed local space (0,0 at top-left).
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

        The cairo context is assumed to be in the element's geometric
        coordinate space. This method applies the `content_transform` before
        drawing the final content.

        Args:
            ctx: The cairo context, already transformed.
        """
        ctx.save()

        # Apply the content_transform relative to the local geometry.
        cairo_content_matrix = cairo.Matrix(
            *self.content_transform.for_cairo()
        )
        ctx.transform(cairo_content_matrix)

        # --- The rest of the drawing logic is now inside this transform ---
        if not self.buffered or not self.surface:
            # Unbuffered: just draw the background.
            ctx.set_source_rgba(*self.background)
            ctx.rectangle(0, 0, self.width, self.height)
            ctx.fill()
        else:
            source_w = self.surface.get_width()
            source_h = self.surface.get_height()

            if source_w > 0 and source_h > 0:
                # Draw the buffered surface. We need to scale it to fit the
                # element's width and height.
                ctx.save()
                scale_x = self.width / source_w
                scale_y = self.height / source_h
                ctx.scale(scale_x, scale_y)
                ctx.set_source_surface(self.surface, 0, 0)
                ctx.get_source().set_filter(cairo.FILTER_GOOD)
                ctx.paint()
                ctx.restore()

        ctx.restore()

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
        return any(c.has_dirty_children() for c in self.children)

    def get_elem_hit(
        self, world_x: float, world_y: float, selectable: bool = False
    ) -> Optional[CanvasElement]:
        """
        Checks for a hit on this element or its children given world
        coordinates.

        The check is performed recursively, starting with the top-most
        child (which is rendered last). The incoming coordinates are always
        in world space.

        Args:
            world_x: The x-coordinate in the canvas's world space.
            world_y: The y-coordinate in the canvas's world space.
            selectable: If True, only selectable elements are checked.

        Returns:
            The `CanvasElement` that was hit, or `None`.
        """
        # 1. Check children first (top-most are last in list, so drawn on top).
        for child in reversed(self.children):
            # Pass the original world coordinates down recursively.
            hit = child.get_elem_hit(world_x, world_y, selectable)
            if hit:
                # A child was hit, so it's on top of us. Return it immediately.
                return hit

        # 2. If no children were hit, check this element itself.
        if selectable and not self.selectable:
            return None

        # To check ourself, transform the world point into our local
        # geometry space.
        try:
            inv_world = self.get_world_transform().invert()
            local_geom_x, local_geom_y = inv_world.transform_point(
                (world_x, world_y)
            )
        except Exception:
            return (
                None  # Cannot hit an element with a non-invertible transform
            )

        # 3. Now, perform a simple bounding box check in our own geometry
        # space.
        if not (
            0 <= local_geom_x < self.width and 0 <= local_geom_y < self.height
        ):
            return None

        # 4. Optional: If inside the bounding box, perform pixel-perfect check.
        # Draggable elements should be hittable anywhere within their bbox.
        if self.pixel_perfect_hit and not self.draggable:
            if not self.is_pixel_opaque(local_geom_x, local_geom_y):
                return None

        # 5. If all checks pass, we have a hit on this element.
        return self

    def is_pixel_opaque(self, local_x: float, local_y: float) -> bool:
        """
        Checks if the pixel at local geometry coordinates is opaque.

        Args:
            local_x: The x-coordinate in the element's local GEOMETRY space.
            local_y: The y-coordinate in the element's local GEOMETRY space.

        Returns:
            True if the pixel is considered a hit (alpha > 0).
        """
        if not self.buffered:
            # For unbuffered elements, pixel_perfect_hit means the element's
            # body is effectively transparent. The hit is determined solely
            # by a bounding box check in the calling method.
            return False

        if not self.surface:
            # Cannot perform pixel check if the surface doesn't exist.
            return False

        return check_pixel_hit(
            surface=self.surface,
            content_transform=self.content_transform,
            element_width=self.width,
            element_height=self.height,
            hit_distance=self.hit_distance,
            local_x=local_x,
            local_y=local_y,
        )

    def on_edit_mode_enter(self):
        """Called when this element becomes the Canvas's edit_context."""
        pass

    def on_edit_mode_leave(self):
        """Called when this element is no longer the Canvas's edit_context."""
        pass

    def handle_edit_press(self, world_x: float, world_y: float) -> bool:
        """
        Handles a mouse press event while in edit mode.

        Args:
            world_x: The x-coordinate of the press in world space.
            world_y: The y-coordinate of the press in world space.

        Returns:
            True if the event was handled, False otherwise.
        """
        return False

    def handle_edit_drag(self, world_dx: float, world_dy: float):
        """
        Handles a mouse drag event while in edit mode.

        Args:
            world_dx: The horizontal drag distance in world coordinates.
            world_dy: The vertical drag distance in world coordinates.
        """
        pass

    def handle_edit_release(self, world_x: float, world_y: float):
        """
        Handles a mouse release event while in edit mode.

        Args:
            world_x: The x-coordinate of the release in world space.
            world_y: The y-coordinate of the release in world space.
        """
        pass

    def handle_drag_move(
        self, world_dx: float, world_dy: float
    ) -> Tuple[float, float]:
        """
        Intercepts a drag move to apply constraints. Subclasses can
        override this to customize drag behavior.

        This method is only called if the `draggable` property is True.

        If `drag_handler_controls_transform` is True, this method is
        responsible for setting the element's transform directly.

        If it's False, this method should return a constrained delta tuple
        `(constrained_dx, constrained_dy)` in world space.
        """
        return world_dx, world_dy

    def dump(self, indent: int = 0):
        """Prints a debug representation of the element and children."""
        pad = "  " * indent
        print(f"{pad}{self.__class__.__name__}: (Data: {self.data})")
        print(f"{pad}  Visible: {self.visible}, Selected: {self.selected}")
        print(f"{pad}  Rect: {self.rect()}")
        print(f"{pad}  Clip: {self.clip}")
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

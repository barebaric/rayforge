from __future__ import annotations
import math
import logging
from typing import Any, Generator, List, Tuple, Optional, Set, Union
from enum import Enum, auto
import cairo
from gi.repository import Gtk, Gdk, Graphene
from blinker import Signal
from ...core.matrix import Matrix
from .element import CanvasElement
from . import transform
from .region import (
    ElementRegion,
    BBOX_REGIONS,
    RESIZE_HANDLES,
    ROTATE_HANDLES,
    ROTATE_SHEAR_HANDLES,
    SHEAR_HANDLES,
)
from .cursor import get_cursor_for_region
from .multiselect import MultiSelectionGroup
from .overlays import render_selection_handles, render_selection_frame
from .intersect import obb_intersects_aabb


logger = logging.getLogger(__name__)
DRAG_THRESHOLD = 5.0


class SelectionMode(Enum):
    """Defines the interaction mode for the current selection."""

    NONE = auto()  # nothing selected
    RESIZE = auto()
    ROTATE_SHEAR = auto()


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
        self.view_transform: Matrix = Matrix.identity()
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
        self._selection_mode: SelectionMode = SelectionMode.NONE
        self._selection_just_changed: bool = False
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
        self._shearing: bool = False
        self._was_dragging: bool = False
        self._transforming_elements: List[CanvasElement] = []

        # --- Rotation State ---
        self._drag_start_angle: float = 0.0
        self._rotation_pivot: Optional[Tuple[float, float]] = None

        # --- Signals ---
        self.move_begin = Signal()
        self.move_end = Signal()
        self.resize_begin = Signal()
        self.resize_end = Signal()
        self.rotate_begin = Signal()
        self.rotate_end = Signal()
        self.shear_begin = Signal()
        self.shear_end = Signal()

        # Fired after any transform gesture ends.
        self.transform_end = Signal()

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

    def _get_world_coords(
        self, widget_x: float, widget_y: float
    ) -> Tuple[float, float]:
        """
        Converts widget pixel coordinates to canvas world coordinates using
        the active view_transform.
        """
        try:
            return self.view_transform.invert().transform_point(
                (widget_x, widget_y)
            )
        except Exception:
            # Fallback to 1:1 if matrix is non-invertible
            return widget_x, widget_y

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
        self._drag_gesture.set_button(Gdk.BUTTON_PRIMARY)
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

    def do_snapshot(self, snapshot):
        """GTK4 snapshot-based drawing handler."""
        width, height = self.get_width(), self.get_height()
        bounds = Graphene.Rect().init(0, 0, width, height)
        ctx = snapshot.append_cairo(bounds)

        # Render world content first
        # Apply the view transform to render all elements in world space.
        ctx.save()
        cairo_matrix = cairo.Matrix(*self.view_transform.for_cairo())
        ctx.transform(cairo_matrix)
        self.root.render(ctx)
        ctx.restore()

        # After restoring the context, we are now in pure pixel space.
        # All overlays are drawn here so they are not affected by
        # view zoom/pan.
        self._render_overlays(ctx)

    def _render_element_overlays(
        self, ctx: cairo.Context, elem: CanvasElement
    ):
        """Recursively calls the draw_overlay method for all elements."""
        elem.draw_overlay(ctx)
        for child in elem.children:
            self._render_element_overlays(ctx, child)

    def _render_overlays(self, ctx: cairo.Context):
        """Renders all non-content overlays in pixel space."""
        # Draw selection frames and handles on top of everything.
        self._render_selection_overlay(ctx, self.root)

        # Allow elements to draw their own custom overlays (e.g., previews)
        self._render_element_overlays(ctx, self.root)

        # Draw the framing rectangle if we are in framing mode.
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

    def _render_selection_overlay(
        self, ctx: cairo.Context, elem: CanvasElement
    ):
        """
        Recursively orchestrates the drawing of selection overlays in pixel
        space.
        """
        is_multi_select = self._selection_group is not None

        # Draw frame for any selected element. If it's a multi-selection,
        # we draw the frame but no handles.
        if elem.selected:
            self._draw_selection_frame(ctx, elem)
            if not is_multi_select:
                self._render_single_selection_overlay(ctx, elem)

        for child in elem.children:
            self._render_selection_overlay(ctx, child)

        # The group overlay is handled once at the root level.
        if elem is self.root and self._selection_group:
            self._render_multi_selection_overlay(ctx, self._selection_group)

    def _draw_selection_frame(self, ctx: cairo.Context, elem: CanvasElement):
        """Draws the dashed selection frame for any given element."""
        screen_transform = self.view_transform @ elem.get_world_transform()
        render_selection_frame(ctx, elem, screen_transform)

    def _render_single_selection_overlay(
        self, ctx: cairo.Context, elem: CanvasElement
    ):
        """Draws the interactive handles for a single selected element."""
        if self._moving or self._resizing or self._rotating or self._shearing:
            return

        screen_transform = self.view_transform @ elem.get_world_transform()
        render_selection_handles(
            ctx,
            target=elem,
            transform_to_screen=screen_transform,
            mode=self._selection_mode,
            hovered_region=self._hovered_region,
            base_handle_size=self.BASE_HANDLE_SIZE,
            with_labels=False,  # Set to True to debug
        )

    def _render_multi_selection_overlay(
        self, ctx: cairo.Context, group: MultiSelectionGroup
    ):
        """Draws the selection frame and handles for a group."""
        group._calculate_bounding_box()

        # The transform from the group's local AABB space to screen space.
        group_offset_transform = Matrix.translation(group.x, group.y)
        transform_to_screen = self.view_transform @ group_offset_transform

        # Draw the frame for the group.
        render_selection_frame(ctx, group, transform_to_screen)

        # Draw handles if not currently transforming.
        if not (
            self._moving or self._resizing or self._rotating or self._shearing
        ):
            render_selection_handles(
                ctx,
                target=group,
                transform_to_screen=transform_to_screen,
                mode=self._selection_mode,
                hovered_region=self._hovered_region,
                base_handle_size=self.BASE_HANDLE_SIZE,
                with_labels=False,  # Set to True to debug
            )

    def _update_hover_state(self, x: float, y: float) -> bool:
        """
        Updates the hover state based on cursor position.

        This is the single source of truth for the interactive region. It
        checks for hits in a specific order:
        1. Resize/rotation handles on the current selection, respecting the
           current selection mode.
        2. Body of any selectable element under the cursor.

        Args:
            x: The x-coordinate in WORLD space.
            y: The y-coordinate in WORLD space.

        Returns:
            True if the hover state changed and a redraw is needed.
        """
        selected_elems = self.get_selected_elements()
        is_multi_select = len(selected_elems) > 1
        new_hovered_region = ElementRegion.NONE
        new_hovered_elem = None

        # Priority 1: Check for a valid handle hit on the current selection.
        # We build a set of candidate regions based on the current mode.
        handle_candidates: Optional[Set[ElementRegion]] = None
        if self._selection_mode == SelectionMode.RESIZE:
            handle_candidates = RESIZE_HANDLES
        elif self._selection_mode == SelectionMode.ROTATE_SHEAR:
            handle_candidates = ROTATE_SHEAR_HANDLES

        if handle_candidates:
            target: Optional[Union[CanvasElement, MultiSelectionGroup]] = None
            if is_multi_select:
                target = self._selection_group
            elif selected_elems:
                target = selected_elems[0]

            if target:
                # Pass the candidates to the hit-test function. It will only
                # return a valid region from this set, or NONE.
                region = target.check_region_hit(
                    x, y, candidates=handle_candidates
                )

                if region != ElementRegion.NONE:
                    new_hovered_region = region
                    if isinstance(target, CanvasElement):
                        new_hovered_elem = target

        # Priority 2: If no valid handles were hit, find the element body.
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
        new_group_hovered = False
        if self._selection_group:
            # Check for body or any handle to set the general group hover flag
            all_group_regions = (
                RESIZE_HANDLES | ROTATE_SHEAR_HANDLES | {ElementRegion.BODY}
            )
            if (
                self._selection_group.check_region_hit(
                    x, y, candidates=all_group_regions
                )
                != ElementRegion.NONE
            ):
                new_group_hovered = True

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
        self._was_dragging = False
        world_x, world_y = self._get_world_coords(x, y)
        self._update_hover_state(world_x, world_y)

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

        # This flag may be used in on_button_release to toggling the
        # selection mode  .
        self._selection_just_changed = selection_changed

        if self._framing_selection:
            if selection_changed:
                self._finalize_selection_state()
            self.queue_draw()
            return

        # If the selection changed, finalize it so the transform logic
        # below has the correct state (e.g., active_elem) to work with.
        if selection_changed:
            self._finalize_selection_state()

        # --- Prepare for Transform ---
        target = self._selection_group or self._active_elem
        if not target:
            self.queue_draw()
            return

        # Special case: rotation needs to be prepared on press.
        if self._active_region in ROTATE_HANDLES:
            self._start_rotation(target, world_x, world_y)

        # Store initial state for the transform. The action itself
        # (_moving=True, etc.) will be initiated in on_mouse_drag.
        if isinstance(target, MultiSelectionGroup):
            self._active_origin = target._bounding_box
            target.store_initial_states()
        elif isinstance(target, CanvasElement):
            self._initial_transform = target.transform.copy()
            self._initial_world_transform = target.get_world_transform().copy()
            tx, ty = target.transform.get_translation()
            self._active_origin = (tx, ty, target.width, target.height)

        self.queue_draw()

    def on_motion(self, gesture, x: float, y: float):
        """
        Handles mouse movement, updating hover state and cursor icon.
        This is the single source of truth for cursor updates.
        """
        # Store raw pixel coordinates for selection frame rendering
        self._last_mouse_x = x
        self._last_mouse_y = y

        is_dragging = (
            self._moving or self._resizing or self._rotating or self._shearing
        )

        # Only update hover state when not dragging.
        if not is_dragging:
            world_x, world_y = self._get_world_coords(x, y)
            if self._update_hover_state(world_x, world_y):
                self.queue_draw()

        # Determine the relevant region: the one being dragged or the one
        # hovered.
        current_region = (
            self._active_region if is_dragging else self._hovered_region
        )

        # Determine the final visual rotation angle for the cursor.
        cursor_angle = 0.0
        selected_elems = self.get_selected_elements()
        cursor_angle = 0.0
        use_absolute_angle = False

        # For all handles (resize, rotate, shear), the cursor angle depends on
        # the total visual rotation of the selection. This is the correct
        # logic.
        if self._selection_group:
            # Group is axis-aligned in world, so only view transform matters.
            cursor_angle = self.view_transform.get_rotation()
        elif selected_elems:
            # For a single element, combine its world transform with the view.
            elem = selected_elems[0]
            transform_to_screen = (
                self.view_transform @ elem.get_world_transform()
            )

            if current_region in SHEAR_HANDLES:
                # For shear, the cursor must align with the visual edge.
                # We calculate the edge's absolute angle and tell the cursor
                # logic not to add its own base angle.
                use_absolute_angle = True
                if current_region in (
                    ElementRegion.SHEAR_TOP,
                    ElementRegion.SHEAR_BOTTOM,
                ):
                    cursor_angle = transform_to_screen.get_x_axis_angle()
                else:  # SHEAR_LEFT, SHEAR_RIGHT
                    cursor_angle = transform_to_screen.get_y_axis_angle()
            else:
                # For resize/rotate, the cursor angle is the element's
                # overall rotation.
                cursor_angle = transform_to_screen.get_rotation()

        cursor = get_cursor_for_region(
            current_region, cursor_angle, absolute=use_absolute_angle
        )
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

    def _update_framing_selection_from_drag(
        self, offset_x: float, offset_y: float
    ):
        """Helper to update the rubber-band selection frame during a drag."""
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

    def on_mouse_drag(self, gesture, offset_x: float, offset_y: float):
        """
        Handles an active drag, dispatching to transform-specific methods.
        It now includes a threshold to distinguish between a click and a
        true drag.
        """
        if self._framing_selection:
            self._update_framing_selection_from_drag(offset_x, offset_y)
            return

        is_transforming = (
            self._moving or self._resizing or self._rotating or self._shearing
        )

        # If a transform hasn't started yet, check if we've passed the drag
        # threshold to prevent tiny movements from hiding handles on a click.
        if not is_transforming:
            dist_sq = offset_x**2 + offset_y**2
            if dist_sq < (DRAG_THRESHOLD**2):
                return  # Not a real drag yet, ignore.

            # Now that the drag is confirmed, set the state and fire signals.
            self._was_dragging = True
            selected_elements = self.get_selected_elements()
            if not selected_elements:
                return

            if self._active_region == ElementRegion.BODY:
                self._moving = True
                self.move_begin.send(self, elements=selected_elements)
            elif self._active_region in ROTATE_HANDLES:
                self._rotating = True
                self.rotate_begin.send(self, elements=selected_elements)
            elif self._active_region in SHEAR_HANDLES:
                self._shearing = True
                self.shear_begin.send(self, elements=selected_elements)
            elif self._active_region != ElementRegion.NONE:
                self._resizing = True
                self.resize_begin.send(self, elements=selected_elements)

            # Set a generic "interactive" flag on the elements being
            # transformed. This allows complex parents (like ShrinkWrapGroup)
            # to react appropriately without the Canvas needing to know
            # about them.
            self._transforming_elements = selected_elements
            for elem in self._transforming_elements:
                elem.begin_interactive_transform()

        # If we reach here, the drag is confirmed and active.
        # Calculate drag delta in WORLD coordinates
        ok, start_x, start_y = self._drag_gesture.get_start_point()
        if not ok:
            return
        current_x, current_y = start_x + offset_x, start_y + offset_y
        start_world_x, start_world_y = self._get_world_coords(start_x, start_y)
        current_world_x, current_world_y = self._get_world_coords(
            current_x, current_y
        )
        world_dx = current_world_x - start_world_x
        world_dy = current_world_y - start_world_y

        if self._selection_group:
            if self._moving:
                self._selection_group.apply_move(world_dx, world_dy)
            elif self._resizing:
                if self._active_origin:
                    self._selection_group.resize_from_drag(
                        self._active_region,
                        world_dx,
                        world_dy,
                        self._active_origin,
                        self._ctrl_pressed,
                        self._shift_pressed,
                    )
                for elem in self._selection_group.elements:
                    elem.trigger_update()
            elif self._rotating:
                if self._rotation_pivot:
                    self._selection_group.rotate_from_drag(
                        current_world_x,
                        current_world_y,
                        self._rotation_pivot,
                        self._drag_start_angle,
                    )
            elif self._shearing:
                if self._active_origin:
                    self._selection_group.shear_from_drag(
                        self._active_region,
                        world_dx,
                        world_dy,
                        self._active_origin,
                    )
            self.queue_draw()
        elif self._active_elem:
            if self._moving:
                if self._active_elem and self._initial_world_transform:
                    transform.move_element(
                        self._active_elem,
                        world_dx,
                        world_dy,
                        self._initial_world_transform,
                    )
            elif self._resizing:
                if (
                    self._active_elem
                    and self._initial_transform
                    and self._initial_world_transform
                ):
                    transform.resize_element(
                        element=self._active_elem,
                        world_dx=world_dx,
                        world_dy=world_dy,
                        initial_local_transform=self._initial_transform,
                        initial_world_transform=self._initial_world_transform,
                        active_region=self._active_region,
                        view_transform=self.view_transform,
                        shift_pressed=self._shift_pressed,
                        ctrl_pressed=self._ctrl_pressed,
                    )
                    self._active_elem.trigger_update()
            elif self._rotating:
                if (
                    self._active_elem
                    and self._initial_world_transform
                    and self._rotation_pivot
                ):
                    transform.rotate_element(
                        element=self._active_elem,
                        world_x=current_world_x,
                        world_y=current_world_y,
                        initial_world_transform=self._initial_world_transform,
                        rotation_pivot=self._rotation_pivot,
                        drag_start_angle=self._drag_start_angle,
                    )
            elif self._shearing:
                if (
                    self._active_elem
                    and self._initial_transform
                    and self._initial_world_transform
                ):
                    transform.shear_element(
                        element=self._active_elem,
                        world_dx=world_dx,
                        world_dy=world_dy,
                        initial_local_transform=self._initial_transform,
                        initial_world_transform=self._initial_world_transform,
                        active_region=self._active_region,
                        view_transform=self.view_transform,
                    )
            self.queue_draw()

    def _start_rotation(
        self,
        target: Union[CanvasElement, MultiSelectionGroup],
        x: float,
        y: float,
    ):
        """
        Stores the initial state for a rotation operation.
        The x and y coordinates are in WORLD space.
        """
        is_group = isinstance(target, MultiSelectionGroup)
        center_x, center_y = (
            target.center if is_group else target.get_world_center()
        )

        # The pivot is always the center of the selection.
        self._rotation_pivot = (center_x, center_y)

        self._drag_start_angle = math.degrees(
            math.atan2(
                y - self._rotation_pivot[1], x - self._rotation_pivot[0]
            )
        )

    def on_button_release(self, gesture, x: float, y: float):
        """
        Handles the end of a drag operation, finalizing transforms and
        emitting the generic `transform_end` signal for subclasses to handle
        model updates.
        """
        if self._framing_selection:
            self._selection_frame_rect = None
            self._selection_before_framing.clear()
            self._finalize_selection_state()
            return

        is_transforming = (
            self._moving or self._resizing or self._rotating or self._shearing
        )
        if not is_transforming:
            return

        elements = self.get_selected_elements()

        # Fire specific signals for detailed event handling
        if self._moving:
            self.move_end.send(self, elements=elements)
        elif self._resizing:
            self.resize_end.send(self, elements=elements)
        elif self._rotating:
            self.rotate_end.send(self, elements=elements)
        elif self._shearing:
            self.shear_end.send(self, elements=elements)

        # Fire the single, generic signal for model synchronization.
        # Subclasses connect to THIS signal to avoid polluting the
        # generic canvas with application-specific logic.
        self.transform_end.send(self, elements=elements)

        # Recalculate group bounding box if it was being transformed
        if self._selection_group:
            self._selection_group._calculate_bounding_box()
            self._active_origin = self._selection_group._bounding_box

        # Notify elements that the interaction is over. This allows parent
        # groups to perform a final state consolidation.
        for elem in self._transforming_elements:
            elem.end_interactive_transform()
        self._transforming_elements.clear()

        # Reset all interaction state variables
        self._resizing, self._moving, self._rotating, self._shearing = (
            False,
            False,
            False,
            False,
        )
        self._active_region = ElementRegion.NONE
        self._initial_transform = None
        self._initial_world_transform = None
        self._rotation_pivot = None

        self.queue_draw()

    def on_click_released(self, gesture, n_press: int, x: float, y: float):
        """
        Handles the completion of a click that did not become a drag.
        This is where the selection mode is toggled.
        """
        if self._framing_selection:
            self._framing_selection = False
            self._selection_frame_rect = None
            self._selection_before_framing.clear()
            # A framing operation (even a zero-pixel one) should finalize
            # the selection but never toggle the mode.
            self._finalize_selection_state()
            self._selection_just_changed = False
            return

        if self._was_dragging:
            self._was_dragging = False
            self._selection_just_changed = False
            return

        # Check for mode switch on simple click
        world_x, world_y = self._get_world_coords(x, y)
        hit = self.root.get_elem_hit(world_x, world_y, selectable=True)
        hover_region = ElementRegion.NONE
        if hit and hit.selected:
            # Re-check region hit to be sure, testing against ALL handles
            # to allow mode switching from resize handles to rotate handles,
            # etc.
            target = self._selection_group if self._selection_group else hit
            hover_region = target.check_region_hit(world_x, world_y)

        # ONLY toggle mode if selection did NOT just change in this click
        if hit and hit.selected and not self._selection_just_changed:
            new_mode = self._selection_mode
            if (
                self._selection_mode != SelectionMode.RESIZE
                and hover_region in BBOX_REGIONS
            ):
                new_mode = SelectionMode.RESIZE
            elif (
                self._selection_mode != SelectionMode.ROTATE_SHEAR
                and hover_region == ElementRegion.BODY
            ):
                new_mode = SelectionMode.ROTATE_SHEAR

            if new_mode != self._selection_mode:
                self._selection_mode = new_mode
                self.queue_draw()

        # Reset the flag after the full click action is complete
        self._selection_just_changed = False

    def _sync_selection_state(self):
        """
        Synchronizes the internal selection state (active element, multi-
        selection group) with the current `selected` flags on elements. It
        fires signals but does NOT change the interaction mode.
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

    def _finalize_selection_state(self):
        """
        Fully updates the selection state after a user interaction, including
        resetting the interaction mode to its default.
        """
        self._sync_selection_state()

        selected = self.get_selected_elements()
        self._selection_mode = SelectionMode.NONE

        if len(selected) > 0:
            self._selection_mode = SelectionMode.RESIZE

    def _get_element_world_corners(
        self, elem: CanvasElement
    ) -> List[Tuple[float, float]]:
        """
        Calculates the four corners of an element in world coordinates.
        """
        world_transform = elem.get_world_transform()
        w, h = elem.width, elem.height
        local_corners = [(0, 0), (w, 0), (w, h), (0, h)]
        return [world_transform.transform_point(p) for p in local_corners]

    def _update_framing_selection(self):
        """
        Updates element selection based on the rubber-band frame.
        """
        if not self._selection_frame_rect:
            return

        frame_x, frame_y, frame_w, frame_h = self._selection_frame_rect
        world_tl = self._get_world_coords(frame_x, frame_y)
        world_br = self._get_world_coords(frame_x + frame_w, frame_y + frame_h)
        world_frame_x = min(world_tl[0], world_br[0])
        world_frame_y = min(world_tl[1], world_br[1])
        world_frame_w = abs(world_br[0] - world_tl[0])
        world_frame_h = abs(world_br[1] - world_tl[1])

        # Avoid selection changes from a simple click (zero-area frame).
        if world_frame_w < 2 and world_frame_h < 2:
            return

        selection_rect = Graphene.Rect().init(
            world_frame_x, world_frame_y, world_frame_w, world_frame_h
        )
        selection_changed = False

        for elem in self.root.get_all_children_recursive():
            if elem.selectable:
                elem_corners = self._get_element_world_corners(elem)
                intersects = obb_intersects_aabb(elem_corners, selection_rect)

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

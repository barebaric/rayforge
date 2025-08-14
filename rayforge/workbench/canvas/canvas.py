from __future__ import annotations
import math
import logging
from typing import Any, Generator, List, Tuple, Optional, Set, Union
import cairo
from gi.repository import Gtk, Gdk, Graphene  # type: ignore
from blinker import Signal
from .element import CanvasElement
from .region import ElementRegion
from .cursor import get_cursor_for_region
from .selection import MultiSelectionGroup
from ...core.matrix import Matrix

logger = logging.getLogger(__name__)


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

        # Render world content first
        # Apply the view transform to render all elements in world space.
        ctx.save()
        m = self.view_transform.m
        cairo_matrix = cairo.Matrix(
            m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2]
        )
        ctx.transform(cairo_matrix)
        self.root.render(ctx)
        ctx.restore()

        # After restoring the context, we are now in pure pixel space.
        # All overlays are drawn here so they are not affected by
        # view zoom/pan.
        self._render_overlays(ctx)

    def _render_overlays(self, ctx: cairo.Context):
        """Renders all non-content overlays in pixel space."""
        # Draw selection frames and handles on top of everything.
        self._render_selection_overlay(ctx, self.root)

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

        if elem.selected and not is_multi_select:
            self._render_single_selection_overlay(ctx, elem)

        for child in elem.children:
            self._render_selection_overlay(ctx, child)

        if elem is self.root and self._selection_group:
            self._render_multi_selection_overlay(ctx, self._selection_group)

    def _render_single_selection_overlay(
        self, ctx: cairo.Context, elem: CanvasElement
    ):
        """Draws the selection frame for a single element in pixel space."""
        ctx.save()

        # Get the matrix that transforms from the element's local space
        # directly to screen (pixel) space.
        screen_transform = self.view_transform @ elem.get_world_transform()

        # Transform the four local corners of the element to screen space.
        w, h = elem.width, elem.height
        corners_local = [(0, 0), (w, 0), (w, h), (0, h)]
        corners_screen = [
            screen_transform.transform_point(p) for p in corners_local
        ]

        # Draw the dashed outline connecting the screen-space corners.
        # Line width and dash pattern are now in fixed pixels.
        ctx.set_source_rgb(0.4, 0.4, 0.4)
        ctx.set_line_width(1.0)
        ctx.set_dash((5, 5))

        ctx.move_to(*corners_screen[0])
        ctx.line_to(*corners_screen[1])
        ctx.line_to(*corners_screen[2])
        ctx.line_to(*corners_screen[3])
        ctx.close_path()
        ctx.stroke()
        ctx.set_dash([])

        # Draw handles if not currently transforming.
        if not (self._moving or self._resizing or self._rotating):
            self._render_handles_overlay(ctx, elem, screen_transform)

        ctx.restore()

    def _render_multi_selection_overlay(
        self, ctx: cairo.Context, group: MultiSelectionGroup
    ):
        """Draws the selection frame for a group in pixel space."""
        ctx.save()
        group._calculate_bounding_box()

        # Get the group's axis-aligned bounding box in world space.
        x, y, w, h = group.x, group.y, group.width, group.height
        corners_world = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        # Transform the world-space corners to screen space.
        corners_screen = [
            self.view_transform.transform_point(p) for p in corners_world
        ]

        # Draw the dashed outline in fixed pixels.
        ctx.set_source_rgb(0.4, 0.4, 0.4)
        ctx.set_line_width(1.0)
        ctx.set_dash((5, 5))

        ctx.move_to(*corners_screen[0])
        ctx.line_to(*corners_screen[1])
        ctx.line_to(*corners_screen[2])
        ctx.line_to(*corners_screen[3])
        ctx.close_path()
        ctx.stroke()
        ctx.set_dash([])

        # Draw handles if not currently transforming.
        if not (self._moving or self._resizing or self._rotating):
            # Define a transform that maps the group's "local" space
            # (a w x h box at origin 0,0) to the screen. This is done
            # by first translating to the group's world position, and
            # then applying the main view transform.
            group_offset_transform = Matrix.translation(group.x, group.y)
            transform_to_screen = self.view_transform @ group_offset_transform
            self._render_handles_overlay(ctx, group, transform_to_screen)

        ctx.restore()

    def _render_handles_overlay(
        self,
        ctx: cairo.Context,
        target: Union[CanvasElement, MultiSelectionGroup],
        transform_to_screen: Matrix,
    ):
        """
        The definitive helper to draw handles. It uses get_region_rect to
        determine handle geometry in local space and then draws the
        resulting polygon in the pixel-space overlay.
        """
        # --- Setup ---
        world_x, world_y = self._get_world_coords(
            self._last_mouse_x, self._last_mouse_y
        )
        is_group = isinstance(target, MultiSelectionGroup)
        is_hovered = (
            self._group_hovered
            if is_group
            else target.check_region_hit(world_x, world_y)
            != ElementRegion.NONE
        )

        # Get scale factors and flip status from the matrix.
        sx_abs, sy_abs = transform_to_screen.get_abs_scale()
        is_view_flipped = transform_to_screen.is_flipped()

        # Check for zero scale using absolute values to prevent incorrect exit.
        if sx_abs < 1e-6 or sy_abs < 1e-6:
            # This is a valid exit, e.g., if the element is scaled to nothing.
            return

        # The handle rendering logic expects the scale compensation factor for
        # the y-axis to be negative if the view is flipped.
        scale_compensation = (sx_abs, -sy_abs if is_view_flipped else sy_abs)
        ctx.set_source_rgba(0.2, 0.5, 0.8, 0.7)

        # --- Drawing Helper ---
        def draw_local_rect_as_overlay(
            rect_local: Tuple[float, float, float, float],
        ):
            lx, ly, lw, lh = rect_local
            if lw <= 0 or lh <= 0:
                return

            local_corners = [
                (lx, ly),
                (lx + lw, ly),
                (lx + lw, ly + lh),
                (lx, ly + lh),
            ]
            screen_corners = [
                transform_to_screen.transform_point(p) for p in local_corners
            ]
            ctx.move_to(*screen_corners[0])
            for i in range(1, 4):
                ctx.line_to(*screen_corners[i])
            ctx.close_path()
            ctx.fill()

        # --- Render Handles and Rotation Line ---
        if is_hovered:
            regions_to_draw = [
                ElementRegion.ROTATION_HANDLE,
                ElementRegion.TOP_LEFT,
                ElementRegion.TOP_RIGHT,
                ElementRegion.BOTTOM_LEFT,
                ElementRegion.BOTTOM_RIGHT,
            ]
            for region in regions_to_draw:
                handle_rect = target.get_region_rect(
                    region, self.BASE_HANDLE_SIZE, scale_compensation
                )
                draw_local_rect_as_overlay(handle_rect)

            # Draw rotation line
            top_middle_local_y = target.height if is_view_flipped else 0.0
            top_middle_local = (target.width / 2, top_middle_local_y)
            rot_handle_rect = target.get_region_rect(
                ElementRegion.ROTATION_HANDLE,
                self.BASE_HANDLE_SIZE,
                scale_compensation,
            )
            # Anchor to the middle of the handle edge closest to the element.
            rot_anchor_x = rot_handle_rect[0] + rot_handle_rect[2] / 2
            rot_anchor_y = (
                rot_handle_rect[1]
                if is_view_flipped
                else (rot_handle_rect[1] + rot_handle_rect[3])
            )
            rot_anchor_local = (rot_anchor_x, rot_anchor_y)

            p1_screen = transform_to_screen.transform_point(top_middle_local)
            p2_screen = transform_to_screen.transform_point(rot_anchor_local)

            ctx.save()
            ctx.set_source_rgba(0.4, 0.4, 0.4, 0.9)
            ctx.set_line_width(1.0)
            ctx.move_to(*p1_screen)
            ctx.line_to(*p2_screen)
            ctx.stroke()
            ctx.restore()

        edge_regions = [
            ElementRegion.TOP_MIDDLE,
            ElementRegion.BOTTOM_MIDDLE,
            ElementRegion.MIDDLE_LEFT,
            ElementRegion.MIDDLE_RIGHT,
        ]
        if self._hovered_region in edge_regions:
            handle_rect = target.get_region_rect(
                self._hovered_region, self.BASE_HANDLE_SIZE, scale_compensation
            )
            draw_local_rect_as_overlay(handle_rect)

    def _update_hover_state(self, x: float, y: float) -> bool:
        """
        Updates the hover state based on cursor position.

        This is the single source of truth for the interactive region. It
        checks for hits in a specific order:
        1. Resize/rotation handles on the current selection.
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
            self._start_rotation(target, world_x, world_y)
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
        # Store raw pixel coordinates for selection frame rendering
        self._last_mouse_x = x
        self._last_mouse_y = y

        world_x, world_y = self._get_world_coords(x, y)
        if not (self._moving or self._resizing or self._rotating):
            if self._update_hover_state(world_x, world_y):
                self.queue_draw()

        if self._moving:
            self.set_cursor(Gdk.Cursor.new_from_name("move"))
            return

        # Determine the final visual rotation angle for the cursor.
        cursor_angle = 0.0
        selected_elems = self.get_selected_elements()
        if self._selection_group:
            # For a group, the visual rotation is determined solely by the
            # view_transform.
            total_transform = self.view_transform
            cursor_angle = total_transform.get_rotation()
        elif selected_elems:
            # For an element, the final visual rotation is the composition of
            # its own transform and the view_transform.
            total_transform = (
                self.view_transform @ selected_elems[0].get_world_transform()
            )
            cursor_angle = total_transform.get_rotation()

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
                self._apply_group_resize(world_dx, world_dy)
                for elem in self._selection_group.elements:
                    elem.trigger_update()
            elif self._rotating:
                self._rotate_selection_group(current_world_x, current_world_y)
            self.queue_draw()
        elif self._active_elem:
            if self._moving:
                self._move_active_element(world_dx, world_dy)
            elif self._resizing:
                self._resize_active_element(world_dx, world_dy)
                self._active_elem.trigger_update()
            elif self._rotating:
                self._rotate_active_element(current_world_x, current_world_y)
            self.queue_draw()

    def _apply_group_resize(self, offset_x: float, offset_y: float):
        """
        Calculates new group bounding box based on the drag offset.
        The offset is now in WORLD coordinates.
        """
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

        # Determine if the view's Y-axis is flipped.
        is_view_flipped = self.view_transform.is_flipped()

        if self._ctrl_pressed:  # Center-out resize
            dw, dh = 0.0, 0.0
            if is_left:
                dw = -offset_x
            elif is_right:
                dw = offset_x

            # The change in height (dh) depends on the view orientation.
            if is_view_flipped:
                if is_top:
                    dh = offset_y  # Drag down (visual) = offset_y < 0
                elif is_bottom:
                    dh = -offset_y  # Drag down (visual) = offset_y < 0
            else:  # Normal Y-down
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
            # X-axis logic is independent of Y-flip.
            if is_left:
                new_x, new_w = orig_x + offset_x, orig_w - offset_x
            elif is_right:
                new_w = orig_w + offset_x

            # Y-axis logic must account for the flipped coordinate system.
            if is_view_flipped:
                # In a Y-up world, orig_y is the bottom.
                if is_top:  # Dragging top edge; anchor is bottom (orig_y).
                    new_h = orig_h + offset_y
                elif is_bottom:  # Dragging bottom; anchor is top.
                    new_y = orig_y + offset_y
                    new_h = orig_h - offset_y
            else:  # Normal Y-down world.
                if is_top:  # Dragging top edge; anchor is bottom.
                    new_y, new_h = orig_y + offset_y, orig_h - offset_y
                elif is_bottom:  # Dragging bottom edge; anchor is top.
                    new_h = orig_h + offset_y

            if self._shift_pressed and orig_w > 0 and orig_h > 0:
                aspect = orig_w / orig_h
                dw, dh = new_w - orig_w, new_h - orig_h
                is_corner = (is_left or is_right) and (is_top or is_bottom)

                # Recalculate one dimension based on the other to maintain
                # aspect ratio
                if (is_corner and abs(dw) > abs(dh) * aspect) or (
                    not is_corner and (is_left or is_right)
                ):
                    new_h = new_w / aspect
                else:
                    new_w = new_h * aspect

                # After constraining dimensions, the origin (new_x, new_y)
                # must be recalculated to keep the anchor point (the opposite
                # side or center) fixed.

                # Horizontal Anchoring
                if is_left:
                    new_x = (orig_x + orig_w) - new_w
                elif not (is_right or is_left):  # Top/Bottom middle handle
                    new_x = orig_x + (orig_w - new_w) / 2
                # If is_right, new_x remains orig_x, which is correct.

                # Vertical Anchoring
                if is_top:
                    if not is_view_flipped:  # Y-down, anchor is bottom edge.
                        new_y = (orig_y + orig_h) - new_h
                elif is_bottom:
                    if is_view_flipped:  # Y-up, anchor is top edge.
                        new_y = (orig_y + orig_h) - new_h
                else:  # Left/Right middle handle
                    new_y = orig_y + (orig_h - new_h) / 2

        # Capture the calculated size before applying the minimum constraint.
        unclamped_w, unclamped_h = new_w, new_h

        new_w, new_h = max(new_w, min_size), max(new_h, min_size)

        # If clamping occurred, the origin (new_x, new_y) must be adjusted
        # to keep the anchor point (the opposite side or center) fixed.
        if self._ctrl_pressed:
            # For center-out resize, the origin shifts by half the clamped
            # amount.
            new_x += (unclamped_w - new_w) / 2
            new_y += (unclamped_h - new_h) / 2
        else:
            # For anchor-based resize, the origin shifts if it's not the
            # anchor.
            if is_left:
                new_x += unclamped_w - new_w

            if is_top and not is_view_flipped:  # Y-down, anchor is bottom.
                new_y += unclamped_h - new_h
            elif is_bottom and is_view_flipped:  # Y-up, anchor is top.
                new_y += unclamped_h - new_h

        new_box = (new_x, new_y, new_w, new_h)
        self._selection_group.apply_resize(new_box, self._active_origin)

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
        self._drag_start_angle = math.degrees(
            math.atan2(y - center_y, x - center_x)
        )

    def _rotate_active_element(self, current_x: float, current_y: float):
        """
        Rotates a single element using a matrix-native approach to preserve
        all existing transformations, including shear.
        The current_x, current_y are in WORLD coordinates.
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

    def _rotate_selection_group(self, current_x: float, current_y: float):
        """
        Rotates the entire selection group based on cursor drag.
        The coordinates are in WORLD space.
        """
        if not self._selection_group:
            return

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
        The offset_x, offset_y are now deltas in WORLD coordinates.
        """
        if (
            not self._active_elem
            or not self._initial_transform
            or not self._initial_world_transform
        ):
            return

        min_size_world = 2.0
        base_w, base_h = self._active_elem.width, self._active_elem.height

        # Determine the effective scale from the element's local geometry
        # to world space. Use absolute scale for calculating minimum sizes.
        world_scale_x, world_scale_y = (
            self._initial_world_transform.get_abs_scale()
        )
        min_size_local_x = (
            min_size_world / world_scale_x if world_scale_x > 1e-6 else 0
        )
        min_size_local_y = (
            min_size_world / world_scale_y if world_scale_y > 1e-6 else 0
        )

        # 1. Determine which VISUAL (semantic) handles are being dragged.
        semantic_is_left = self._active_region in {
            ElementRegion.TOP_LEFT,
            ElementRegion.MIDDLE_LEFT,
            ElementRegion.BOTTOM_LEFT,
        }
        semantic_is_right = self._active_region in {
            ElementRegion.TOP_RIGHT,
            ElementRegion.MIDDLE_RIGHT,
            ElementRegion.BOTTOM_RIGHT,
        }
        semantic_is_top = self._active_region in {
            ElementRegion.TOP_LEFT,
            ElementRegion.TOP_MIDDLE,
            ElementRegion.TOP_RIGHT,
        }
        semantic_is_bottom = self._active_region in {
            ElementRegion.BOTTOM_LEFT,
            ElementRegion.BOTTOM_MIDDLE,
            ElementRegion.BOTTOM_RIGHT,
        }

        # 2. Bridge the semantic-geometric gap.
        # The geometric edge corresponding to a visual handle depends on
        # the view. We check if the view's coordinate system is flipped.
        is_view_flipped = self.view_transform.is_flipped()

        if is_view_flipped:
            # In a flipped view, the visual 'top' handle is at the geometric
            # 'bottom' (y=h) of the element, and vice-versa.
            is_top = semantic_is_bottom
            is_bottom = semantic_is_top
        else:
            # In a normal view, visual and geometric handles align.
            is_top = semantic_is_top
            is_bottom = semantic_is_bottom

        # X-axis is not affected by the Y-flip.
        is_left = semantic_is_left
        is_right = semantic_is_right

        # 3. Transform the world-space drag offset into the element's
        #    initial local coordinate system (unrotated frame of reference).
        initial_world_no_trans = self._initial_world_transform.copy()
        initial_world_no_trans.m[0, 2] = initial_world_no_trans.m[1, 2] = 0.0
        inv_rot_scale = initial_world_no_trans.invert()
        local_delta_x, local_delta_y = inv_rot_scale.transform_vector(
            (offset_x, offset_y)
        )

        # 4. Calculate desired change in size (dw, dh) in local space.
        # This logic is now purely geometric and correct because 'is_top' and
        # 'is_bottom' refer to the correct geometric edges.
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

        # 5. Handle aspect ratio constraint (Shift key).
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

        # 6. Calculate new size and the scale factors to apply.
        new_w = max(min_size_local_x, base_w + dw)
        new_h = max(min_size_local_y, base_h + dh)
        scale_x = new_w / base_w if base_w > 0 else 1.0
        scale_y = new_h / base_h if base_h > 0 else 1.0

        # 7. Define the fixed anchor point in the element's LOCAL GEOMETRY.
        # This logic is also geometric and is now correct.
        anchor_norm_x = (
            0.5 if not (is_left or is_right) else (1.0 if is_left else 0.0)
        )
        anchor_norm_y = (
            0.5 if not (is_top or is_bottom) else (1.0 if is_top else 0.0)
        )
        if self._ctrl_pressed:
            anchor_norm_x, anchor_norm_y = 0.5, 0.5

        anchor_local_geom = (anchor_norm_x * base_w, anchor_norm_y * base_h)

        # 8. Construct the delta transform: a scale around the LOCAL anchor.
        t_to_origin = Matrix.translation(
            -anchor_local_geom[0], -anchor_local_geom[1]
        )
        m_scale = Matrix.scale(scale_x, scale_y)
        t_from_origin = Matrix.translation(
            anchor_local_geom[0], anchor_local_geom[1]
        )
        delta_transform_local = t_from_origin @ m_scale @ t_to_origin

        # 9. Apply this local delta to the element's initial local transform.
        new_local_transform = self._initial_transform @ delta_transform_local

        # 10. Apply the new, complete transform matrix to the element.
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

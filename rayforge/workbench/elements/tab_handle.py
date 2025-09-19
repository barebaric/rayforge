import cairo
import math
import logging
from typing import TYPE_CHECKING, cast, Optional, Tuple, List
from gi.repository import Gdk
from copy import deepcopy

from ..canvas.element import CanvasElement
from ...core.tab import Tab
from ...undo import ChangePropertyCommand
from ...core.matrix import Matrix

if TYPE_CHECKING:
    from .workpiece import WorkPieceView
    from ..surface import WorkSurface

logger = logging.getLogger(__name__)


class TabHandleElement(CanvasElement):
    """
    A canvas element representing a single Tab, which is always visible
    and can be dragged along its parent's vector path.
    """

    def __init__(self, tab_data: Tab, parent: "WorkPieceView"):
        super().__init__(
            x=0,
            y=0,
            width=1.0,  # A unit square, scaled by the transform
            height=1.0,
            data=tab_data,
            parent=parent,
            selectable=True,
            draggable=True,
            show_selection_frame=False,
            drag_handler_controls_transform=True,
            preserves_selection_on_click=True,  # This is the key flag
            clip=False,
        )
        self._initial_tabs_state: Optional[List[Tab]] = None
        # Cache for geometric calculations, in parent's normalized (0-1) space.
        self._local_pos_norm: Tuple[float, float] = (0.0, 0.0)
        self._local_tangent_norm: Tuple[float, float] = (1.0, 0.0)

    def on_attached(self):
        """Lifecycle hook called when added to the canvas."""
        assert self.canvas
        self.canvas.move_begin.connect(self._on_drag_begin)
        self.canvas.move_end.connect(self._on_drag_end)

    def on_detached(self):
        """Lifecycle hook called before being removed from the canvas."""
        assert self.canvas
        self.canvas.move_begin.disconnect(self._on_drag_begin)
        self.canvas.move_end.disconnect(self._on_drag_end)

    def _on_drag_begin(
        self,
        sender,
        elements: List[CanvasElement],
        drag_target: Optional[CanvasElement] = None,
    ):
        """Called by the canvas when a move operation starts."""
        # The drag_target is the specific element being manipulated, which
        # is exactly what we need. This is more robust than checking the
        # general selection ('elements' list).
        if drag_target is self:
            parent_view = cast("WorkPieceView", self.parent)
            # Store a deepcopy of the entire tabs list for the undo command
            self._initial_tabs_state = deepcopy(parent_view.data.tabs)
            logger.debug(f"Drag begin for tab {self.data.uid}")

    def _on_drag_end(
        self,
        sender,
        elements: List[CanvasElement],
        drag_target: Optional[CanvasElement] = None,
    ):
        """Called by the canvas when a move operation ends."""
        # Check against the explicit drag_target.
        if drag_target is self and self._initial_tabs_state is not None:
            parent_view = cast("WorkPieceView", self.parent)
            work_surface = cast("WorkSurface", self.canvas)
            doc = work_surface.editor.doc

            # The new state is the current state of the model's tabs list,
            # which was modified live during the drag.
            new_tabs_state = deepcopy(parent_view.data.tabs)

            # Do not modify the model directly. Create a command with the
            # old and new states, and let the history manager apply the change.
            # This ensures a single, correct 'updated' signal is fired.
            cmd = ChangePropertyCommand(
                target=parent_view.data,
                property_name="tabs",
                new_value=new_tabs_state,
                old_value=self._initial_tabs_state,
                name=_("Move Tab"),
            )
            # The model is currently in the 'new' state due to live dragging.
            # We must set it back to the 'old' state so the undo manager can
            # correctly apply the 'new' state and record the change.
            # We bypass the public property setter to avoid firing a premature,
            # incorrect update signal, thus fixing the race condition.
            parent_view.data._tabs = self._initial_tabs_state
            doc.history_manager.execute(cmd)

            self._initial_tabs_state = None
            logger.debug(f"Drag end for tab {self.data.uid}")

    def handle_drag_move(
        self, world_dx: float, world_dy: float
    ) -> Tuple[float, float]:
        """
        Optimized drag handler that avoids redundant geometry calculations.
        """
        parent_view = cast("WorkPieceView", self.parent)
        vectors = parent_view.data.vectors
        if not self.canvas or not vectors:
            return world_dx, world_dy

        # Get mouse position in world coordinates
        world_mouse_x, world_mouse_y = self.canvas._get_world_coords(
            self.canvas._last_mouse_x, self.canvas._last_mouse_y
        )

        # Transform to parent's local, natural millimeter space
        try:
            inv_parent_world = parent_view.get_world_transform().invert()
            local_x_norm, local_y_norm = inv_parent_world.transform_point(
                (world_mouse_x, world_mouse_y)
            )
        except Exception:
            return world_dx, world_dy

        natural_size = parent_view.data.get_natural_size()
        if natural_size and None not in natural_size:
            natural_w, natural_h = cast(Tuple[float, float], natural_size)
        else:
            natural_w, natural_h = parent_view.data.get_local_size()

        if natural_w <= 1e-9 or natural_h <= 1e-9:
            return world_dx, world_dy

        local_x_mm = local_x_norm * natural_w
        local_y_mm = local_y_norm * natural_h

        # 1. Find closest point ONCE. This gives us index, t, and position.
        closest = vectors.find_closest_point(local_x_mm, local_y_mm)
        if not closest:
            return world_dx, world_dy
        segment_index, t, local_pos_mm = closest

        # 2. Get the tangent vector. This recalculates the point, but we
        #    ignore it and use the more precise one from find_closest_point.
        tangent_result = vectors.get_point_and_tangent_at(segment_index, t)
        if not tangent_result:
            return world_dx, world_dy
        _, local_tangent_mm = tangent_result

        # 3. Update the underlying Tab data model directly.
        tab_to_update = cast(Tab, self.data)
        tab_to_update.segment_index = segment_index
        tab_to_update.t = t

        # 4. Update the handle's internal geometry caches directly, avoiding
        #    the expensive update_base_geometry() call. This is the key
        #    performance optimization.
        self._local_pos_norm = (
            local_pos_mm[0] / natural_w,
            local_pos_mm[1] / natural_h,
        )
        self._local_tangent_norm = (
            local_tangent_mm[0] / natural_w,
            local_tangent_mm[1] / natural_h,
        )

        # The canvas will trigger a render, which will use the new cached
        # geometry to update the handle's on-screen transform.
        return world_dx, world_dy

    def render(self, ctx: cairo.Context):
        """
        Overrides render to ensure transform is always up-to-date before
        drawing.
        """
        self.update_transform()
        super().render(ctx)

    def update_base_geometry(self):
        """
        Calculates the handle's position and tangent vector based on its
        data model. This is used for initialization and non-performance
        -critical updates, but is bypassed by the drag handler.
        """
        parent_view = cast("WorkPieceView", self.parent)
        tab = cast(Tab, self.data)
        if not parent_view.data.vectors or tab.segment_index >= len(
            parent_view.data.vectors.commands
        ):
            return

        result = parent_view.data.vectors.get_point_and_tangent_at(
            tab.segment_index, tab.t
        )
        if not result:
            return
        local_pos_mm, local_tangent_mm = result

        natural_size = parent_view.data.get_natural_size()
        if natural_size and None not in natural_size:
            natural_w, natural_h = cast(Tuple[float, float], natural_size)
        else:
            natural_w, natural_h = parent_view.data.get_local_size()

        if natural_w > 1e-9 and natural_h > 1e-9:
            self._local_pos_norm = (
                local_pos_mm[0] / natural_w,
                local_pos_mm[1] / natural_h,
            )
            # Normalize the tangent vector relative to the parent's unit space.
            self._local_tangent_norm = (
                local_tangent_mm[0] / natural_w,
                local_tangent_mm[1] / natural_h,
            )

    def update_transform(self):
        """
        Calculates and sets this handle's transform to be a fixed pixel size
        with the correct orientation, regardless of parent transformations.
        """
        if not self.canvas or not self.parent:
            return

        parent_element = cast(CanvasElement, self.parent)
        work_surface = cast("WorkSurface", self.canvas)

        # 1. Get transforms and scales
        parent_world_transform = parent_element.get_world_transform()
        zoom_x, zoom_y = work_surface.get_view_scale()

        # 2. Transform local normalized pos/tangent into world space
        world_pos = parent_world_transform.transform_point(
            self._local_pos_norm
        )
        world_tangent = parent_world_transform.transform_vector(
            self._local_tangent_norm
        )

        # 3. Calculate visually correct angle from the world-space tangent
        world_angle_rad = math.atan2(world_tangent[1], world_tangent[0])

        # 4. Define handle size in pixels and convert to world units
        TARGET_WIDTH_PX = 10.0
        TARGET_LENGTH_PX = 22.0
        handle_width_world = TARGET_WIDTH_PX / zoom_x
        handle_length_world = TARGET_LENGTH_PX / zoom_y

        # 5. Construct the desired handle transform in WORLD space
        desired_world_transform = (
            Matrix.translation(world_pos[0], world_pos[1])
            @ Matrix.rotation(math.degrees(world_angle_rad))
            @ Matrix.scale(handle_width_world, handle_length_world)
            @ Matrix.translation(-0.5, -0.5)  # Center handle on its origin
        )

        # 6. Convert this world transform back into the handle's LOCAL
        # transform
        try:
            inv_parent_world = parent_world_transform.invert()
            local_transform = inv_parent_world @ desired_world_transform
            self.set_transform(local_transform)
        except Exception:
            # Invert can fail if parent is scaled to zero
            pass

    def draw(self, ctx: cairo.Context):
        """Draws the tab handle as a themed slot shape with a grip."""
        if not self.canvas:
            return

        style_context = self.canvas.get_style_context()

        # Define fallback RGBA colors
        fallback_bg = Gdk.RGBA(red=0.3, green=0.5, blue=0.9, alpha=0.8)
        fallback_fg = Gdk.RGBA(red=0.5, green=0.5, blue=0.9, alpha=0.9)
        fallback_grip = Gdk.RGBA(red=0.9, green=0.9, blue=0.9, alpha=0.5)

        # Use accent colors as required
        found, bg_color = style_context.lookup_color("accent_bg_color")
        bg_color = bg_color if found else fallback_bg
        found, fg_color = style_context.lookup_color("accent_color")
        fg_color = fg_color if found else fallback_fg
        found, grip_color = style_context.lookup_color("accent_fg_color")
        grip_color = grip_color if found else fallback_grip

        if self.is_hovered:
            bg_color.alpha = min(1.0, bg_color.alpha + 0.15)
            fg_color.alpha = 1.0

        # Deconstruct the element's transform to find its screen geometry
        original_ctm = ctx.get_matrix()
        p00 = original_ctm.transform_point(0, 0)
        p10 = original_ctm.transform_point(1, 0)
        p01 = original_ctm.transform_point(0, 1)

        vx_w, vy_w = p10[0] - p00[0], p10[1] - p00[1]
        screen_width = math.hypot(vx_w, vy_w)

        vx_l, vy_l = p01[0] - p00[0], p01[1] - p00[1]
        screen_length = math.hypot(vx_l, vy_l)

        if screen_width < 1 or screen_length < 1:
            return

        orientation_angle_rad = math.atan2(vy_l, vx_l)

        # Draw the shape in a clean, screen-aligned coordinate system
        ctx.save()
        try:
            ctx.identity_matrix()
            center_x = p00[0] + (vx_w + vx_l) / 2.0
            center_y = p00[1] + (vy_w + vy_l) / 2.0
            ctx.translate(center_x, center_y)
            ctx.rotate(orientation_angle_rad - math.pi / 2.0)

            w, h = screen_width, screen_length

            def _create_slot_path():
                """Helper to build the slot path geometry."""
                ctx.new_path()
                if h >= w:  # Taller than wide
                    radius = w / 2.0
                    y1 = -(h / 2.0 - radius)
                    y2 = h / 2.0 - radius
                    ctx.arc(0, y1, radius, math.pi, 0)
                    ctx.line_to(radius, y2)
                    ctx.arc(0, y2, radius, 0, math.pi)
                    ctx.close_path()
                else:  # Wider than tall
                    radius = h / 2.0
                    x1 = -(w / 2.0 - radius)
                    x2 = w / 2.0 - radius
                    ctx.arc(x1, 0, radius, math.pi / 2.0, 3.0 * math.pi / 2.0)
                    ctx.line_to(x2, -radius)
                    ctx.arc(x2, 0, radius, 3.0 * math.pi / 2.0, math.pi / 2.0)
                    ctx.close_path()

            # 1. Fill the slot background
            _create_slot_path()
            ctx.set_source_rgba(
                bg_color.red, bg_color.green, bg_color.blue, bg_color.alpha
            )
            ctx.fill()

            # 2. Draw the grip lines on top of the fill
            ctx.new_path()
            if h >= w:  # Vertical slot: lines are horizontal
                grip_len = w * 0.7
                x_start, x_end = -grip_len / 2.0, grip_len / 2.0
                y_spacing = h * 0.15
                ctx.move_to(x_start, -y_spacing)
                ctx.line_to(x_end, -y_spacing)
                ctx.move_to(x_start, 0)
                ctx.line_to(x_end, 0)
                ctx.move_to(x_start, y_spacing)
                ctx.line_to(x_end, y_spacing)
            else:  # Horizontal slot: lines are vertical
                grip_len = h * 0.8
                y_start, y_end = -grip_len / 2.0, grip_len / 2.0
                x_spacing = w * 0.15
                ctx.move_to(-x_spacing, y_start)
                ctx.line_to(-x_spacing, y_end)
                ctx.move_to(0, y_start)
                ctx.line_to(0, y_end)
                ctx.move_to(x_spacing, y_start)
                ctx.line_to(x_spacing, y_end)

            ctx.set_line_width(1.5)
            ctx.set_source_rgba(
                grip_color.red,
                grip_color.green,
                grip_color.blue,
                grip_color.alpha * 0.6,
            )
            ctx.stroke()

            # 3. Stroke the slot outline on top of everything
            _create_slot_path()
            ctx.set_line_width(1.0)
            ctx.set_source_rgba(
                fg_color.red,
                fg_color.green,
                fg_color.blue,
                fg_color.alpha * 0.6,
            )
            ctx.stroke()

        finally:
            ctx.restore()

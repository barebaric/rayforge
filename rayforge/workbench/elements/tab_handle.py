import cairo
import math
import logging
from typing import TYPE_CHECKING, cast, Optional, Tuple, List
from gi.repository import Gdk
from copy import deepcopy

from ..canvas.element import CanvasElement
from ...core.tab import Tab
from ...undo import ChangePropertyCommand

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
        Overrides drag behavior. Because drag_handler_controls_transform
        is True, this method directly updates the element's transform.
        """
        parent_view = cast("WorkPieceView", self.parent)
        if not self.canvas or not parent_view.data.vectors:
            return world_dx, world_dy

        # Get the current mouse position in world coordinates (mm).
        world_mouse_x, world_mouse_y = self.canvas._get_world_coords(
            self.canvas._last_mouse_x, self.canvas._last_mouse_y
        )

        # Transform world mouse pos to the parent workpiece's local 1x1
        # unit space
        try:
            inv_parent_world = parent_view.get_world_transform().invert()
            local_x_norm, local_y_norm = inv_parent_world.transform_point(
                (world_mouse_x, world_mouse_y)
            )
        except Exception:
            return world_dx, world_dy

        # The vectors exist in the workpiece's "natural" untransformed geometry
        # space. We must convert the normalized local point into that space.
        natural_size = parent_view.data.get_natural_size()
        if natural_size and None not in natural_size:
            natural_w, natural_h = cast(Tuple[float, float], natural_size)
        else:
            natural_w, natural_h = parent_view.data.get_local_size()

        if natural_w <= 0 or natural_h <= 0:
            return world_dx, world_dy

        local_x_mm = local_x_norm * natural_w
        local_y_mm = local_y_norm * natural_h

        # Find the closest point on the source geometry using mm coordinates
        result = parent_view.data.vectors.find_closest_point(
            local_x_mm, local_y_mm
        )
        if not result:
            return world_dx, world_dy

        # Update the tab data model directly for live feedback
        seg_idx, t, _ = result
        tab_to_update = cast(Tab, self.data)
        tab_to_update.segment_index = seg_idx
        tab_to_update.t = t

        # Reposition the visual handle based on the updated model data
        parent_view._position_handle_from_tab(self)

        # Return original delta; it will be ignored by the canvas, but
        # this satisfies the method signature and type hints.
        return world_dx, world_dy

    def draw(self, ctx: cairo.Context):
        """Draws the tab handle as a themed slot shape with a grip."""
        if not self.canvas:
            return

        style_context = self.canvas.get_style_context()

        # Define fallback RGBA colors
        fallback_bg = Gdk.RGBA(red=0.3, green=0.5, blue=0.9, alpha=0.8)
        fallback_fg = Gdk.RGBA(red=0.9, green=0.9, blue=0.9, alpha=0.9)

        # Use accent colors as required
        found, bg_color = style_context.lookup_color("accent_bg_color")
        bg_color = bg_color if found else fallback_bg
        found, fg_color = style_context.lookup_color("accent_color")
        fg_color = fg_color if found else fallback_fg

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

            # Draw Slot Path (always axis-aligned in this new context)
            ctx.new_sub_path()
            if h >= w:  # Taller than wide, or a circle
                radius = w / 2.0
                ctx.arc(0, -(h / 2.0 - radius), radius, math.pi, 0)
                ctx.arc(0, h / 2.0 - radius, radius, 0, math.pi)
            else:  # Wider than tall
                radius = h / 2.0
                ctx.arc(
                    -(w / 2.0 - radius),
                    0,
                    radius,
                    math.pi / 2.0,
                    3.0 * math.pi / 2.0,
                )
                ctx.arc(
                    w / 2.0 - radius,
                    0,
                    radius,
                    3.0 * math.pi / 2.0,
                    math.pi / 2.0,
                )
            ctx.close_path()

            # Fill and stroke
            ctx.set_source_rgba(
                bg_color.red, bg_color.green, bg_color.blue, bg_color.alpha
            )
            ctx.fill_preserve()
            ctx.set_line_width(1.0)
            ctx.set_source_rgba(
                fg_color.red,
                fg_color.green,
                fg_color.blue,
                fg_color.alpha * 0.7,
            )
            ctx.stroke()

            # Draw Grip Lines
            ctx.new_path()
            if h >= w:  # Vertical slot: lines are horizontal
                grip_len = w * 0.7
                x_start, x_end = -grip_len / 2.0, grip_len / 2.0
                y_spacing = min(h * 0.1, w * 0.4)
                ctx.move_to(x_start, -y_spacing)
                ctx.line_to(x_end, -y_spacing)
                ctx.move_to(x_start, 0)
                ctx.line_to(x_end, 0)
                ctx.move_to(x_start, y_spacing)
                ctx.line_to(x_end, y_spacing)
            else:  # Horizontal slot: lines are vertical
                grip_len = h * 0.7
                y_start, y_end = -grip_len / 2.0, grip_len / 2.0
                x_spacing = min(w * 0.1, h * 0.4)
                ctx.move_to(-x_spacing, y_start)
                ctx.line_to(-x_spacing, y_end)
                ctx.move_to(0, y_start)
                ctx.line_to(0, y_end)
                ctx.move_to(x_spacing, y_start)
                ctx.line_to(x_spacing, y_end)

            ctx.set_line_width(1.0)
            ctx.set_source_rgba(
                fg_color.red, fg_color.green, fg_color.blue, fg_color.alpha
            )
            ctx.stroke()

        finally:
            ctx.restore()

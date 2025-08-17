from __future__ import annotations
import math
import cairo
from typing import cast
from gi.repository import GLib

from ...core.matrix import Matrix
from .element import CanvasElement


class ShrinkWrapGroup(CanvasElement):
    """
    A generic group element that automatically calculates its bounding
    box to tightly enclose all of its children ("shrink-wrap").

    When its bounds are updated, it adjusts its own transformation matrix and
    simultaneously calculates and applies compensating transforms to all its
    children, so their world position, scale, and rotation remain unchanged.
    """

    def __init__(self, x: float = 0, y: float = 0, **kwargs):
        super().__init__(x, y, 1, 1, clip=False, **kwargs)
        self._bounds_update_scheduled: bool = False

    def _schedule_update_bounds(self):
        """Schedules a deferred bounds update, debouncing multiple requests."""
        if self._bounds_update_scheduled:
            return
        self._bounds_update_scheduled = True
        GLib.idle_add(self._do_update_bounds)

    def _do_update_bounds(self) -> bool:
        """The idle callback that performs the actual update."""
        self._bounds_update_scheduled = False
        self.update_bounds()
        if self.canvas:
            self.canvas.queue_draw()
        return False  # Do not call again

    def on_child_transform_changed(self, child: "CanvasElement"):
        """Override to handle updates synchronously or via preview."""
        if child._is_under_interactive_transform:
            # While dragging, do NOT update transforms. Just redraw so the
            # overlay preview is shown.
            if self.canvas:
                self.canvas.queue_draw()
        else:
            # The interaction has just ended. Update the bounds *immediately*
            # and synchronously. This ensures that when the WorkSurface's
            # _on_transform_end handler runs moments later, the group's
            # transform is already correct.
            self.update_bounds()
            if self.canvas:
                self.canvas.queue_draw()

    def on_child_list_changed(self):
        """
        When children are added/removed, schedule a deferred update for safety.
        """
        self._schedule_update_bounds()

    def update_bounds(self):
        """
        Calculates and applies the group's new transform and compensating
        child transforms. This is only called when the scene is stable.
        """
        if not self.children:
            self.set_transform(Matrix.identity())
            return

        child_desired_world_transforms = {
            child: child.get_world_transform() for child in self.children
        }

        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for child, world_transform in child_desired_world_transforms.items():
            x, y, w, h = world_transform.transform_rectangle(
                (0, 0, child.width, child.height)
            )
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        if not all(map(math.isfinite, [min_x, min_y, max_x, max_y])):
            return

        new_world_w = max(max_x - min_x, 1e-9)
        new_world_h = max(max_y - min_y, 1e-9)

        new_group_world_transform = Matrix.translation(
            min_x, min_y
        ) @ Matrix.scale(new_world_w, new_world_h)

        parent_world_transform = (
            cast(CanvasElement, self.parent).get_world_transform()
            if isinstance(self.parent, CanvasElement)
            else Matrix.identity()
        )
        try:
            new_group_local_transform = (
                parent_world_transform.invert() @ new_group_world_transform
            )
            inv_new_group_world = new_group_world_transform.invert()
        except Exception:
            return

        self._set_transform_silent(new_group_local_transform)
        for child, desired_world in child_desired_world_transforms.items():
            new_child_local = inv_new_group_world @ desired_world
            child._set_transform_silent(new_child_local)

        if isinstance(self.parent, CanvasElement):
            self.parent.on_child_transform_changed(self)

    def draw_overlay(self, ctx: cairo.Context):
        """
        Draws a live preview of the bounding box during a child transform.
        """
        if not self.canvas or not any(
            c._is_under_interactive_transform for c in self.children
        ):
            return

        # Calculate the current bounding box in world coordinates
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
        for child in self.children:
            x, y, w, h = child.get_world_bounding_box()
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        if not all(map(math.isfinite, [min_x, min_y, max_x, max_y])):
            return

        # Transform the world-space AABB to pixel-space for drawing
        points = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        ]
        pixel_points = [
            self.canvas.view_transform.transform_point(p) for p in points
        ]

        # Draw the dashed preview frame
        ctx.save()
        ctx.set_source_rgba(0.5, 0.7, 1.0, 0.9)
        ctx.set_line_width(1.0)
        ctx.set_dash([4.0, 2.0])

        p1_x, p1_y = pixel_points[0]
        ctx.move_to(round(p1_x) + 0.5, round(p1_y) + 0.5)
        for x, y in pixel_points[1:]:
            ctx.line_to(round(x) + 0.5, round(y) + 0.5)
        ctx.close_path()
        ctx.stroke()
        ctx.restore()

    def draw(self, ctx: cairo.Context):
        """
        Draws a crisp, dashed bounding box for the group, but only when
        it is NOT selected and NOT being interactively updated via a child.
        """
        is_interacting = any(
            c._is_under_interactive_transform for c in self.children
        )

        if self.selected or not self.canvas or is_interacting:
            return

        # 1. Calculate the final screen coordinates of the group's corners.
        transform_to_screen = (
            self.canvas.view_transform @ self.get_world_transform()
        )
        if transform_to_screen.has_zero_scale():
            return

        # The group's local coordinate system is a unit square.
        local_corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        screen_corners = [
            transform_to_screen.transform_point(p) for p in local_corners
        ]

        # 2. Reset the transformation matrix to draw directly in screen space.
        ctx.save()
        ctx.identity_matrix()

        # 3. Draw the path with pixel-perfect dimensions.
        ctx.set_source_rgba(0.5, 0.7, 1.0, 0.9)
        ctx.set_line_width(1.0)  # 1 pixel
        ctx.set_dash([4.0, 2.0])  # 4 pixels on, 2 pixels off

        # For crisp lines, it's best to draw on the half-pixel grid.
        start_x, start_y = screen_corners[0]
        ctx.move_to(round(start_x) + 0.5, round(start_y) + 0.5)
        for x, y in screen_corners[1:]:
            ctx.line_to(round(x) + 0.5, round(y) + 0.5)

        ctx.close_path()
        ctx.stroke()

        # 4. Restore the original transformation matrix.
        ctx.restore()

import math
from collections.abc import Callable
from gettext import gettext as _
from typing import ClassVar

import cairo

from ...core.commands import EllipseCommand, EllipsePreviewState
from .base import SketcherKey, SketchTool
from .snap_mixin import SnapMixin


class CircleTool(SnapMixin, SketchTool):
    """Handles creating ellipses/circles via two clicks or drag.

    - Default: click -> move -> click (drag-to-create also works)
    - Ctrl: constrain to circle (equal radii)
    - Shift: center the ellipse on the starting point
    - Tab: toggle magnetic snap
    """

    ICON = "sketch-circle-symbolic"
    LABEL = _("Ellipse")
    SHORTCUTS: ClassVar[list[str]] = ["gc"]
    CURSOR_ICON = "sketch-circle-symbolic"

    DRAG_THRESHOLD = 2.0

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: EllipsePreviewState | None = None
        self._ctrl_held = False
        self._shift_held = False
        self._press_world_pos: tuple[float, float] | None = None
        self._in_press = False
        self._last_preview_pos: tuple[float, float] | None = None

    def is_available(self, target, target_type) -> bool:
        return target is None

    def shortcut_is_active(self) -> bool:
        return True

    def get_preview_state(self) -> EllipsePreviewState | None:
        return self._preview_state

    def on_deactivate(self):
        """Clean up if the tool is deactivated mid-creation."""
        self._press_world_pos = None
        self._in_press = False
        self._last_preview_pos = None
        if self._preview_state is not None:
            start_id = self._preview_state.start_id
            start_temp = self._preview_state.start_temp

            EllipseCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )
            self._preview_state = None
            self.element.preview_changed.send(self.element)

            if start_temp:
                self.element.remove_point_if_unused(start_id)

            self.element.mark_dirty()

        self.clear_snap_result()

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        exclude_points = set()
        if self._preview_state is not None:
            exclude_points = self._preview_state.get_preview_point_ids()

        mx, my = self.query_snap_for_creation(
            self.element, mx, my, exclude_points
        )

        pid_hit = self.get_snapped_point_id()

        if self._preview_state is None:
            self._preview_state = EllipseCommand.start_preview(
                self.element.sketch.registry, mx, my, snapped_pid=pid_hit
            )
            self._press_world_pos = (world_x, world_y)
            self._in_press = True
            self._last_preview_pos = (mx, my)
            self.element.preview_changed.send(self.element)
            self.element.mark_dirty()
        else:
            self._commit(mx, my, pid_hit)
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        self._in_press = False
        if self._preview_state is None or self._press_world_pos is None:
            return

        dx = world_x - self._press_world_pos[0]
        dy = world_y - self._press_world_pos[1]
        if math.hypot(dx, dy) < self.DRAG_THRESHOLD:
            return

        # Commit at the position the preview was last drawn at, so the
        # shape matches the preview regardless of pointer jitter between
        # the last motion event and the release.
        if self._last_preview_pos is None:
            return

        mx, my = self._last_preview_pos
        pid_hit = self.get_snapped_point_id()
        self._commit(mx, my, pid_hit)

    def _commit(self, mx: float, my: float, pid_hit: int | None) -> None:
        """Finalize the preview ellipse at the given snapped position."""
        if self._preview_state is None:
            return

        preview_ids = self._preview_state.get_preview_point_ids()
        start_id = self._preview_state.start_id
        start_temp = self._preview_state.start_temp

        EllipseCommand.cleanup_preview(
            self.element.sketch.registry, self._preview_state
        )
        self._preview_state = None
        self._press_world_pos = None
        self._in_press = False
        self._last_preview_pos = None
        self.element.preview_changed.send(self.element)

        final_pid = None if pid_hit in preview_ids else pid_hit

        if self._is_degenerate(start_id, mx, my):
            if start_temp:
                self.element.remove_point_if_unused(start_id)
        else:
            cmd = EllipseCommand(
                self.element.sketch,
                start_id,
                (mx, my),
                end_pid=final_pid,
                is_start_temp=start_temp,
                center_on_start=self._shift_held,
                constrain_circle=self._ctrl_held,
            )
            self.element.execute_command(cmd)

        self.element.mark_dirty()
        self.clear_snap_result()

    def _is_degenerate(self, start_id: int, mx: float, my: float) -> bool:
        """True when the commit position matches the start point."""
        try:
            start_p = self.element.sketch.registry.get_point(start_id)
        except (IndexError, KeyError):
            return False
        return math.hypot(mx - start_p.x, my - start_p.y) < 1e-6

    def on_hover_motion(self, world_x: float, world_y: float):
        """Updates the live preview of the ellipse."""
        if self._preview_state is None:
            self.clear_snap_result()
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        preview_ids = self._preview_state.get_preview_point_ids()
        mx, my = self.query_snap_for_creation(
            self.element, mx, my, preview_ids
        )
        self._last_preview_pos = (mx, my)

        try:
            EllipseCommand.update_preview(
                self.element.sketch.registry,
                self._preview_state,
                mx,
                my,
                center_on_start=self._shift_held,
                constrain_circle=self._ctrl_held,
            )
            self.element.mark_dirty()
        except (IndexError, KeyError):
            self.on_deactivate()

    def draw_overlay(self, ctx: cairo.Context):
        """Draw snap feedback during creation."""
        if self._preview_state is not None:
            self.draw_snap_feedback(ctx, self.element)

    def handle_key_event(
        self, key: SketcherKey, shift: bool = False, ctrl: bool = False
    ) -> bool:
        """Handle modifier keys for ellipse creation."""
        if self._preview_state is None:
            return False

        if key == SketcherKey.ESCAPE:
            self.on_deactivate()
            return True

        if key == SketcherKey.TAB:
            self.toggle_magnetic_snap()
            return True

        return False

    def on_modifier_change(self, shift: bool = False, ctrl: bool = False):
        """Called when modifier keys change during drag."""
        if self._preview_state is None:
            return

        changed = self._ctrl_held != ctrl or self._shift_held != shift
        self._ctrl_held = ctrl
        self._shift_held = shift

        if changed:
            self.element.mark_dirty()

    def get_active_shortcuts(
        self,
    ) -> list[tuple[str | list[str], str, Callable[[], bool] | None]]:
        """Returns shortcuts for the status bar."""
        if self._preview_state is not None:
            shortcuts: list[
                tuple[str | list[str], str, Callable[[], bool] | None]
            ] = [
                ("Shift", _("Center on start point"), None),
                ("Ctrl", _("Constrain to circle"), None),
                ("Tab", _("Toggle Magnetic Snap"), None),
            ]
            if not self._in_press:
                shortcuts.append(("Click", _("Set edge point"), None))
            shortcuts.append(("Esc", _("Cancel"), None))
            return shortcuts
        return []

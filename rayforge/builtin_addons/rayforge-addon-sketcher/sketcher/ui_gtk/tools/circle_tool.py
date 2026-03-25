from gettext import gettext as _
from typing import Callable, List, Optional, Tuple, Union

from ...core.commands import EllipseCommand, EllipsePreviewState
from .base import SketchTool, SketcherKey


class CircleTool(SketchTool):
    """Handles creating ellipses/circles via drag-to-create.

    - Default: drag creates an ellipse fitting the bounding box
    - Ctrl: constrain to circle (equal radii)
    - Shift: center the ellipse on the starting point
    """

    ICON = "sketch-circle-symbolic"
    LABEL = _("Ellipse")
    SHORTCUTS = ["gc"]
    CURSOR_ICON = "sketch-circle-symbolic"

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: Optional[EllipsePreviewState] = None
        self._ctrl_held = False
        self._shift_held = False

    def is_available(self, target, target_type) -> bool:
        return target is None

    def shortcut_is_active(self) -> bool:
        return True

    def get_preview_state(self) -> Optional[EllipsePreviewState]:
        return self._preview_state

    def on_deactivate(self):
        """Clean up if the tool is deactivated mid-creation."""
        if self._preview_state is not None:
            start_id = self._preview_state.start_id
            start_temp = self._preview_state.start_temp

            EllipseCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )
            self._preview_state = None

            if start_temp:
                self.element.remove_point_if_unused(start_id)

            self.element.mark_dirty()

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )
        hit_type, hit_obj = self.element.hittester.get_hit_data(
            world_x, world_y, self.element
        )
        pid_hit = hit_obj if hit_type == "point" else None

        if self._preview_state is None:
            self._preview_state = EllipseCommand.start_preview(
                self.element.sketch.registry, mx, my, snapped_pid=pid_hit
            )
            self.element.mark_dirty()
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        if self._preview_state is None:
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )
        hit_type, hit_obj = self.element.hittester.get_hit_data(
            world_x, world_y, self.element
        )
        pid_hit = hit_obj if hit_type == "point" else None

        preview_ids = self._preview_state.get_preview_point_ids()
        start_id = self._preview_state.start_id
        start_temp = self._preview_state.start_temp

        EllipseCommand.cleanup_preview(
            self.element.sketch.registry, self._preview_state
        )
        self._preview_state = None

        final_pid = None if pid_hit in preview_ids else pid_hit

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

    def on_hover_motion(self, world_x: float, world_y: float):
        """Updates the live preview of the ellipse."""
        if self._preview_state is None:
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

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

    def handle_key_event(
        self, key: SketcherKey, shift: bool = False, ctrl: bool = False
    ) -> bool:
        """Handle modifier keys for ellipse creation."""
        if self._preview_state is None:
            return False

        if key == SketcherKey.ESCAPE:
            self.on_deactivate()
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
    ) -> List[Tuple[Union[str, List[str]], str, Optional[Callable[[], bool]]]]:
        """Returns shortcuts for the status bar."""
        if self._preview_state is not None:
            return [
                ("Shift", _("Center on start point"), None),
                ("Ctrl", _("Constrain to circle"), None),
                ("Esc", _("Cancel"), None),
            ]
        return []

from typing import Optional

from ....core.sketcher.commands import (
    RoundedRectCommand,
    RoundedRectPreviewState,
)
from .base import SketchTool


class RoundedRectTool(SketchTool):
    """Handles creating rounded rectangles."""

    DEFAULT_RADIUS = 10.0

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: Optional[RoundedRectPreviewState] = None

    def on_deactivate(self):
        """Clean up if the tool is deactivated mid-creation."""
        if self._preview_state is None:
            return

        start_id = self._preview_state.start_id
        start_temp = self._preview_state.start_temp

        RoundedRectCommand.cleanup_preview(
            self.element.sketch.registry, self._preview_state
        )

        if start_temp:
            self.element.remove_point_if_unused(start_id)

        self._preview_state = None
        self.element.mark_dirty()

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )
        hit_type, hit_obj = self.element.hittester.get_hit_data(
            world_x, world_y, self.element
        )
        pid_hit = hit_obj if hit_type == "point" else None
        return self._handle_click(pid_hit, mx, my)

    def on_hover_motion(self, world_x: float, world_y: float):
        """Updates the live preview of the rounded rectangle."""
        if self._preview_state is None:
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        try:
            RoundedRectCommand.update_preview(
                self.element.sketch.registry, self._preview_state, mx, my
            )
            self.element.mark_dirty()
        except (IndexError, KeyError):
            self.on_deactivate()

    def _handle_click(
        self, pid_hit: Optional[int], mx: float, my: float
    ) -> bool:
        if self._preview_state is None:
            # --- First Click: Start preview ---
            self._preview_state = RoundedRectCommand.start_preview(
                self.element.sketch.registry,
                mx,
                my,
                snapped_pid=pid_hit,
                radius=self.DEFAULT_RADIUS,
            )
        else:
            # --- Second Click: Finalize the rounded rectangle ---
            start_id = self._preview_state.start_id
            start_temp = self._preview_state.start_temp

            RoundedRectCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )
            self._preview_state = None

            cmd = RoundedRectCommand(
                self.element.sketch,
                start_id,
                (mx, my),
                self.DEFAULT_RADIUS,
                is_start_temp=start_temp,
            )
            self.element.execute_command(cmd)

        self.element.mark_dirty()
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

from gettext import gettext as _
from typing import Optional

from ....core.sketcher.commands import BezierCommand, BezierPreviewState
from .base import SketchTool


class BezierTool(SketchTool):
    """Handles creating cubic bezier curves (Start -> CP1 -> CP2/End)."""

    SHORTCUT = ("gb", _("Bezier"))

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: Optional[BezierPreviewState] = None

    def get_preview_state(self) -> Optional[BezierPreviewState]:
        return self._preview_state

    def on_deactivate(self):
        """Clean up any intermediate points if the bezier was not finished."""
        if self._preview_state is not None:
            BezierCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )

            if self._preview_state.start_temp:
                self.element.remove_point_if_unused(
                    self._preview_state.start_id
                )
            if self._preview_state.cp1_temp and self._preview_state.cp1_id:
                self.element.remove_point_if_unused(self._preview_state.cp1_id)
            if self._preview_state.cp2_temp and self._preview_state.cp2_id:
                self.element.remove_point_if_unused(self._preview_state.cp2_id)

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

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_hover_motion(self, world_x: float, world_y: float):
        """Updates the live preview of the bezier."""
        if self._preview_state is None:
            return

        if not self._preview_state.has_cp1:
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        try:
            BezierCommand.update_preview(
                self.element.sketch.registry, self._preview_state, mx, my
            )
            self.element.mark_dirty()
        except IndexError:
            self.on_deactivate()

    def _handle_click(self, pid_hit, mx, my) -> bool:
        if self._preview_state is not None:
            try:
                self.element.sketch.registry.get_point(
                    self._preview_state.start_id
                )
            except IndexError:
                self.on_deactivate()
                return True

        if self._preview_state is None:
            self._preview_state = BezierCommand.start_preview(
                self.element.sketch.registry, mx, my, snapped_pid=pid_hit
            )
            self.element.update_bounds_from_sketch()

        elif not self._preview_state.has_cp1:
            if pid_hit == self._preview_state.start_id:
                self.element.mark_dirty()
                return True

            BezierCommand.set_cp1(
                self.element.sketch.registry,
                self._preview_state,
                mx,
                my,
                snapped_pid=pid_hit,
            )
            self.element.update_bounds_from_sketch()

        elif not self._preview_state.has_cp2:
            BezierCommand.set_cp2_and_preview(
                self.element.sketch.registry,
                self._preview_state,
                mx,
                my,
                snapped_pid=pid_hit,
            )
            self.element.update_bounds_from_sketch()

        else:
            if self._preview_state is None:
                return False
            if self._preview_state.cp1_id is None:
                return False
            if self._preview_state.cp2_id is None:
                return False

            preview_ids = self._preview_state.get_preview_point_ids()
            start_id = self._preview_state.start_id
            start_temp = self._preview_state.start_temp
            cp1_id = self._preview_state.cp1_id
            cp1_temp = self._preview_state.cp1_temp
            cp2_id = self._preview_state.cp2_id
            cp2_temp = self._preview_state.cp2_temp

            BezierCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )

            self._preview_state = None

            final_pid = None if pid_hit in preview_ids else pid_hit

            cmd = BezierCommand(
                self.element.sketch,
                start_id,
                cp1_id,
                cp2_id,
                (mx, my),
                end_pid=final_pid,
                is_start_temp=start_temp,
                is_cp1_temp=cp1_temp,
                is_cp2_temp=cp2_temp,
            )
            self.element.execute_command(cmd)

        self.element.mark_dirty()
        return True

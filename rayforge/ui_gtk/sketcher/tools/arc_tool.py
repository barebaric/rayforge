from gettext import gettext as _
from typing import Optional

from ....core.sketcher.commands import ArcCommand, ArcPreviewState
from .base import SketchTool


class ArcTool(SketchTool):
    """Handles creating arcs (Center -> Start -> End)."""

    SHORTCUT = ("ga", _("Arc"))

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: Optional[ArcPreviewState] = None

    def on_deactivate(self):
        """Clean up any intermediate points if the arc was not finished."""
        if self._preview_state is not None:
            if self._preview_state.has_start_point:
                ArcCommand.cleanup_preview(
                    self.element.sketch.registry, self._preview_state
                )

            if self._preview_state.center_temp:
                self.element.remove_point_if_unused(
                    self._preview_state.center_id
                )
            if self._preview_state.start_temp:
                self.element.remove_point_if_unused(
                    self._preview_state.start_id
                )

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
        """Updates the live preview of the arc."""
        if self._preview_state is None:
            return

        if not self._preview_state.has_start_point:
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        try:
            ArcCommand.update_preview(
                self.element.sketch.registry, self._preview_state, mx, my
            )
            self.element.mark_dirty()
        except IndexError:
            self.on_deactivate()

    def _handle_click(self, pid_hit, mx, my) -> bool:
        # State machine: Center -> Start -> End

        if self._preview_state is not None:
            try:
                self.element.sketch.registry.get_point(
                    self._preview_state.center_id
                )
            except IndexError:
                self.on_deactivate()
                return True

        if self._preview_state is None:
            # Step 1: Center Point
            self._preview_state = ArcCommand.start_center_preview(
                self.element.sketch.registry, mx, my, snapped_pid=pid_hit
            )
            self.element.update_bounds_from_sketch()

        elif not self._preview_state.has_start_point:
            # Step 2: Start Point
            if pid_hit == self._preview_state.center_id:
                self.element.mark_dirty()
                return True

            ArcCommand.set_start_point(
                self.element.sketch.registry,
                self._preview_state,
                mx,
                my,
                snapped_pid=pid_hit,
            )
            self.element.update_bounds_from_sketch()

        else:
            # Step 3: End Point (Finalize)
            if self._preview_state is None:
                return False
            if self._preview_state.start_id is None:
                return False

            preview_ids = self._preview_state.get_preview_point_ids()
            clockwise = self._preview_state.clockwise
            center_id = self._preview_state.center_id
            center_temp = self._preview_state.center_temp
            start_id = self._preview_state.start_id
            start_temp = self._preview_state.start_temp

            ArcCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )

            self._preview_state = None

            final_pid = None if pid_hit in preview_ids else pid_hit

            cmd = ArcCommand(
                self.element.sketch,
                center_id,
                start_id,
                (mx, my),
                end_pid=final_pid,
                is_center_temp=center_temp,
                is_start_temp=start_temp,
                clockwise=clockwise,
            )
            self.element.execute_command(cmd)

        self.element.mark_dirty()
        return True

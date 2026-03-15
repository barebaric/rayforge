from typing import Optional

from ....core.sketcher.commands import CircleCommand, CirclePreviewState
from .base import SketchTool


class CircleTool(SketchTool):
    """Handles creating circles (Center -> Radius Point)."""

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: Optional[CirclePreviewState] = None

    def on_deactivate(self):
        """Clean up if the tool is deactivated mid-creation."""
        if self._preview_state is not None:
            center_id = self._preview_state.center_id
            center_temp = self._preview_state.center_temp

            CircleCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )
            self._preview_state = None

            if center_temp:
                self.element.remove_point_if_unused(center_id)

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
        """Updates the live preview of the circle."""
        if self._preview_state is None:
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        try:
            CircleCommand.update_preview(
                self.element.sketch.registry, self._preview_state, mx, my
            )
            self.element.mark_dirty()
        except (IndexError, KeyError):
            self.on_deactivate()

    def _handle_click(self, pid_hit, mx, my) -> bool:
        if self._preview_state is None:
            # --- First Click: Start preview ---
            self._preview_state = CircleCommand.start_preview(
                self.element.sketch.registry, mx, my, snapped_pid=pid_hit
            )
            self.element.selection.clear()
            self.element.selection.select_point(
                self._preview_state.center_id, False
            )
        else:
            # --- Second Click: Finalize the circle ---
            preview_radius_id = self._preview_state.radius_id

            center_id = self._preview_state.center_id
            center_temp = self._preview_state.center_temp

            CircleCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )
            self._preview_state = None

            # If we hit our own preview point, treat it as no snap
            final_pid = None if pid_hit == preview_radius_id else pid_hit

            # Cannot have radius point at center
            if final_pid != center_id:
                cmd = CircleCommand(
                    self.element.sketch,
                    center_id,
                    (mx, my),
                    end_pid=final_pid,
                    is_center_temp=center_temp,
                )
                self.element.execute_command(cmd)

                # Get the final end point ID for selection
                if cmd.add_cmd and cmd.add_cmd.points:
                    for p in cmd.add_cmd.points:
                        if p.x == mx or (
                            final_pid is not None and p.id == final_pid
                        ):
                            final_pid = p.id
                            break

                self.element.selection.clear()
                if final_pid is not None:
                    self.element.selection.select_point(final_pid, False)

        self.element.mark_dirty()
        return True

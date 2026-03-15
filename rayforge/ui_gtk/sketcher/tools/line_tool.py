from gettext import gettext as _
from typing import Optional

from ....core.sketcher.commands import (
    LineCommand,
    LinePreviewState,
)
from .base import SketchTool


class LineTool(SketchTool):
    """Handles creating lines between points."""

    SHORTCUT = ("gl", _("Line"))

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: Optional[LinePreviewState] = None

    def on_deactivate(self):
        """Clean up if the tool is deactivated mid-creation."""
        if self._preview_state is not None:
            start_id = self._preview_state.start_id
            start_temp = self._preview_state.start_temp

            LineCommand.cleanup_preview(
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
        return self._handle_click(pid_hit, mx, my)

    def on_hover_motion(self, world_x: float, world_y: float):
        """Updates the live preview of the line."""
        if self._preview_state is None:
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        try:
            LineCommand.update_preview(
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
            self._preview_state = LineCommand.start_preview(
                self.element.sketch.registry, mx, my, snapped_pid=pid_hit
            )
            self.element.selection.clear()
            self.element.selection.select_point(
                self._preview_state.start_id, False
            )
        else:
            # --- Second Click: Finalize the line ---
            preview_ids = self._preview_state.get_preview_point_ids()
            start_id = self._preview_state.start_id
            start_temp = self._preview_state.start_temp

            LineCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )
            self._preview_state = None

            # If we hit a preview point, treat it as no snap (use mouse coords)
            final_pid = None if pid_hit in preview_ids else pid_hit

            # Handle case where start point was deleted during preview
            try:
                self.element.sketch.registry.get_point(start_id)
            except IndexError:
                self.element.mark_dirty()
                return True

            # Create the command to generate the line
            cmd = LineCommand(
                self.element.sketch,
                start_id,
                (mx, my),
                end_pid=final_pid,
                is_start_temp=start_temp,
            )
            self.element.execute_command(cmd)

            if cmd.committed_end_id is not None:
                try:
                    end_pt = self.element.sketch.registry.get_point(
                        cmd.committed_end_id
                    )
                    self._preview_state = LineCommand.start_preview(
                        self.element.sketch.registry,
                        end_pt.x,
                        end_pt.y,
                        snapped_pid=cmd.committed_end_id,
                    )
                    self.element.selection.clear()
                    self.element.selection.select_point(
                        cmd.committed_end_id, False
                    )
                except IndexError:
                    pass

        self.element.mark_dirty()
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

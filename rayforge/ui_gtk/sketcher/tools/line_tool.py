from gettext import gettext as _
from typing import Callable, List, Optional, Tuple, Union

from ....core.sketcher.commands import (
    LineCommand,
    LinePreviewState,
)
from .base import SketchTool, SketcherKey
from .dimension_input import DimensionInputHandler


class LineTool(SketchTool):
    """Handles creating lines between points."""

    ICON = "sketch-line-symbolic"
    LABEL = _("Line")
    SHORTCUTS = ["gl"]

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: Optional[LinePreviewState] = None
        self._dim_input = DimensionInputHandler()

    def is_available(self, target, target_type) -> bool:
        return target is None

    def shortcut_is_active(self) -> bool:
        return True

    def get_preview_state(self) -> Optional[LinePreviewState]:
        return self._preview_state

    def on_deactivate(self):
        """Clean up if the tool is deactivated mid-creation."""
        self._dim_input.cancel()
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

        if self._dim_input.is_active():
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
        else:
            # --- Second Click: Finalize the line ---
            preview_ids = self._preview_state.get_preview_point_ids()
            start_id = self._preview_state.start_id
            start_temp = self._preview_state.start_temp

            LineCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )
            self._preview_state = None
            self._dim_input.cancel()

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
                except IndexError:
                    pass

        self.element.mark_dirty()
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def handle_text_input(self, text: str) -> bool:
        """Handle numeric input for setting line length."""
        if self._preview_state is None:
            return False

        if not self._dim_input.is_active():
            self._dim_input.start()

        handled = self._dim_input.handle_text_input(text)
        if handled:
            self.element.mark_dirty()
        return handled

    def handle_key_event(
        self, key: SketcherKey, shift: bool = False, ctrl: bool = False
    ) -> bool:
        """Handle special keys for dimension input."""
        if self._preview_state is None:
            return False

        if key == SketcherKey.BACKSPACE:
            if self._dim_input.is_active():
                self._dim_input.handle_backspace()
                self.element.mark_dirty()
                return True
            return False

        if key == SketcherKey.DELETE:
            if self._dim_input.is_active():
                self._dim_input.handle_delete()
                self.element.mark_dirty()
                return True
            return False

        if key == SketcherKey.TAB:
            if self._dim_input.is_active():
                self._apply_dimension_input()
                return True
            return False

        if key == SketcherKey.RETURN:
            if self._dim_input.is_active():
                self._apply_dimension_input()
                return True
            return False

        if key == SketcherKey.ESCAPE:
            if self._dim_input.is_active():
                self._dim_input.cancel()
                self.element.mark_dirty()
                return True
            return False

        return False

    def _apply_dimension_input(self):
        """Apply the dimension input to the preview line."""
        if self._preview_state is None:
            self._dim_input.cancel()
            return

        values = self._dim_input.commit()
        if values is None or len(values) == 0:
            return

        length = values[0]
        if length is None:
            return

        self._preview_state.set_length(self.element.sketch.registry, length)
        self._finalize_shape(fixed_length=length)
        self.element.mark_dirty()

    def _finalize_shape(self, fixed_length: Optional[float] = None):
        if self._preview_state is None:
            return
        start_id = self._preview_state.start_id
        start_temp = self._preview_state.start_temp
        try:
            end_pt = self.element.sketch.registry.get_point(
                self._preview_state.end_id
            )
        except IndexError:
            self._preview_state = None
            self._dim_input.cancel()
            self.element.mark_dirty()
            return
        mx = end_pt.x
        my = end_pt.y
        LineCommand.cleanup_preview(
            self.element.sketch.registry, self._preview_state
        )
        self._preview_state = None
        self._dim_input.cancel()
        cmd = LineCommand(
            self.element.sketch,
            start_id,
            (mx, my),
            end_pid=None,
            is_start_temp=start_temp,
            fixed_length=fixed_length,
        )
        self.element.execute_command(cmd)
        self.element.mark_dirty()

    def get_active_shortcuts(
        self,
    ) -> List[Tuple[Union[str, List[str]], str, Optional[Callable[[], bool]]]]:
        """Returns shortcuts for the status bar."""
        if self._preview_state is not None:
            if self._dim_input.is_active():
                return self._dim_input.get_active_shortcuts()
            return [
                ("0-9", _("Type length"), None),
            ]
        return []

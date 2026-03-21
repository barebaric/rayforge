from gettext import gettext as _
from typing import Callable, List, Optional, Tuple, Union

from ....core.sketcher.commands import CircleCommand, CirclePreviewState
from .base import SketchTool, SketcherKey
from .dimension_input import DimensionInputHandler


class CircleTool(SketchTool):
    """Handles creating circles (Center -> Radius Point)."""

    ICON = "sketch-circle-symbolic"
    LABEL = _("Circle")
    SHORTCUTS = ["gc"]

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: Optional[CirclePreviewState] = None
        self._dim_input = DimensionInputHandler()

    def is_available(self, target, target_type) -> bool:
        return target is None

    def shortcut_is_active(self) -> bool:
        return True

    def get_preview_state(self) -> Optional[CirclePreviewState]:
        return self._preview_state

    def on_deactivate(self):
        """Clean up if the tool is deactivated mid-creation."""
        self._dim_input.cancel()
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

        if self._dim_input.is_active():
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
        else:
            # --- Second Click: Finalize the circle ---
            preview_ids = self._preview_state.get_preview_point_ids()
            center_id = self._preview_state.center_id
            center_temp = self._preview_state.center_temp

            CircleCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )
            self._preview_state = None
            self._dim_input.cancel()

            final_pid = None if pid_hit in preview_ids else pid_hit

            if final_pid != center_id:
                cmd = CircleCommand(
                    self.element.sketch,
                    center_id,
                    (mx, my),
                    end_pid=final_pid,
                    is_center_temp=center_temp,
                )
                self.element.execute_command(cmd)

        self.element.mark_dirty()
        return True

    def handle_text_input(self, text: str) -> bool:
        """Handle numeric input for setting circle diameter."""
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
        """Apply dimension input to preview circle and finalize shape."""
        if self._preview_state is None:
            self._dim_input.cancel()
            return

        values = self._dim_input.commit()
        if values is None or len(values) == 0:
            return

        diameter = values[0]
        if diameter is None:
            return

        registry = self.element.sketch.registry
        self._preview_state.set_diameter(registry, diameter)
        self._finalize_shape(fixed_diameter=diameter)
        self.element.mark_dirty()

    def _finalize_shape(self, fixed_diameter: Optional[float] = None):
        if self._preview_state is None:
            return

        center_id = self._preview_state.center_id
        center_temp = self._preview_state.center_temp
        radius_id = self._preview_state.radius_id

        try:
            radius_pt = self.element.sketch.registry.get_point(radius_id)
        except IndexError:
            self._preview_state = None
            self._dim_input.cancel()
            self.element.mark_dirty()
            return

        mx = radius_pt.x
        my = radius_pt.y
        CircleCommand.cleanup_preview(
            self.element.sketch.registry, self._preview_state
        )
        self._preview_state = None
        self._dim_input.cancel()
        cmd = CircleCommand(
            self.element.sketch,
            center_id,
            (mx, my),
            end_pid=None,
            is_center_temp=center_temp,
            fixed_diameter=fixed_diameter,
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
                ("0-9", _("Type diameter"), None),
            ]
        return []

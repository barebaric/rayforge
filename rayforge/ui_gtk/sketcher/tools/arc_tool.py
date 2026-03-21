from gettext import gettext as _
from typing import Callable, List, Optional, Tuple, Union

from ....core.sketcher.commands import ArcCommand, ArcPreviewState
from .base import SketchTool, SketcherKey
from .dimension_input import DimensionInputHandler


class ArcTool(SketchTool):
    """Handles creating arcs (Center -> Start -> End)."""

    ICON = "sketch-arc-symbolic"
    LABEL = _("Arc")
    SHORTCUTS = ["ga"]

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: Optional[ArcPreviewState] = None
        self._dim_input = DimensionInputHandler()

    def is_available(self, target, target_type) -> bool:
        return target is None

    def shortcut_is_active(self) -> bool:
        return True

    def get_preview_state(self) -> Optional[ArcPreviewState]:
        return self._preview_state

    def on_deactivate(self):
        """Clean up any intermediate points if the arc was not finished."""
        self._dim_input.cancel()
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

        if self._dim_input.is_active():
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
            self._dim_input.cancel()

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

    def handle_text_input(self, text: str) -> bool:
        """Handle numeric input for setting arc radius."""
        if self._preview_state is None:
            return False

        if not self._preview_state.has_start_point:
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

        if not self._preview_state.has_start_point:
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
        """Apply the dimension input to the preview arc."""
        if self._preview_state is None:
            self._dim_input.cancel()
            return

        values = self._dim_input.commit()
        if values is None or len(values) == 0:
            return

        radius = values[0]
        if radius is None:
            return

        self._preview_state.set_radius(self.element.sketch.registry, radius)
        self._finalize_shape(fixed_radius=radius)
        self.element.mark_dirty()

    def _finalize_shape(self, fixed_radius: Optional[float] = None):
        if self._preview_state is None:
            return
        if self._preview_state.start_id is None:
            return
        if self._preview_state.temp_end_id is None:
            return

        clockwise = self._preview_state.clockwise
        center_id = self._preview_state.center_id
        center_temp = self._preview_state.center_temp
        start_id = self._preview_state.start_id
        start_temp = self._preview_state.start_temp

        try:
            end_pt = self.element.sketch.registry.get_point(
                self._preview_state.temp_end_id
            )
        except IndexError:
            self._preview_state = None
            self._dim_input.cancel()
            self.element.mark_dirty()
            return

        mx = end_pt.x
        my = end_pt.y
        ArcCommand.cleanup_preview(
            self.element.sketch.registry, self._preview_state
        )
        self._preview_state = None
        self._dim_input.cancel()
        cmd = ArcCommand(
            self.element.sketch,
            center_id,
            start_id,
            (mx, my),
            end_pid=None,
            is_center_temp=center_temp,
            is_start_temp=start_temp,
            clockwise=clockwise,
            fixed_radius=fixed_radius,
        )
        self.element.execute_command(cmd)
        self.element.mark_dirty()

    def get_active_shortcuts(
        self,
    ) -> List[Tuple[Union[str, List[str]], str, Optional[Callable[[], bool]]]]:
        """Returns shortcuts for the status bar."""
        if self._preview_state is not None:
            if self._preview_state.has_start_point:
                if self._dim_input.is_active():
                    return self._dim_input.get_active_shortcuts()
                return [
                    ("0-9", _("Type radius"), None),
                ]
        return []

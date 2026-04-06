from gettext import gettext as _
from typing import Callable, List, Optional, Tuple, Union
import cairo

from ...core.commands import (
    RectangleCommand,
    RectanglePreviewState,
)
from .base import SketchTool, SketcherKey
from .dimension_input import DimensionInputHandler
from .snap_mixin import SnapMixin


class RectangleTool(SnapMixin, SketchTool):
    """Handles creating rectangles.

    - Tab: toggle magnetic snap
    """

    ICON = "sketch-rect-symbolic"
    LABEL = _("Rectangle")
    SHORTCUTS = ["gr"]
    CURSOR_ICON = "sketch-rect-symbolic"

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: Optional[RectanglePreviewState] = None
        self._dim_input = DimensionInputHandler(
            field_count=2, field_labels=[_("W"), _("H")]
        )

    def is_available(self, target, target_type) -> bool:
        return target is None

    def shortcut_is_active(self) -> bool:
        return True

    def get_preview_state(self) -> Optional[RectanglePreviewState]:
        return self._preview_state

    def on_deactivate(self):
        """Clean up if the tool is deactivated mid-creation."""
        self._dim_input.cancel()
        if self._preview_state is not None:
            start_id = self._preview_state.start_id
            start_temp = self._preview_state.start_temp

            RectangleCommand.cleanup_preview(
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

        return self._handle_click(pid_hit, mx, my)

    def on_hover_motion(self, world_x: float, world_y: float):
        """Updates the live preview of the rectangle."""
        if self._preview_state is None:
            self.clear_snap_result()
            return

        if self._dim_input.is_active():
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        preview_ids = self._preview_state.get_preview_point_ids()
        mx, my = self.query_snap_for_creation(
            self.element, mx, my, preview_ids
        )

        try:
            RectangleCommand.update_preview(
                self.element.sketch.registry, self._preview_state, mx, my
            )
            self.element.mark_dirty()
        except (IndexError, KeyError):
            self.on_deactivate()

    def draw_overlay(self, ctx: cairo.Context):
        """Draw snap feedback during creation."""
        if self._preview_state is not None:
            self.draw_snap_feedback(ctx, self.element)

    def _handle_click(
        self, pid_hit: Optional[int], mx: float, my: float
    ) -> bool:
        if self._preview_state is None:
            self._preview_state = RectangleCommand.start_preview(
                self.element.sketch.registry, mx, my, snapped_pid=pid_hit
            )
            self.element.preview_changed.send(self.element)
        else:
            preview_ids = self._preview_state.get_preview_point_ids()
            start_id = self._preview_state.start_id
            start_temp = self._preview_state.start_temp

            RectangleCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )
            self._preview_state = None
            self.element.preview_changed.send(self.element)
            self._dim_input.cancel()

            final_pid = None if pid_hit in preview_ids else pid_hit

            cmd = RectangleCommand(
                self.element.sketch,
                start_id,
                (mx, my),
                end_pid=final_pid,
                is_start_temp=start_temp,
            )
            self.element.execute_command(cmd)

            self.clear_snap_result()

        self.element.mark_dirty()
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def handle_text_input(self, text: str) -> bool:
        """Handle numeric input for setting rectangle dimensions."""
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

        if key == SketcherKey.TAB:
            if self._dim_input.is_active():
                handled, should_apply, committed_field = (
                    self._dim_input.handle_tab(shift=shift)
                )
                if should_apply:
                    self._apply_dimension_input()
                elif handled and committed_field is not None:
                    self._apply_field_constraint(committed_field)
                    self.element.mark_dirty()
                elif handled:
                    self.element.mark_dirty()
                return True
            else:
                self.toggle_magnetic_snap()
                return True

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

    def _apply_field_constraint(self, field_index: int) -> None:
        """Apply constraint for a specific field when Tabbing out of it."""
        if self._preview_state is None:
            return

        value = self._dim_input.get_field_value(field_index)
        if value is None:
            return

        registry = self.element.sketch.registry
        if field_index == 0:
            self._preview_state.set_dimensions(registry, width=value)
        elif field_index == 1:
            self._preview_state.set_dimensions(registry, height=value)

    def _apply_dimension_input(self):
        """Apply the dimension input to the preview rectangle."""
        if self._preview_state is None:
            self._dim_input.cancel()
            return

        values = self._dim_input.commit()
        if values is None:
            return

        registry = self.element.sketch.registry
        width = values[0] if len(values) > 0 else None
        height = values[1] if len(values) > 1 else None

        if width is not None or height is not None:
            self._preview_state.set_dimensions(
                registry, width=width, height=height
            )
        self._finalize_shape(fixed_width=width, fixed_height=height)
        self.element.mark_dirty()

    def _finalize_shape(
        self,
        fixed_width: Optional[float] = None,
        fixed_height: Optional[float] = None,
    ):
        if self._preview_state is None:
            return
        start_id = self._preview_state.start_id
        start_temp = self._preview_state.start_temp
        try:
            end_pt = self.element.sketch.registry.get_point(
                self._preview_state.p_end_id
            )
        except IndexError:
            self._preview_state = None
            self.element.preview_changed.send(self.element)
            self._dim_input.cancel()
            self.element.mark_dirty()
            return
        mx = end_pt.x
        my = end_pt.y
        RectangleCommand.cleanup_preview(
            self.element.sketch.registry, self._preview_state
        )
        self._preview_state = None
        self.element.preview_changed.send(self.element)
        self._dim_input.cancel()
        cmd = RectangleCommand(
            self.element.sketch,
            start_id,
            (mx, my),
            end_pid=None,
            is_start_temp=start_temp,
            fixed_width=fixed_width,
            fixed_height=fixed_height,
        )
        self.element.execute_command(cmd)
        self.clear_snap_result()
        self.element.mark_dirty()

    def get_active_shortcuts(
        self,
    ) -> List[Tuple[Union[str, List[str]], str, Optional[Callable[[], bool]]]]:
        """Returns shortcuts for the status bar."""
        if self._preview_state is not None:
            if self._dim_input.is_active():
                return self._dim_input.get_active_shortcuts()
            return [
                ("0-9", _("Type dimensions (W H)"), None),
                ("Tab", _("Toggle Magnetic Snap"), None),
            ]
        return []

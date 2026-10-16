import math
from collections.abc import Callable
from gettext import gettext as _
from typing import ClassVar

import cairo

from ...core.commands import PolygonCommand, PolygonPreviewState
from ...core.commands.polygon import (
    DEFAULT_SIDES,
    DEFAULT_STAR_RATIO,
    check_shape,
)
from .base import SketcherKey, SketchTool
from .dimension_input import DimensionInputHandler
from .snap_mixin import SnapMixin

RADIUS_FIELD = 0
SIDES_FIELD = 1
STAR_FIELD = 2


class PolygonTool(SnapMixin, SketchTool):
    """Handles creating regular polygons and stars.

    - Default: click the center -> move -> click (drag also works); the
      pointer sets the radius and the rotation of the first corner
    - Ctrl: make it a star
    - Type R, corner count and star ratio; Tab moves between fields
    - Tab: toggle magnetic snap
    """

    ICON = "sketch-polygon-symbolic"
    LABEL = _("Polygon")
    SHORTCUTS: ClassVar[list[str]] = ["gx"]
    CURSOR_ICON = "sketch-polygon-symbolic"
    PIE_GROUP: ClassVar[str | None] = "rectangle"

    DRAG_THRESHOLD = 2.0

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: PolygonPreviewState | None = None
        self._dim_input = DimensionInputHandler(
            field_count=3, field_labels=[_("R"), _("N"), _("Star")]
        )
        self._sides = DEFAULT_SIDES
        self._typed_star_ratio: float | None = None
        self._ctrl_held = False
        self._press_world_pos: tuple[float, float] | None = None
        self._in_press = False

    def is_available(self, target, target_type) -> bool:
        return target is None

    def shortcut_is_active(self) -> bool:
        return True

    def get_preview_state(self) -> PolygonPreviewState | None:
        return self._preview_state

    def _star_ratio(self) -> float | None:
        if self._typed_star_ratio is not None:
            return self._typed_star_ratio
        return DEFAULT_STAR_RATIO if self._ctrl_held else None

    def _discard_preview(self) -> None:
        if self._preview_state is None:
            return
        center_id = self._preview_state.center_id
        center_temp = self._preview_state.center_temp
        PolygonCommand.cleanup_preview(
            self.element.sketch.registry, self._preview_state
        )
        self._preview_state = None
        self.element.preview_changed.send(self.element)
        if center_temp:
            self.element.remove_point_if_unused(center_id)

    def on_deactivate(self):
        """Clean up if the tool is deactivated mid-creation."""
        self._dim_input.cancel()
        self._press_world_pos = None
        self._in_press = False
        self._typed_star_ratio = None
        self._discard_preview()
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
        if self._preview_state is None:
            self._preview_state = PolygonCommand.start_preview(
                self.element.sketch.registry,
                mx,
                my,
                snapped_pid=self.get_snapped_point_id(),
                sides=self._sides,
                star_ratio=self._star_ratio(),
            )
            self._press_world_pos = (world_x, world_y)
            self._in_press = True
            self.element.preview_changed.send(self.element)
            self.element.mark_dirty()
        else:
            self._commit(end_pid=self._snapped_corner_id())
        return True

    def _snapped_corner_id(self) -> int | None:
        """The snapped point the first corner lands on, if any."""
        pid = self.get_snapped_point_id()
        state = self._preview_state
        if state is None or pid in state.get_preview_point_ids():
            return None
        return pid

    def on_release(self, world_x: float, world_y: float):
        self._in_press = False
        if self._preview_state is None or self._press_world_pos is None:
            return
        dx = world_x - self._press_world_pos[0]
        dy = world_y - self._press_world_pos[1]
        if math.hypot(dx, dy) < self.DRAG_THRESHOLD:
            return
        self._commit(end_pid=self._snapped_corner_id())

    def on_hover_motion(self, world_x: float, world_y: float):
        """Updates the live preview of the polygon."""
        if self._preview_state is None:
            self.clear_snap_result()
            return
        if self._dim_input.is_active():
            return
        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )
        mx, my = self.query_snap_for_creation(
            self.element, mx, my, self._preview_state.get_preview_point_ids()
        )
        try:
            PolygonCommand.update_preview(
                self.element.sketch.registry, self._preview_state, mx, my
            )
            self.element.mark_dirty()
        except (IndexError, KeyError):
            self.on_deactivate()

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_modifier_change(self, shift: bool = False, ctrl: bool = False):
        """Ctrl turns the preview into a star while it is held."""
        if self._ctrl_held == ctrl:
            return
        self._ctrl_held = ctrl
        if self._preview_state is None:
            return
        self._preview_state.set_shape(
            self.element.sketch.registry, self._sides, self._star_ratio()
        )
        self.element.mark_dirty()

    def draw_overlay(self, ctx: cairo.Context):
        """Draw snap feedback during creation."""
        if self._preview_state is not None:
            self.draw_snap_feedback(ctx, self.element)

    def _commit(
        self,
        fixed_radius: float | None = None,
        end_pid: int | None = None,
    ) -> None:
        """Create the polygon the preview shows."""
        state = self._preview_state
        if state is None:
            return
        registry = self.element.sketch.registry
        try:
            first = registry.get_point(state.radius_id)
        except IndexError:
            self.on_deactivate()
            return
        end_pos = (first.x, first.y)
        sides, star_ratio = state.sides, state.star_ratio
        PolygonCommand.cleanup_preview(registry, state)
        self._preview_state = None
        self._press_world_pos = None
        self._in_press = False
        self._typed_star_ratio = None
        self._dim_input.cancel()
        self.element.preview_changed.send(self.element)
        cmd = PolygonCommand(
            self.element.sketch,
            state.center_id,
            end_pos,
            end_pid=end_pid,
            sides=sides,
            star_ratio=star_ratio,
            is_center_temp=state.center_temp,
            fixed_radius=fixed_radius,
        )
        self.element.execute_command(cmd)
        self.element.mark_dirty()
        self.clear_snap_result()

    def handle_text_input(self, text: str) -> bool:
        """Handle numeric input for radius, corner count and star ratio."""
        if self._preview_state is None:
            return False
        if not self._dim_input.is_active():
            self._dim_input.start()
        handled = self._dim_input.handle_text_input(text)
        if handled:
            self.element.mark_dirty()
        return handled

    def _apply_field(self, field_index: int, value: float | None) -> None:
        """Apply one typed value to the preview; invalid ones are ignored."""
        state = self._preview_state
        if state is None or value is None:
            return
        registry = self.element.sketch.registry
        if field_index == RADIUS_FIELD and value > 0:
            state.set_radius(registry, value)
            return
        sides, star_ratio = state.sides, state.star_ratio
        if field_index == SIDES_FIELD:
            sides = round(value)
        elif field_index == STAR_FIELD:
            star_ratio = value
        try:
            check_shape(sides, star_ratio)
        except ValueError:
            return
        self._sides = sides
        if field_index == STAR_FIELD:
            self._typed_star_ratio = star_ratio
        state.set_shape(registry, sides, star_ratio)

    def _apply_dimension_input(self) -> None:
        """Apply every typed value and create the polygon."""
        values = self._dim_input.commit()
        if values is None or self._preview_state is None:
            return
        for index in (SIDES_FIELD, STAR_FIELD, RADIUS_FIELD):
            if index < len(values):
                self._apply_field(index, values[index])
        state = self._preview_state
        fixed_radius = state.locked_radius if state is not None else None
        self._commit(fixed_radius=fixed_radius)

    def _handle_tab(self, shift: bool) -> bool:
        if not self._dim_input.is_active():
            self.toggle_magnetic_snap()
            return True
        handled, should_apply, field = self._dim_input.handle_tab(shift=shift)
        if should_apply:
            self._apply_dimension_input()
        elif handled and field is not None:
            self._apply_field(field, self._dim_input.get_field_value(field))
        self.element.mark_dirty()
        return True

    def handle_key_event(
        self, key: SketcherKey, shift: bool = False, ctrl: bool = False
    ) -> bool:
        """Handle special keys for dimension input."""
        if self._preview_state is None:
            return False
        if key == SketcherKey.TAB:
            return self._handle_tab(shift)
        if not self._dim_input.is_active():
            return False
        if key == SketcherKey.BACKSPACE:
            self._dim_input.handle_backspace()
        elif key == SketcherKey.DELETE:
            self._dim_input.handle_delete()
        elif key == SketcherKey.RETURN:
            self._apply_dimension_input()
        elif key == SketcherKey.ESCAPE:
            self._dim_input.cancel()
        else:
            return False
        self.element.mark_dirty()
        return True

    def get_active_shortcuts(
        self,
    ) -> list[tuple[str | list[str], str, Callable[[], bool] | None]]:
        """Returns shortcuts for the status bar."""
        if self._preview_state is None:
            return []
        if self._dim_input.is_active():
            return self._dim_input.get_active_shortcuts()
        shortcuts: list[
            tuple[str | list[str], str, Callable[[], bool] | None]
        ] = [
            ("Ctrl", _("Star"), None),
            ("0-9", _("Type radius, corners and star ratio (R N Star)"), None),
            ("Tab", _("Toggle Magnetic Snap"), None),
        ]
        if not self._in_press:
            shortcuts.append(("Click", _("Set first corner"), None))
        return shortcuts

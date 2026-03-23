from gettext import gettext as _
import logging
from typing import Callable, List, Optional, Tuple, Union
from rayforge.ui_gtk.shared.keyboard import PRIMARY_KEY_NAME
from ...core.commands import BezierCommand, BezierPreviewState
from ...core.entities import Bezier
from .base import SketchTool

logger = logging.getLogger(__name__)


DRAG_THRESHOLD = 2.0


class PathTool(SketchTool):
    """
    Handles creating lines and bezier curves with a unified workflow.

    Workflow:
    - Click once: starts line preview from start point
    - Hover: live preview of line segment
    - Second click without drag: creates line segment, starts next preview
    - Second click with drag: creates bezier where drag controls the "bow"
    """

    ICON = "sketch-bezier-symbolic"
    LABEL = _("Path")
    SHORTCUTS = ["gp", "gl"]
    ACTION_SHORTCUT = "l"
    CURSOR_ICON = "sketch-line-symbolic"

    def __init__(self, element):
        super().__init__(element)
        self._preview_state: Optional[BezierPreviewState] = None
        self._press_pos: Optional[Tuple[float, float]] = None
        self._waypoint_model_pos: Optional[Tuple[float, float]] = None
        self._snapped_pid: Optional[int] = None
        self._dragging: bool = False
        self._in_press: bool = False
        self._mirror_cp_offset: Optional[Tuple[float, float]] = None
        self._release_handled: bool = False
        self.hovered_point_id: Optional[int] = None

    def is_available(self, target, target_type) -> bool:
        return target is None

    def shortcut_is_active(self) -> bool:
        return True

    def get_active_shortcuts(
        self,
    ) -> List[Tuple[Union[str, List[str]], str, Optional[Callable[[], bool]]]]:
        return [
            (PRIMARY_KEY_NAME, _("Snap to Grid"), lambda: self._dragging),
            (
                "Shift",
                _("Constrain to Axis"),
                lambda: self._preview_state is not None,
            ),
        ]

    def get_preview_state(self) -> Optional[BezierPreviewState]:
        return self._preview_state

    def on_deactivate(self):
        """Clean up if the tool is deactivated mid-creation."""
        logger.debug(
            f"on_deactivate: preview_state={self._preview_state is not None}"
        )
        if self._preview_state is not None:
            start_id = self._preview_state.start_id
            start_temp = self._preview_state.start_temp

            BezierCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )
            self._preview_state = None

            if start_temp:
                self.element.remove_point_if_unused(start_id)

            self.element.mark_dirty()

        self._press_pos = None
        self._waypoint_model_pos = None
        self._snapped_pid = None
        self._dragging = False
        self._in_press = False
        self._mirror_cp_offset = None
        self._release_handled = False
        self.hovered_point_id = None

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )
        hit_type, hit_obj = self.element.hittester.get_hit_data(
            world_x, world_y, self.element
        )
        pid_hit = hit_obj if hit_type == "point" else None

        logger.debug(
            f"on_press: preview_state={self._preview_state is not None}, "
            f"snapped={pid_hit}"
        )

        if (
            self._preview_state is not None
            and pid_hit is not None
            and pid_hit not in self._preview_state.get_preview_point_ids()
        ):
            self._snapped_pid = pid_hit
            try:
                snapped_pt = self.element.sketch.registry.get_point(pid_hit)
                BezierCommand.update_preview(
                    self.element.sketch.registry,
                    self._preview_state,
                    snapped_pt.x,
                    snapped_pt.y,
                )
            except (IndexError, KeyError):
                pass

            is_line = self._preview_state.is_line_preview
            if is_line:
                self._finalize_line_segment()
            else:
                self._finalize_bezier_segment()

            if self._preview_state is not None:
                BezierCommand.cleanup_preview(
                    self.element.sketch.registry, self._preview_state
                )
                self._preview_state = None

            self._press_pos = None
            self._waypoint_model_pos = None
            self._snapped_pid = None
            self._dragging = False
            self._mirror_cp_offset = None
            self.hovered_point_id = None
            self.element.mark_dirty()
            return False

        self._press_pos = (world_x, world_y)
        self._waypoint_model_pos = (mx, my)
        self._snapped_pid = pid_hit
        self._dragging = False
        self._in_press = True
        self._release_handled = False

        if self._preview_state is None:
            self._preview_state = BezierCommand.start_preview(
                self.element.sketch.registry, mx, my, snapped_pid=pid_hit
            )
            logger.debug(
                f"start_preview created: entity_id="
                f"{self._preview_state.temp_entity_id}"
            )

        self.element.mark_dirty()
        return False

    def _constrain_to_axis(self, mx: float, my: float) -> Tuple[float, float]:
        """Constrain model position to horizontal or vertical from start."""
        if self._preview_state is None:
            return mx, my
        try:
            start_pt = self.element.sketch.registry.get_point(
                self._preview_state.start_id
            )
            if start_pt:
                dx = mx - start_pt.x
                dy = my - start_pt.y
                if abs(dx) > abs(dy):
                    return mx, start_pt.y
                else:
                    return start_pt.x, my
        except IndexError:
            pass
        return mx, my

    def on_drag(self, world_dx: float, world_dy: float):
        ps = self._preview_state
        is_line = ps.is_line_preview if ps else None
        logger.debug(
            f"on_drag: press_pos={self._press_pos}, "
            f"preview_state={self._preview_state is not None}, "
            f"dragging={self._dragging}, is_line_preview={is_line}"
        )
        if self._press_pos is None or self._preview_state is None:
            return

        if self._waypoint_model_pos is None:
            return

        current_x = self._press_pos[0] + world_dx
        current_y = self._press_pos[1] + world_dy

        mx, my = self.element.hittester.screen_to_model(
            current_x, current_y, self.element
        )

        if self.element.canvas and self.element.canvas._shift_pressed:
            mx, my = self._constrain_to_axis(mx, my)

        if not self._preview_state.is_line_preview:
            self._dragging = True
            logger.debug(
                f"on_drag (bezier): calling update_control_point "
                f"at ({mx:.1f}, {my:.1f})"
            )
            BezierCommand.update_control_point(
                self.element.sketch.registry,
                self._preview_state,
                mx,
                my,
            )
            self.element.mark_dirty()
            return

        if self._preview_state.end_id is None:
            return

        start_pt = self.element.sketch.registry.get_point(
            self._preview_state.start_id
        )
        end_pt = self.element.sketch.registry.get_point(
            self._preview_state.end_id
        )
        if start_pt and end_pt:
            dist_sq = (start_pt.x - end_pt.x) ** 2 + (
                start_pt.y - end_pt.y
            ) ** 2
            has_virtual_cp = self._mirror_cp_offset is not None
            if dist_sq < 1.0 and not has_virtual_cp:
                return

        dx = current_x - self._press_pos[0]
        dy = current_y - self._press_pos[1]
        dist_sq = dx * dx + dy * dy

        if dist_sq < DRAG_THRESHOLD * DRAG_THRESHOLD:
            return

        if not self._dragging:
            self._dragging = True

            logger.debug(
                f"convert_to_bezier: waypoint="
                f"({self._waypoint_model_pos[0]:.1f}, "
                f"{self._waypoint_model_pos[1]:.1f}), "
                f"drag=({mx:.1f}, {my:.1f}), "
                f"mirror={self._mirror_cp_offset}"
            )
            BezierCommand.convert_to_bezier(
                self.element.sketch.registry,
                self._preview_state,
                self._waypoint_model_pos[0],
                self._waypoint_model_pos[1],
                mx,
                my,
                mirror_cp_offset=self._mirror_cp_offset,
            )
        else:
            logger.debug(
                f"on_drag: calling update_control_point "
                f"at ({mx:.1f}, {my:.1f})"
            )
            BezierCommand.update_control_point(
                self.element.sketch.registry,
                self._preview_state,
                mx,
                my,
            )

        self.element.mark_dirty()

    def on_release(self, world_x: float, world_y: float):
        if self._release_handled:
            return
        self._release_handled = True

        self._in_press = False
        logger.debug(
            f"on_release: preview_state={self._preview_state is not None}, "
            f"dragging={self._dragging}"
        )

        if self._preview_state is None:
            self._press_pos = None
            self._waypoint_model_pos = None
            self._snapped_pid = None
            self._dragging = False
            return

        endpoint_moved = self._end_point_moved()
        is_line = self._preview_state.is_line_preview
        logger.debug(
            f"on_release: endpoint_moved={endpoint_moved}, "
            f"is_line_preview={is_line}"
        )

        if not is_line:
            if endpoint_moved:
                logger.debug("on_release: calling _finalize_bezier_segment")
                self._finalize_bezier_segment()
            else:
                logger.debug(
                    "on_release: bezier preview, endpoint not moved, "
                    "keeping preview"
                )
        elif self._dragging:
            logger.debug(
                "on_release: dragging, calling _finalize_bezier_segment"
            )
            self._finalize_bezier_segment()
        else:
            if endpoint_moved:
                logger.debug("on_release: calling _finalize_line_segment")
                self._finalize_line_segment()

        self._press_pos = None
        self._waypoint_model_pos = None
        self._snapped_pid = None
        self._dragging = False
        self.element.mark_dirty()

    def _end_point_moved(self) -> bool:
        """Check if the preview end point has moved from the start position."""
        if self._preview_state is None:
            return False
        if self._preview_state.end_id is None:
            return False

        try:
            start_pt = self.element.sketch.registry.get_point(
                self._preview_state.start_id
            )
            end_pt = self.element.sketch.registry.get_point(
                self._preview_state.end_id
            )
            dx = end_pt.x - start_pt.x
            dy = end_pt.y - start_pt.y
            dist_sq = dx * dx + dy * dy
            return dist_sq > 0.01
        except IndexError:
            return False

    def _finalize_line_segment(self):
        """Finalize the current segment and start a new preview."""
        if self._preview_state is None:
            return

        preview_ids = self._preview_state.get_preview_point_ids()
        start_id = self._preview_state.start_id
        start_temp = self._preview_state.start_temp
        end_id = self._preview_state.end_id

        if end_id is None:
            return

        try:
            end_pt = self.element.sketch.registry.get_point(end_id)
            end_x, end_y = end_pt.x, end_pt.y
        except (IndexError, AttributeError):
            return

        final_pid = None
        if (
            self._snapped_pid is not None
            and self._snapped_pid not in preview_ids
        ):
            final_pid = self._snapped_pid
            try:
                snapped_pt = self.element.sketch.registry.get_point(final_pid)
                end_x, end_y = snapped_pt.x, snapped_pt.y
            except IndexError:
                final_pid = None

        if final_pid == start_id:
            self._press_pos = None
            self._waypoint_model_pos = None
            self._snapped_pid = None
            self._dragging = False
            self.element.mark_dirty()
            return

        has_virtual_cp = self._mirror_cp_offset is not None

        BezierCommand.cleanup_preview(
            self.element.sketch.registry, self._preview_state
        )
        self._preview_state = None
        self._mirror_cp_offset = None

        cmd = BezierCommand(
            self.element.sketch,
            start_id,
            (end_x, end_y),
            end_pid=final_pid,
            is_start_temp=start_temp,
            is_line=not has_virtual_cp,
        )
        self.element.execute_command(cmd)

        if cmd.committed_end_id is not None:
            try:
                new_start_pt = self.element.sketch.registry.get_point(
                    cmd.committed_end_id
                )
                self._preview_state = BezierCommand.start_preview(
                    self.element.sketch.registry,
                    new_start_pt.x,
                    new_start_pt.y,
                    snapped_pid=cmd.committed_end_id,
                )
            except IndexError:
                pass

    def _finalize_bezier_segment(self):
        """Finalize the current segment as a bezier and start a new preview."""
        if self._preview_state is None:
            return

        if self._preview_state.is_line_preview:
            return

        if not self._end_point_moved():
            return

        preview_ids = self._preview_state.get_preview_point_ids()
        start_id = self._preview_state.start_id
        start_temp = self._preview_state.start_temp
        end_id = self._preview_state.end_id

        if end_id is None:
            return

        try:
            end_pt = self.element.sketch.registry.get_point(end_id)
            end_x, end_y = end_pt.x, end_pt.y
        except (IndexError, AttributeError):
            return

        final_pid = None
        if (
            self._snapped_pid is not None
            and self._snapped_pid not in preview_ids
        ):
            final_pid = self._snapped_pid
            try:
                snapped_pt = self.element.sketch.registry.get_point(final_pid)
                end_x, end_y = snapped_pt.x, snapped_pt.y
            except IndexError:
                final_pid = None

        if final_pid == start_id:
            BezierCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )
            self._preview_state = None
            self._press_pos = None
            self._waypoint_model_pos = None
            self._snapped_pid = None
            self._dragging = False
            self.element.mark_dirty()
            return

        temp_entity_id = self._preview_state.temp_entity_id
        temp_entity = None
        cp1 = None
        cp2 = None
        if temp_entity_id is not None:
            temp_entity = self.element.sketch.registry.get_entity(
                temp_entity_id
            )
        if isinstance(temp_entity, Bezier):
            cp1 = temp_entity.cp1
            cp2 = temp_entity.cp2

        self._mirror_cp_offset = self._preview_state.virtual_cp
        logger.debug(
            f"_finalize_bezier_segment: cp1={cp1}, cp2={cp2}, "
            f"virtual_cp={self._preview_state.virtual_cp}, "
            f"mirror={self._mirror_cp_offset}"
        )

        BezierCommand.cleanup_preview(
            self.element.sketch.registry, self._preview_state
        )
        self._preview_state = None

        cmd = BezierCommand(
            self.element.sketch,
            start_id,
            (end_x, end_y),
            end_pid=final_pid,
            is_start_temp=start_temp,
            is_line=False,
            cp1=cp1,
            cp2=cp2,
        )
        self.element.execute_command(cmd)

        if cmd.committed_end_id is not None:
            try:
                new_start_pt = self.element.sketch.registry.get_point(
                    cmd.committed_end_id
                )
                self._preview_state = BezierCommand.start_preview(
                    self.element.sketch.registry,
                    new_start_pt.x,
                    new_start_pt.y,
                    snapped_pid=cmd.committed_end_id,
                    virtual_cp=self._mirror_cp_offset,
                )
            except IndexError:
                pass

    def on_hover_motion(self, world_x: float, world_y: float):
        """Updates the live preview of the line/bezier."""
        if self._preview_state is None:
            return

        if self._in_press:
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        if self.element.canvas and self.element.canvas._shift_pressed:
            mx, my = self._constrain_to_axis(mx, my)

        hit_type, hit_obj = self.element.hittester.get_hit_data(
            world_x, world_y, self.element
        )

        new_hovered_pid = None
        if hit_type == "point":
            pid = hit_obj
            preview_ids = self._preview_state.get_preview_point_ids()
            if pid not in preview_ids:
                new_hovered_pid = pid

        if self.hovered_point_id != new_hovered_pid:
            self.hovered_point_id = new_hovered_pid
            self.element.mark_dirty()

        try:
            if new_hovered_pid is not None:
                snapped_pt = self.element.sketch.registry.get_point(
                    new_hovered_pid
                )
                if snapped_pt:
                    BezierCommand.update_preview(
                        self.element.sketch.registry,
                        self._preview_state,
                        snapped_pt.x,
                        snapped_pt.y,
                    )
            else:
                BezierCommand.update_preview(
                    self.element.sketch.registry, self._preview_state, mx, my
                )
            self.element.mark_dirty()
        except (IndexError, KeyError):
            self.on_deactivate()

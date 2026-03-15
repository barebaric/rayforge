from typing import Optional

from ....core.sketcher.commands import (
    ArcCommand,
    ArcPreviewState,
)
from .base import SketchTool


class ArcTool(SketchTool):
    """Handles creating arcs (Center -> Start -> End)."""

    def __init__(self, element):
        super().__init__(element)
        self.center_id: Optional[int] = None
        self.start_id: Optional[int] = None
        self.center_temp: bool = False
        self.start_temp: bool = False
        self._preview_state: Optional[ArcPreviewState] = None

    def on_deactivate(self):
        """Clean up any intermediate points if the arc was not finished."""
        if self._preview_state is not None:
            ArcCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )
            self._preview_state = None

        if self.start_temp:
            self.element.remove_point_if_unused(self.start_id)
        if self.center_temp:
            self.element.remove_point_if_unused(self.center_id)

        self.center_id = None
        self.start_id = None
        self.center_temp = False
        self.start_temp = False
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

        if self.center_id is not None:
            try:
                self.element.sketch.registry.get_point(self.center_id)
            except IndexError:
                # Center point was deleted, reset the tool completely
                self.on_deactivate()

        if self.start_id is not None:
            try:
                self.element.sketch.registry.get_point(self.start_id)
            except IndexError:
                # Start point was deleted, reset to expecting start point
                self.start_id = None
                self.start_temp = False
                if self._preview_state is not None:
                    ArcCommand.cleanup_preview(
                        self.element.sketch.registry, self._preview_state
                    )
                    self._preview_state = None

        if self.center_id is None:
            # Step 1: Center Point
            if pid_hit is None:
                pid_hit = self.element.sketch.add_point(mx, my)
                self.center_temp = True
                self.element.update_bounds_from_sketch()
            else:
                self.center_temp = False

            self.center_id = pid_hit
            self.element.selection.clear()
            self.element.selection.select_point(pid_hit, False)

        elif self.start_id is None:
            # Step 2: Start Point
            if pid_hit is None:
                pid_hit = self.element.sketch.add_point(mx, my)
                self.start_temp = True
                self.element.update_bounds_from_sketch()
            else:
                self.start_temp = False

            # Cannot start where center is
            if pid_hit != self.center_id:
                self.start_id = pid_hit
                self.element.selection.select_point(pid_hit, True)

                # Create a temporary End point and Arc entity to visualize
                # dragging
                self._preview_state = ArcCommand.start_preview(
                    self.element.sketch.registry,
                    mx,
                    my,
                    center_id=self.center_id,
                    center_temp=self.center_temp,
                    start_id=self.start_id,
                    start_temp=self.start_temp,
                )

        else:
            # Step 3: End Point (Finalize)
            if self._preview_state is None:
                return False

            preview_end_id = self._preview_state.temp_end_id
            clockwise = self._preview_state.clockwise
            center_id = self._preview_state.center_id
            center_temp = self._preview_state.center_temp
            start_id = self._preview_state.start_id
            start_temp = self._preview_state.start_temp

            ArcCommand.cleanup_preview(
                self.element.sketch.registry, self._preview_state
            )

            self._preview_state = None

            if pid_hit == preview_end_id:
                pid_hit = None

            cmd = ArcCommand(
                self.element.sketch,
                center_id,
                start_id,
                (mx, my),
                end_pid=pid_hit,
                is_center_temp=center_temp,
                is_start_temp=start_temp,
                clockwise=clockwise,
            )
            self.element.execute_command(cmd)

            # Reset tool state
            self.center_id = None
            self.start_id = None
            self.center_temp = False
            self.start_temp = False

            # Select the last point
            self.element.selection.clear()
            if cmd.end_pid is not None:
                self.element.selection.select_point(cmd.end_pid, False)
            elif cmd.add_cmd and cmd.add_cmd.points:
                self.element.selection.select_point(
                    cmd.add_cmd.points[-1].id, False
                )

        self.element.mark_dirty()
        return True

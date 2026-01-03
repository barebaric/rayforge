from typing import Optional
from ..commands import AddItemsCommand
from ..entities import Point, Circle
from .base import SketchTool


class CircleTool(SketchTool):
    """Handles creating circles (Center -> Radius Point)."""

    def __init__(self, element):
        super().__init__(element)
        self.center_id: Optional[int] = None
        self.center_temp: bool = False

        # Live Preview State
        self.temp_radius_id: Optional[int] = None
        self.temp_entity_id: Optional[int] = None

    def _cleanup_temps(self):
        """Removes temporary preview entities from the registry."""
        if self.temp_entity_id is not None:
            # Remove entity
            self.element.sketch.registry.entities = [
                e
                for e in self.element.sketch.registry.entities
                if e.id != self.temp_entity_id
            ]
            # Rebuild map
            reg = self.element.sketch.registry
            reg._entity_map = {e.id: e for e in reg.entities}
            self.temp_entity_id = None

        if self.temp_radius_id is not None:
            self.element.remove_point_if_unused(self.temp_radius_id)
            self.temp_radius_id = None

    def on_deactivate(self):
        """Clean up the center point if a circle was not finished."""
        self._cleanup_temps()
        if self.center_temp:
            self.element.remove_point_if_unused(self.center_id)
        self.center_id = None
        self.center_temp = False
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
        if (
            self.center_id is None
            or self.temp_entity_id is None
            or self.temp_radius_id is None
        ):
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        try:
            r_pt = self.element.sketch.registry.get_point(self.temp_radius_id)
            r_pt.x = mx
            r_pt.y = my
            self.element.mark_dirty()
        except IndexError:
            pass

    def _handle_click(self, pid_hit, mx, my) -> bool:
        # State machine: Center -> Radius Point

        if self.center_id is not None:
            try:
                self.element.sketch.registry.get_point(self.center_id)
            except IndexError:
                # Center point was deleted, reset the tool
                self.on_deactivate()

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

            # Create a temporary Radius point and Circle entity
            self.temp_radius_id = self.element.sketch.add_point(mx, my)
            self.temp_entity_id = self.element.sketch.add_circle(
                self.center_id, self.temp_radius_id
            )

        else:
            # Step 2: Radius Point (Finalize)

            # If we hit our own preview point, ignore the hit so we create a
            # new, real point
            if pid_hit == self.temp_radius_id:
                pid_hit = None

            # Clean up the preview geometry before adding the real command
            self._cleanup_temps()

            new_point = None
            if pid_hit is None:
                temp_id = self.element.sketch.registry._id_counter
                pid_hit = temp_id
                new_point = Point(temp_id, mx, my)

            # Cannot have radius point at center
            if pid_hit != self.center_id:
                temp_circle_id = self.element.sketch.registry._id_counter + (
                    1 if new_point else 0
                )
                new_circle = Circle(temp_circle_id, self.center_id, pid_hit)

                points_to_add = [new_point] if new_point else []

                if self.center_temp:
                    try:
                        p = self.element.sketch.registry.get_point(
                            self.center_id
                        )
                        self.element.sketch.registry.points.remove(p)
                        points_to_add.append(p)
                    except (IndexError, ValueError):
                        pass

                cmd = AddItemsCommand(
                    self.element.sketch,
                    _("Add Circle"),
                    points=points_to_add,
                    entities=[new_circle],
                )
                self.element.execute_command(cmd)

                # Reset for next circle
                self.center_id = None
                self.center_temp = False
                self.element.selection.clear()
                self.element.selection.select_point(pid_hit, False)

        self.element.mark_dirty()
        return True

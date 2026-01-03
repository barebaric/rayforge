import math
from typing import Optional
from ..commands import AddItemsCommand
from ..constraints import EqualDistanceConstraint
from ..entities import Point, Arc
from .base import SketchTool


class ArcTool(SketchTool):
    """Handles creating arcs (Center -> Start -> End)."""

    def __init__(self, element):
        super().__init__(element)
        self.center_id: Optional[int] = None
        self.start_id: Optional[int] = None
        self.center_temp: bool = False
        self.start_temp: bool = False

        # Live Preview State
        self.temp_end_id: Optional[int] = None
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

        if self.temp_end_id is not None:
            self.element.remove_point_if_unused(self.temp_end_id)
            self.temp_end_id = None

    def on_deactivate(self):
        """Clean up any intermediate points if the arc was not finished."""
        self._cleanup_temps()

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
        # Only update if we are in the final stage (drawing the arc curve)
        if (
            self.center_id is None
            or self.start_id is None
            or self.temp_entity_id is None
            or self.temp_end_id is None
        ):
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        try:
            c = self.element.sketch.registry.get_point(self.center_id)
            s = self.element.sketch.registry.get_point(self.start_id)
            e = self.element.sketch.registry.get_point(self.temp_end_id)
            arc_ent = self.element.sketch.registry.get_entity(
                self.temp_entity_id
            )
        except IndexError:
            return

        if not isinstance(arc_ent, Arc):
            return

        # 1. Project mouse position onto the circle defined by Center-Start
        radius = math.hypot(s.x - c.x, s.y - c.y)
        curr_dist = math.hypot(mx - c.x, my - c.y)

        if curr_dist > 1e-9:
            scale = radius / curr_dist
            final_x = c.x + (mx - c.x) * scale
            final_y = c.y + (my - c.y) * scale
        else:
            final_x, final_y = mx, my

        e.x = final_x
        e.y = final_y

        # 2. Determine Winding (Clockwise/Counter-Clockwise)
        # We determine direction based on the cross product of
        # Vector(Center->Start) and Vector(Center->Mouse).
        # This allows reversing direction by passing back through the start
        # line.
        vec_s_x, vec_s_y = s.x - c.x, s.y - c.y
        vec_m_x, vec_m_y = mx - c.x, my - c.y

        # 2D Cross Product: A_x * B_y - A_y * B_x
        det = vec_s_x * vec_m_y - vec_s_y * vec_m_x

        # In standard Y-Up math, det < 0 is CW.
        # In Y-Down (Screen), det > 0 is CW.
        # The Sketcher coordinate system behavior depends on transforms,
        # but typically det < 0 maps to "Clockwise" property for arcs in CAD.
        arc_ent.clockwise = bool(det < 0)

        self.element.mark_dirty()

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
                self._cleanup_temps()

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
                self.temp_end_id = self.element.sketch.add_point(mx, my)
                self.temp_entity_id = self.element.sketch.add_arc(
                    self.start_id, self.temp_end_id, self.center_id
                )

        else:
            # Step 3: End Point (Finalize)

            # Determine logic from the Preview State
            is_clockwise = False
            if self.temp_entity_id is not None:
                temp_ent = self.element.sketch.registry.get_entity(
                    self.temp_entity_id
                )
                if isinstance(temp_ent, Arc):
                    is_clockwise = temp_ent.clockwise

            # If we hit our own preview point, ignore the hit so we create a
            # new, real point
            if pid_hit == self.temp_end_id:
                pid_hit = None

            # Clean up the preview geometry before adding the real command
            self._cleanup_temps()

            new_point = None
            if pid_hit is None:
                c = self.element.sketch.registry.get_point(self.center_id)
                s = self.element.sketch.registry.get_point(self.start_id)

                radius = math.hypot(s.x - c.x, s.y - c.y)
                curr_dist = math.hypot(mx - c.x, my - c.y)

                if curr_dist > 1e-9:
                    # Project onto circle
                    scale = radius / curr_dist
                    final_x = c.x + (mx - c.x) * scale
                    final_y = c.y + (my - c.y) * scale
                else:
                    final_x, final_y = mx, my

                temp_id = self.element.sketch.registry._id_counter
                pid_hit = temp_id
                new_point = Point(temp_id, final_x, final_y)

            # Cannot end at start or center
            if pid_hit != self.start_id and pid_hit != self.center_id:
                temp_arc_id = self.element.sketch.registry._id_counter + (
                    1 if new_point else 0
                )
                new_arc = Arc(
                    temp_arc_id,
                    self.start_id,
                    pid_hit,
                    self.center_id,
                    clockwise=bool(is_clockwise),
                )

                # ENFORCE ARC GEOMETRY: dist(C, S) == dist(C, E)
                geom_constr = EqualDistanceConstraint(
                    self.center_id, self.start_id, self.center_id, pid_hit
                )

                points_to_add = [new_point] if new_point else []

                # Adopt temp points
                if self.center_temp:
                    try:
                        p = self.element.sketch.registry.get_point(
                            self.center_id
                        )
                        self.element.sketch.registry.points.remove(p)
                        points_to_add.append(p)
                    except (IndexError, ValueError):
                        pass

                if self.start_temp:
                    try:
                        p = self.element.sketch.registry.get_point(
                            self.start_id
                        )
                        self.element.sketch.registry.points.remove(p)
                        points_to_add.append(p)
                    except (IndexError, ValueError):
                        pass

                cmd = AddItemsCommand(
                    self.element.sketch,
                    _("Add Arc"),
                    points=points_to_add,
                    entities=[new_arc],
                    constraints=[geom_constr],
                )
                self.element.execute_command(cmd)

                # Reset tool state
                self.center_id = None
                self.start_id = None
                self.center_temp = False
                self.start_temp = False

                # Select the last point
                self.element.selection.clear()
                self.element.selection.select_point(pid_hit, False)

        self.element.mark_dirty()
        return True

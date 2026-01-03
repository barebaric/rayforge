import math
from typing import List, Tuple
from ...geo import primitives as geo_primitives
from ..commands import AddFillCommand, RemoveFillCommand
from ..entities import Circle
from .base import SketchTool


class FillTool(SketchTool):
    """Handles creating and removing fills from closed regions."""

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        if n_press != 1:
            return False

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )
        sketch = self.element.sketch
        registry = sketch.registry

        all_loops = sketch._find_all_closed_loops()
        loops_under_cursor = []

        # Find all loops containing the click point
        for loop in all_loops:
            is_hit = False
            # Special case for single-entity circles
            if len(loop) == 1:
                entity = registry.get_entity(loop[0][0])
                if isinstance(entity, Circle):
                    center = registry.get_point(entity.center_idx)
                    radius_pt = registry.get_point(entity.radius_pt_idx)
                    radius = math.hypot(
                        radius_pt.x - center.x, radius_pt.y - center.y
                    )
                    dist_to_center = math.hypot(mx - center.x, my - center.y)
                    if dist_to_center < radius:
                        is_hit = True
            else:
                # General polygon case
                polygon: List[Tuple[float, float]] = []
                is_valid = True
                for eid, fwd in loop:
                    entity = registry.get_entity(eid)
                    if not entity:
                        is_valid = False
                        break
                    # Simple polygon from endpoints for hit testing
                    p_ids = entity.get_point_ids()
                    start_pid = p_ids[0] if fwd else p_ids[1]
                    try:
                        p = registry.get_point(start_pid)
                        polygon.append((p.x, p.y))
                    except IndexError:
                        is_valid = False
                        break
                if is_valid and geo_primitives.is_point_in_polygon(
                    (mx, my), polygon
                ):
                    is_hit = True

            if is_hit:
                area = abs(sketch._calculate_loop_signed_area(loop))
                loops_under_cursor.append((area, loop))

        if not loops_under_cursor:
            return False

        # Select the smallest area loop under the cursor
        loops_under_cursor.sort(key=lambda x: x[0])
        target_loop = loops_under_cursor[0][1]
        target_loop_set = frozenset(target_loop)

        # Check if a fill already exists for this loop
        existing_fill = None
        for fill in sketch.fills:
            if frozenset(fill.boundary) == target_loop_set:
                existing_fill = fill
                break

        if existing_fill:
            # Remove the fill
            cmd = RemoveFillCommand(self.element.sketch, existing_fill)
            self.element.execute_command(cmd)
        else:
            # Add a new fill
            cmd = AddFillCommand(self.element.sketch, target_loop)
            self.element.execute_command(cmd)

        self.element.mark_dirty()
        return True

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

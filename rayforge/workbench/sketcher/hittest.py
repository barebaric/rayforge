import math
import cairo
from collections import defaultdict
from typing import Optional, Tuple, Any
from rayforge.core.sketcher.entities import Line, Arc, Circle, Entity
from rayforge.core.sketcher.constraints import (
    HorizontalConstraint,
    VerticalConstraint,
    DistanceConstraint,
    RadiusConstraint,
    DiameterConstraint,
    PerpendicularConstraint,
    CoincidentConstraint,
    PointOnLineConstraint,
)
from rayforge.core.geo.primitives import (
    find_closest_point_on_line_segment,
    line_intersection,
)


class SketchHitTester:
    """Handles geometric hit testing for sketch elements."""

    def __init__(self, snap_distance: float = 12.0):
        self.snap_distance = snap_distance

    def screen_to_model(
        self, wx: float, wy: float, element: Any
    ) -> Tuple[float, float]:
        """
        Converts world coordinates to Model coordinates
        (accounting for content_transform).
        World -> Local -> Model
        """
        try:
            # 1. World -> Local
            inv_world = element.get_world_transform().invert()
            lx, ly = inv_world.transform_point((wx, wy))

            # 2. Local -> Model (Inverse of content_transform)
            inv_content = element.content_transform.invert()
            mx, my = inv_content.transform_point((lx, ly))

            return mx, my
        except Exception:
            return 0.0, 0.0

    def get_model_to_screen_transform(self, element: Any) -> Any:
        """
        Returns Matrix: Model -> Screen.
        """
        if not element.canvas:
            return cairo.Matrix()

        local_to_screen = (
            element.canvas.view_transform @ element.get_world_transform()
        )
        model_to_local = element.content_transform

        return local_to_screen @ model_to_local

    def get_hit_data(
        self, wx: float, wy: float, element: Any
    ) -> Tuple[Optional[str], Any]:
        """
        Determines what was clicked using Model coordinates.
        Returns (type_string, object_id_or_index).
        Priorities: Points > Overlays (Constraints/Junctions) > Entities.
        """
        if not element.canvas:
            return None, None

        # 1. Points (most specific target)
        hit_pid = self._hit_test_points(wx, wy, element)
        if hit_pid is not None:
            return "point", hit_pid

        # 2. Overlays (Constraints and Junctions)
        hit_type, hit_obj = self._hit_test_overlays(wx, wy, element)
        if hit_type is not None:
            return hit_type, hit_obj

        # 3. Entities (Lines/Arcs/Circles)
        hit_entity = self._hit_test_entities(wx, wy, element)
        if hit_entity is not None:
            return "entity", hit_entity

        return None, None

    def _hit_test_points(self, wx, wy, element) -> Optional[int]:
        """Precise point hit-testing in SCREEN coordinates."""
        if not element.canvas:
            return None

        to_screen = self.get_model_to_screen_transform(element)
        sx_in, sy_in = element.canvas.view_transform.transform_point((wx, wy))

        # Use the visual radius of the point + a small buffer
        threshold = element.point_radius + 2.0
        best_pid = None
        min_dist_sq = float("inf")

        points = element.sketch.registry.points or []
        for p in points:
            sx, sy = to_screen.transform_point((p.x, p.y))
            dist_sq = (sx - sx_in) ** 2 + (sy - sy_in) ** 2
            if dist_sq < threshold**2 and dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_pid = p.id
        return best_pid

    def _hit_test_overlays(self, wx, wy, element) -> Tuple[Optional[str], Any]:
        if not element.canvas:
            return None, None
        to_screen = self.get_model_to_screen_transform(element)
        sx_in, sy_in = element.canvas.view_transform.transform_point((wx, wy))

        # Test explicit constraints first
        constraints = element.sketch.constraints or []
        for idx, constr in enumerate(constraints):
            if self._is_constraint_hit(
                constr, sx_in, sy_in, to_screen, element
            ):
                return "constraint", idx

        # Test implicit junction constraints
        junction_pid = self._hit_test_junctions(
            sx_in, sy_in, to_screen, element
        )
        if junction_pid is not None:
            return "junction", junction_pid

        return None, None

    def _hit_test_junctions(self, sx, sy, to_screen, element) -> Optional[int]:
        registry = element.sketch.registry
        point_counts = defaultdict(int)
        for entity in registry.entities:
            if isinstance(entity, Line):
                point_counts[entity.p1_idx] += 1
                point_counts[entity.p2_idx] += 1
            elif isinstance(entity, Arc):
                point_counts[entity.start_idx] += 1
                point_counts[entity.end_idx] += 1
                point_counts[entity.center_idx] += 1
            elif isinstance(entity, Circle):
                point_counts[entity.center_idx] += 1
                point_counts[entity.radius_pt_idx] += 1

        for pid, count in point_counts.items():
            if count > 1:
                try:
                    p = registry.get_point(pid)
                    spx, spy = to_screen.transform_point((p.x, p.y))
                    if math.hypot(sx - spx, sy - spy) < 13.0:
                        return pid
                except IndexError:
                    continue
        return None

    def _is_constraint_hit(self, constr, sx, sy, to_screen, element) -> bool:
        registry = element.sketch.registry
        click_radius = 13.0

        def safe_get(pid):
            try:
                return registry.get_point(pid)
            except Exception:
                return None

        if isinstance(constr, (HorizontalConstraint, VerticalConstraint)):
            p1 = safe_get(constr.p1)
            p2 = safe_get(constr.p2)
            if p1 and p2:
                s1 = to_screen.transform_point((p1.x, p1.y))
                s2 = to_screen.transform_point((p2.x, p2.y))

                t = 0.2
                mx = s1[0] + (s2[0] - s1[0]) * t
                my = s1[1] + (s2[1] - s1[1]) * t
                cx = (
                    mx if isinstance(constr, HorizontalConstraint) else mx + 10
                )
                cy = (
                    my - 10 if isinstance(constr, HorizontalConstraint) else my
                )

                return math.hypot(sx - cx, sy - cy) < click_radius

        elif isinstance(constr, (RadiusConstraint, DiameterConstraint)):
            pos_data = self.get_circular_label_pos(constr, to_screen, element)
            if pos_data:
                label_sx, label_sy, _, _ = pos_data
                return math.hypot(sx - label_sx, sy - label_sy) < 15

        elif isinstance(constr, DistanceConstraint):
            p1 = safe_get(constr.p1)
            p2 = safe_get(constr.p2)
            if p1 and p2:
                s1 = to_screen.transform_point((p1.x, p1.y))
                s2 = to_screen.transform_point((p2.x, p2.y))
                mx, my = (s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2
                return math.hypot(sx - mx, sy - my) < 15

        elif isinstance(constr, PerpendicularConstraint):
            data = self.get_perp_intersection_screen(
                constr, to_screen, element
            )
            if data:
                vx, vy = data[0], data[1]
                return math.hypot(sx - vx, sy - vy) < 20

        elif isinstance(constr, CoincidentConstraint):
            # Prefer hit-testing the non-origin point for discoverability
            origin_id = getattr(element.sketch, "origin_id", -1)
            pid_to_check = constr.p1
            if constr.p1 == origin_id and origin_id != -1:
                pid_to_check = constr.p2

            pt_to_check = safe_get(pid_to_check)
            if pt_to_check:
                s_pt = to_screen.transform_point(
                    (pt_to_check.x, pt_to_check.y)
                )
                return math.hypot(sx - s_pt[0], sy - s_pt[1]) < click_radius

        elif isinstance(constr, PointOnLineConstraint):
            pt = safe_get(constr.point_id)
            if pt:
                s_pt = to_screen.transform_point((pt.x, pt.y))
                return math.hypot(sx - s_pt[0], sy - s_pt[1]) < click_radius

        return False

    def _hit_test_entities(self, wx, wy, element) -> Optional[Entity]:
        mx, my = self.screen_to_model(wx, wy, element)

        scale = 1.0
        if element.canvas and hasattr(element.canvas, "get_view_scale"):
            scale_x, _ = element.canvas.get_view_scale()
            scale = scale_x if scale_x > 1e-9 else 1.0
        threshold = self.snap_distance / scale

        def safe_get(pid):
            try:
                return element.sketch.registry.get_point(pid)
            except Exception:
                return None

        entities = element.sketch.registry.entities or []
        for entity in entities:
            if isinstance(entity, Line):
                p1 = safe_get(entity.p1_idx)
                p2 = safe_get(entity.p2_idx)
                if p1 and p2:
                    _, _, dist_sq = find_closest_point_on_line_segment(
                        (p1.x, p1.y), (p2.x, p2.y), mx, my
                    )
                    if dist_sq < threshold**2:
                        return entity

            elif isinstance(entity, (Arc, Circle)):
                center = safe_get(entity.center_idx)
                if not center:
                    continue

                radius = 0.0
                if isinstance(entity, Arc):
                    start = safe_get(entity.start_idx)
                    if start:
                        radius = math.hypot(
                            start.x - center.x, start.y - center.y
                        )
                elif isinstance(entity, Circle):
                    radius_pt = safe_get(entity.radius_pt_idx)
                    if radius_pt:
                        radius = math.hypot(
                            radius_pt.x - center.x, radius_pt.y - center.y
                        )

                if radius == 0.0:
                    continue

                dist_mouse = math.hypot(mx - center.x, my - center.y)

                if abs(dist_mouse - radius) < threshold:
                    if isinstance(entity, Circle):
                        return entity

                    if isinstance(entity, Arc):
                        start = safe_get(entity.start_idx)
                        end = safe_get(entity.end_idx)
                        if start and end:
                            angle_mouse = math.atan2(
                                my - center.y, mx - center.x
                            )
                            angle_start = math.atan2(
                                start.y - center.y, start.x - center.x
                            )
                            angle_end = math.atan2(
                                end.y - center.y, end.x - center.x
                            )

                            if self._is_angle_between(
                                angle_mouse,
                                angle_start,
                                angle_end,
                                entity.clockwise,
                            ):
                                return entity
        return None

    def _is_angle_between(self, target, start, end, clockwise):
        """Checks if target angle is between start and end angles."""
        # Normalize to [0, 2PI)
        target %= 2 * math.pi
        start %= 2 * math.pi
        end %= 2 * math.pi

        if clockwise:
            # Clockwise: Start > End (visually, usually means start > end)
            if start < end:
                start += 2 * math.pi
            if target > start:
                target -= 2 * math.pi
            return end <= target <= start
        else:
            # Counter-Clockwise: Start < End
            if end < start:
                end += 2 * math.pi
            if target < start:
                target += 2 * math.pi
            return start <= target <= end

    def get_circular_label_pos(self, constr, to_screen, element):
        """Calculates screen position for Radius/Diameter constraint labels."""
        entity_id = -1
        if isinstance(constr, RadiusConstraint):
            entity_id = constr.entity_id
        elif isinstance(constr, DiameterConstraint):
            entity_id = constr.circle_id

        entity = element.sketch.registry.get_entity(entity_id)
        if not isinstance(entity, (Arc, Circle)):
            return None

        center = element.sketch.registry.get_point(entity.center_idx)
        if not center:
            return None

        radius, mid_angle = 0.0, 0.0

        if isinstance(entity, Arc):
            start = element.sketch.registry.get_point(entity.start_idx)
            end = element.sketch.registry.get_point(entity.end_idx)
            if not (start and end):
                return None

            radius = math.hypot(start.x - center.x, start.y - center.y)
            start_a = math.atan2(start.y - center.y, start.x - center.x)
            end_a = math.atan2(end.y - center.y, end.x - center.x)

            angle_range = end_a - start_a
            if entity.clockwise:
                if angle_range > 0:
                    angle_range -= 2 * math.pi
            else:
                if angle_range < 0:
                    angle_range += 2 * math.pi
            mid_angle = start_a + angle_range / 2.0

        elif isinstance(entity, Circle):
            radius_pt = element.sketch.registry.get_point(entity.radius_pt_idx)
            if not radius_pt:
                return None
            radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
            mid_angle = math.atan2(
                radius_pt.y - center.y, radius_pt.x - center.x
            )

        if radius == 0.0:
            return None

        scale = 1.0
        if element.canvas and hasattr(element.canvas, "get_view_scale"):
            scale_x, _ = element.canvas.get_view_scale()
            scale = scale_x if scale_x > 1e-9 else 1.0

        label_dist = radius + 20 / scale
        label_mx = center.x + label_dist * math.cos(mid_angle)
        label_my = center.y + label_dist * math.sin(mid_angle)
        label_sx, label_sy = to_screen.transform_point((label_mx, label_my))

        # Position on the arc for the leader line
        arc_mid_mx = center.x + radius * math.cos(mid_angle)
        arc_mid_my = center.y + radius * math.sin(mid_angle)
        arc_mid_sx, arc_mid_sy = to_screen.transform_point(
            (arc_mid_mx, arc_mid_my)
        )

        return label_sx, label_sy, arc_mid_sx, arc_mid_sy

    def get_perp_intersection_screen(self, constr, to_screen, element):
        """Calculates intersection point and angles for perp visualization."""
        l1 = element.sketch.registry.get_entity(constr.l1_id)
        l2 = element.sketch.registry.get_entity(constr.l2_id)

        if not (isinstance(l1, Line) and isinstance(l2, Line)):
            return None

        def safe_get(pid):
            try:
                return element.sketch.registry.get_point(pid)
            except Exception:
                return None

        p1, p2 = safe_get(l1.p1_idx), safe_get(l1.p2_idx)
        p3, p4 = safe_get(l2.p1_idx), safe_get(l2.p2_idx)

        if not (p1 and p2 and p3 and p4):
            return None

        pt = line_intersection(
            (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)
        )
        if not pt:
            return None

        ix, iy = pt
        sx, sy = to_screen.transform_point((ix, iy))

        def get_screen_pt(p):
            return to_screen.transform_point((p.x, p.y))

        s_p1, s_p2 = get_screen_pt(p1), get_screen_pt(p2)
        s_p3, s_p4 = get_screen_pt(p3), get_screen_pt(p4)

        def dist_sq(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        vec1_p = (
            s_p1 if dist_sq(s_p1, (sx, sy)) > dist_sq(s_p2, (sx, sy)) else s_p2
        )
        vec2_p = (
            s_p3 if dist_sq(s_p3, (sx, sy)) > dist_sq(s_p4, (sx, sy)) else s_p4
        )

        ang1 = math.atan2(vec1_p[1] - sy, vec1_p[0] - sx)
        ang2 = math.atan2(vec2_p[1] - sy, vec2_p[0] - sx)

        return sx, sy, ang1, ang2

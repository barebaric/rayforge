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
    EqualLengthConstraint,
    SymmetryConstraint,
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

    def _get_equality_symbol_pos(self, entity, to_screen, element):
        """Calculates screen pos for an equality symbol on an entity."""

        def safe_get(pid):
            try:
                return element.sketch.registry.get_point(pid)
            except Exception:
                return None

        # 1. Get anchor point (mid_x, mid_y) and normal_angle in MODEL space
        mid_x, mid_y, normal_angle = 0.0, 0.0, 0.0

        if isinstance(entity, Line):
            p1 = safe_get(entity.p1_idx)
            p2 = safe_get(entity.p2_idx)
            if not (p1 and p2):
                return None
            mid_x = (p1.x + p2.x) / 2.0
            mid_y = (p1.y + p2.y) / 2.0
            tangent_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            normal_angle = tangent_angle - (math.pi / 2.0)

        elif isinstance(entity, Arc):
            center = safe_get(entity.center_idx)
            start = safe_get(entity.start_idx)
            end = safe_get(entity.end_idx)
            if not (center and start and end):
                return None

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
            radius = math.hypot(start.x - center.x, start.y - center.y)
            mid_x = center.x + radius * math.cos(mid_angle)
            mid_y = center.y + radius * math.sin(mid_angle)
            normal_angle = mid_angle

        elif isinstance(entity, Circle):
            center = safe_get(entity.center_idx)
            radius_pt = safe_get(entity.radius_pt_idx)
            if not (center and radius_pt):
                return None
            radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
            normal_angle = math.atan2(
                radius_pt.y - center.y, radius_pt.x - center.x
            )
            mid_x = center.x + radius * math.cos(normal_angle)
            mid_y = center.y + radius * math.sin(normal_angle)

        scale = 1.0
        if element.canvas and hasattr(element.canvas, "get_view_scale"):
            scale, _ = element.canvas.get_view_scale()
            scale = max(scale, 1e-9)
        offset_dist_model = 15.0 / scale
        final_x = mid_x + offset_dist_model * math.cos(normal_angle)
        final_y = mid_y + offset_dist_model * math.sin(normal_angle)
        return to_screen.transform_point((final_x, final_y))

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
            data = self.get_perp_visuals_screen(constr, to_screen, element)
            if data:
                vx, vy, _, _ = data
                return math.hypot(sx - vx, sy - vy) < 20
            return False

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

        elif isinstance(constr, SymmetryConstraint):
            p1 = safe_get(constr.p1)
            p2 = safe_get(constr.p2)
            if not (p1 and p2):
                return False
            s1 = to_screen.transform_point((p1.x, p1.y))
            s2 = to_screen.transform_point((p2.x, p2.y))
            mx = (s1[0] + s2[0]) / 2.0
            my = (s1[1] + s2[1]) / 2.0
            angle = math.atan2(s2[1] - s1[1], s2[0] - s1[0])
            offset = 12.0
            lx = mx - offset * math.cos(angle)
            ly = my - offset * math.sin(angle)
            rx = mx + offset * math.cos(angle)
            ry = my + offset * math.sin(angle)
            if math.hypot(sx - lx, sy - ly) < click_radius:
                return True
            if math.hypot(sx - rx, sy - ry) < click_radius:
                return True

        elif isinstance(constr, EqualLengthConstraint):
            for entity_id in constr.entity_ids:
                entity = registry.get_entity(entity_id)
                if not entity:
                    continue
                pos = self._get_equality_symbol_pos(entity, to_screen, element)
                if pos:
                    esx, esy = pos
                    if math.hypot(sx - esx, sy - esy) < 15:
                        return True

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
                            if self._is_angle_on_arc(
                                angle_mouse,
                                center,
                                start,
                                end,
                                entity.clockwise,
                            ):
                                return entity
        return None

    def _is_angle_on_arc(self, target_angle, center, start, end, clockwise):
        """Checks if a target angle lies on the arc segment."""
        angle_start = math.atan2(start.y - center.y, start.x - center.x)
        angle_end = math.atan2(end.y - center.y, end.x - center.x)

        # Normalize all angles to [0, 2*pi)
        target = (target_angle + 2 * math.pi) % (2 * math.pi)
        start_norm = (angle_start + 2 * math.pi) % (2 * math.pi)
        end_norm = (angle_end + 2 * math.pi) % (2 * math.pi)

        if clockwise:
            # If start < end, it wraps around 0. e.g., start at 330 deg, end
            # at 30 deg.
            if start_norm > end_norm:
                return target >= start_norm or target <= end_norm
            # Normal case where start > end.
            return target >= start_norm and target <= end_norm
        else:  # Counter-clockwise
            # If start > end, it wraps around 0. e.g., start at 330 deg, end
            # at 30 deg.
            if start_norm > end_norm:
                return target >= start_norm or target <= end_norm
            # Normal case where start < end.
            return target >= start_norm and target <= end_norm

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

    def get_perp_visuals_screen(
        self, constr, to_screen, element
    ) -> Optional[Tuple[float, float, Optional[float], Optional[float]]]:
        """
        Calculates screen position and angles for perpendicular visualization.
        Returns (sx, sy, angle1, angle2). Angles are only for Line-Line case.
        """
        registry = element.sketch.registry
        e1 = registry.get_entity(constr.e1_id)
        e2 = registry.get_entity(constr.e2_id)
        if not (e1 and e2):
            return None

        # --- Case 1: Line-Line ---
        if isinstance(e1, Line) and isinstance(e2, Line):
            return self._get_perp_line_line_visuals(e1, e2, to_screen, element)

        # --- Case 2: Line-Shape ---
        line, shape = (e1, e2) if isinstance(e1, Line) else (e2, e1)
        if isinstance(line, Line) and isinstance(shape, (Arc, Circle)):
            return self._get_perp_line_shape_visuals(
                line, shape, to_screen, element
            )

        # --- Case 3: Shape-Shape ---
        if isinstance(e1, (Arc, Circle)) and isinstance(e2, (Arc, Circle)):
            return self._get_perp_shape_shape_visuals(
                e1, e2, to_screen, element
            )

        return None

    def _get_perp_line_line_visuals(self, l1, l2, to_screen, element):
        def safe_get(pid):
            try:
                return element.sketch.registry.get_point(pid)
            except Exception:
                return None

        p1 = safe_get(l1.p1_idx)
        p2 = safe_get(l1.p2_idx)
        p3 = safe_get(l2.p1_idx)
        p4 = safe_get(l2.p2_idx)
        if not (p1 and p2 and p3 and p4):
            return None
        pt = line_intersection(
            (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)
        )
        if not pt:  # Fallback for parallel lines
            m1x, m1y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
            m2x, m2y = (p3.x + p4.x) / 2, (p3.y + p4.y) / 2
            pt = ((m1x + m2x) / 2, (m1y + m2y) / 2)

        ix, iy = pt
        sx, sy = to_screen.transform_point((ix, iy))
        s_p1 = to_screen.transform_point((p1.x, p1.y))
        s_p2 = to_screen.transform_point((p2.x, p2.y))
        s_p3 = to_screen.transform_point((p3.x, p3.y))
        s_p4 = to_screen.transform_point((p4.x, p4.y))

        def dist_sq(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        v1p = (
            s_p1 if dist_sq(s_p1, (sx, sy)) > dist_sq(s_p2, (sx, sy)) else s_p2
        )
        v2p = (
            s_p3 if dist_sq(s_p3, (sx, sy)) > dist_sq(s_p4, (sx, sy)) else s_p4
        )
        ang1 = math.atan2(v1p[1] - sy, v1p[0] - sx)
        ang2 = math.atan2(v2p[1] - sy, v2p[0] - sx)
        return sx, sy, ang1, ang2

    def _get_perp_line_shape_visuals(self, line, shape, to_screen, element):
        def safe_get(pid):
            try:
                return element.sketch.registry.get_point(pid)
            except Exception:
                return None

        def is_on_line_segment(pt, p1, p2):
            # Check if point pt is on segment p1-p2 using dot products
            dot1 = (pt[0] - p1.x) * (p2.x - p1.x) + (pt[1] - p1.y) * (
                p2.y - p1.y
            )
            if dot1 < 0:
                return False
            dot2 = (pt[0] - p2.x) * (p1.x - p2.x) + (pt[1] - p2.y) * (
                p1.y - p2.y
            )
            if dot2 < 0:
                return False
            return True

        center = safe_get(shape.center_idx)
        lp1 = safe_get(line.p1_idx)
        lp2 = safe_get(line.p2_idx)
        if not (center and lp1 and lp2):
            return None

        dxL, dyL = lp2.x - lp1.x, lp2.y - lp1.y
        lenL = math.hypot(dxL, dyL)
        if lenL < 1e-9:
            return None
        ux, uy = dxL / lenL, dyL / lenL

        if isinstance(shape, Arc):
            sp = safe_get(shape.start_idx)
        else:
            sp = safe_get(shape.radius_pt_idx)
        if not sp:
            return None
        radius = math.hypot(sp.x - center.x, sp.y - center.y)

        # Intersection points of line (through center) and circle
        ix1, iy1 = center.x + radius * ux, center.y + radius * uy
        ix2, iy2 = center.x - radius * ux, center.y - radius * uy

        valid_points = []
        for ix, iy in [(ix1, iy1), (ix2, iy2)]:
            on_line = is_on_line_segment((ix, iy), lp1, lp2)
            on_arc = True
            if isinstance(shape, Arc):
                start, end = safe_get(shape.start_idx), safe_get(shape.end_idx)
                if not (start and end):
                    continue
                angle = math.atan2(iy - center.y, ix - center.x)
                on_arc = self._is_angle_on_arc(
                    angle, center, start, end, shape.clockwise
                )

            if on_line and on_arc:
                valid_points.append((ix, iy))

        if valid_points:
            best_pt = valid_points[0]
            # If two valid points, choose closer to line midpoint
            if len(valid_points) > 1:
                lmx, lmy = (lp1.x + lp2.x) / 2, (lp1.y + lp2.y) / 2
                d1_sq = (best_pt[0] - lmx) ** 2 + (best_pt[1] - lmy) ** 2
                d2_sq = (valid_points[1][0] - lmx) ** 2 + (
                    valid_points[1][1] - lmy
                ) ** 2
                if d2_sq < d1_sq:
                    best_pt = valid_points[1]
            sx, sy = to_screen.transform_point(best_pt)
            return sx, sy, None, None

        # Fallback: projection of center on line (is the center itself)
        sx, sy = to_screen.transform_point((center.x, center.y))
        return sx, sy, None, None

    def _get_perp_shape_shape_visuals(self, s1, s2, to_screen, element):
        def safe_get(pid):
            try:
                return element.sketch.registry.get_point(pid)
            except Exception:
                return None

        c1 = safe_get(s1.center_idx)
        c2 = safe_get(s2.center_idx)
        if not (c1 and c2):
            return None

        def get_radius(s, c):
            if isinstance(s, Arc):
                sp = safe_get(s.start_idx)
            else:
                sp = safe_get(s.radius_pt_idx)
            if not sp or not c:
                return 0
            return math.hypot(sp.x - c.x, sp.y - c.y)

        r1 = get_radius(s1, c1)
        r2 = get_radius(s2, c2)

        d_sq = (c2.x - c1.x) ** 2 + (c2.y - c1.y) ** 2
        d = math.sqrt(d_sq) if d_sq > 1e-9 else 0
        if d == 0:
            return None  # Concentric circles can't be perpendicular

        # Intersection point formula
        a = (r1**2 - r2**2 + d_sq) / (2 * d)
        try:
            h = math.sqrt(max(0, r1**2 - a**2))
        except ValueError:
            return None  # No real intersection

        x2 = c1.x + a * (c2.x - c1.x) / d
        y2 = c1.y + a * (c2.y - c1.y) / d

        # Two possible intersection points
        ix1 = x2 + h * (c2.y - c1.y) / d
        iy1 = y2 - h * (c2.x - c1.x) / d
        ix2 = x2 - h * (c2.y - c1.y) / d
        iy2 = y2 + h * (c2.x - c1.x) / d

        valid_points = []
        for ix, iy in [(ix1, iy1), (ix2, iy2)]:
            on_s1 = True
            if isinstance(s1, Arc):
                start, end = safe_get(s1.start_idx), safe_get(s1.end_idx)
                if start and end:
                    angle = math.atan2(iy - c1.y, ix - c1.x)
                    on_s1 = self._is_angle_on_arc(
                        angle, c1, start, end, s1.clockwise
                    )

            on_s2 = True
            if isinstance(s2, Arc):
                start, end = safe_get(s2.start_idx), safe_get(s2.end_idx)
                if start and end:
                    angle = math.atan2(iy - c2.y, ix - c2.x)
                    on_s2 = self._is_angle_on_arc(
                        angle, c2, start, end, s2.clockwise
                    )

            if on_s1 and on_s2:
                valid_points.append((ix, iy))

        if not valid_points:
            # Fallback: one of the infinite intersection points
            sx, sy = to_screen.transform_point((ix1, iy1))
            return sx, sy, None, None

        best_pt = valid_points[0]
        if len(valid_points) > 1:

            def get_midpoint(arc, center):
                if not isinstance(arc, Arc):
                    return None
                s = safe_get(arc.start_idx)
                e = safe_get(arc.end_idx)
                if not (s and e and center):
                    return None
                sa = math.atan2(s.y - center.y, s.x - center.x)
                ea = math.atan2(e.y - center.y, e.x - center.x)
                ar = ea - sa
                if arc.clockwise:
                    if ar > 0:
                        ar -= 2 * math.pi
                else:
                    if ar < 0:
                        ar += 2 * math.pi
                mid_a = sa + ar / 2.0
                r = math.hypot(s.x - center.x, s.y - center.y)
                return center.x + r * math.cos(mid_a), center.y + r * math.sin(
                    mid_a
                )

            m1 = get_midpoint(s1, c1)
            m2 = get_midpoint(s2, c2)

            if m1 and m2:
                d1_sq = (
                    (valid_points[0][0] - m1[0]) ** 2
                    + (valid_points[0][1] - m1[1]) ** 2
                    + (valid_points[0][0] - m2[0]) ** 2
                    + (valid_points[0][1] - m2[1]) ** 2
                )
                d2_sq = (
                    (valid_points[1][0] - m1[0]) ** 2
                    + (valid_points[1][1] - m1[1]) ** 2
                    + (valid_points[1][0] - m2[0]) ** 2
                    + (valid_points[1][1] - m2[1]) ** 2
                )
                if d2_sq < d1_sq:
                    best_pt = valid_points[1]

        sx, sy = to_screen.transform_point(best_pt)
        return sx, sy, None, None

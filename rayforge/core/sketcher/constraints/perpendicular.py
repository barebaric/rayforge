from __future__ import annotations
import math
from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    List,
    Optional,
    Callable,
    TYPE_CHECKING,
    cast,
)
from .base import Constraint
from ..entities import Line, Arc, Circle
from rayforge.core.geo.primitives import (
    line_intersection,
    circle_circle_intersection,
    is_point_on_segment,
)

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry


class PerpendicularConstraint(Constraint):
    """
    Enforces perpendicularity between two entities.
    - Line/Line: Vectors are at 90 degrees.
    - Line/Arc, Line/Circle: Line passes through the shape's center.
    - Arc/Arc, Arc/Circle, Circle/Circle: Shapes intersect at a right angle.
    """

    def __init__(self, e1_id: int, e2_id: int):
        self.e1_id = e1_id
        self.e2_id = e2_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "PerpendicularConstraint",
            "e1_id": self.e1_id,
            "e2_id": self.e2_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerpendicularConstraint":
        return cls(e1_id=data["e1_id"], e2_id=data["e2_id"])

    def _get_radius_sq(
        self, shape: Union[Arc, Circle], reg: "EntityRegistry"
    ) -> float:
        """Helper to get squared radius of an Arc or Circle."""
        center = reg.get_point(shape.center_idx)
        if isinstance(shape, Arc):
            start = reg.get_point(shape.start_idx)
            return (start.x - center.x) ** 2 + (start.y - center.y) ** 2
        elif isinstance(shape, Circle):
            radius_pt = reg.get_point(shape.radius_pt_idx)
            return (radius_pt.x - center.x) ** 2 + (
                radius_pt.y - center.y
            ) ** 2
        return 0.0

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> float:
        e1 = reg.get_entity(self.e1_id)
        e2 = reg.get_entity(self.e2_id)

        if e1 is None or e2 is None:
            return 0.0

        # Case 1: Line-Line
        if isinstance(e1, Line) and isinstance(e2, Line):
            p1 = reg.get_point(e1.p1_idx)
            p2 = reg.get_point(e1.p2_idx)
            p3 = reg.get_point(e2.p1_idx)
            p4 = reg.get_point(e2.p2_idx)

            dx1, dy1 = p2.x - p1.x, p2.y - p1.y
            dx2, dy2 = p4.x - p3.x, p4.y - p3.y
            # Dot product
            return dx1 * dx2 + dy1 * dy2

        # Case 2: Line-Arc/Circle
        line, shape = None, None
        if isinstance(e1, Line) and isinstance(e2, (Arc, Circle)):
            line, shape = e1, e2
        elif isinstance(e2, Line) and isinstance(e1, (Arc, Circle)):
            line, shape = e2, e1

        if line and shape:
            # Constraint: Line must pass through the shape's center
            # (i.e., line points and center are collinear)
            lp1 = reg.get_point(line.p1_idx)
            lp2 = reg.get_point(line.p2_idx)
            # This cast is safe due to the isinstance checks above
            shape_with_center = cast(Union[Arc, Circle], shape)
            center = reg.get_point(shape_with_center.center_idx)
            # Cross product (lp2-lp1) x (center-lp1)
            return (lp2.x - lp1.x) * (center.y - lp1.y) - (
                center.x - lp1.x
            ) * (lp2.y - lp1.y)

        # Case 3: Arc/Circle - Arc/Circle
        shape1, shape2 = None, None
        if isinstance(e1, (Arc, Circle)) and isinstance(e2, (Arc, Circle)):
            shape1, shape2 = e1, e2

        if shape1 and shape2:
            # Constraint: The circles intersect at a right angle.
            # Geometric property: r1^2 + r2^2 = d^2, where d is distance
            # between centers.
            c1 = reg.get_point(shape1.center_idx)
            c2 = reg.get_point(shape2.center_idx)

            r1_sq = self._get_radius_sq(shape1, reg)
            r2_sq = self._get_radius_sq(shape2, reg)

            dist_centers_sq = (c2.x - c1.x) ** 2 + (c2.y - c1.y) ** 2

            return r1_sq + r2_sq - dist_centers_sq

        return 0.0

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        e1 = reg.get_entity(self.e1_id)
        e2 = reg.get_entity(self.e2_id)
        grad: Dict[int, List[Tuple[float, float]]] = {}

        if e1 is None or e2 is None:
            return {}

        def add_grad(pid, gx, gy):
            if pid not in grad:
                grad[pid] = [(0.0, 0.0)]
            cx, cy = grad[pid][0]
            grad[pid][0] = (cx + gx, cy + gy)

        if isinstance(e1, Line) and isinstance(e2, Line):
            p1 = reg.get_point(e1.p1_idx)
            p2 = reg.get_point(e1.p2_idx)
            p3 = reg.get_point(e2.p1_idx)
            p4 = reg.get_point(e2.p2_idx)
            dx1, dy1 = p2.x - p1.x, p2.y - p1.y
            dx2, dy2 = p4.x - p3.x, p4.y - p3.y
            # E = dx1*dx2 + dy1*dy2
            add_grad(e1.p1_idx, -dx2, -dy2)
            add_grad(e1.p2_idx, dx2, dy2)
            add_grad(e2.p1_idx, -dx1, -dy1)
            add_grad(e2.p2_idx, dx1, dy1)

        elif (isinstance(e1, Line) and isinstance(e2, (Arc, Circle))) or (
            isinstance(e2, Line) and isinstance(e1, (Arc, Circle))
        ):
            line, shape = (e1, e2) if isinstance(e1, Line) else (e2, e1)
            # Ensure type safety
            if isinstance(line, Line) and isinstance(shape, (Arc, Circle)):
                l1 = reg.get_point(line.p1_idx)
                l2 = reg.get_point(line.p2_idx)
                c = reg.get_point(shape.center_idx)
                dx = l2.x - l1.x
                dy = l2.y - l1.y
                # E = dx*(c.y - l1.y) - (c.x - l1.x)*dy
                add_grad(shape.center_idx, -dy, dx)
                add_grad(line.p1_idx, dy - (c.y - l1.y), -dx + (c.x - l1.x))
                add_grad(line.p2_idx, c.y - l1.y, -(c.x - l1.x))

        elif isinstance(e1, (Arc, Circle)) and isinstance(e2, (Arc, Circle)):
            # r1^2 + r2^2 - dist_sq
            s1, s2 = e1, e2
            c1 = reg.get_point(s1.center_idx)
            c2 = reg.get_point(s2.center_idx)

            # Center 1 part:
            # -d(dist)/dc1 + d(r1)/dc1 (if arc/circle)
            d_dist_x = 2 * (c2.x - c1.x)
            d_dist_y = 2 * (c2.y - c1.y)

            # R1 derivs
            dr1_c_x, dr1_c_y = 0.0, 0.0
            if isinstance(s1, Arc):
                p1 = reg.get_point(s1.start_idx)
                dr1_c_x = -2 * (p1.x - c1.x)
                dr1_c_y = -2 * (p1.y - c1.y)
                add_grad(s1.start_idx, 2 * (p1.x - c1.x), 2 * (p1.y - c1.y))
            elif isinstance(s1, Circle):
                p1 = reg.get_point(s1.radius_pt_idx)
                dr1_c_x = -2 * (p1.x - c1.x)
                dr1_c_y = -2 * (p1.y - c1.y)
                add_grad(
                    s1.radius_pt_idx,
                    2 * (p1.x - c1.x),
                    2 * (p1.y - c1.y),
                )

            # R2 derivs
            dr2_c_x, dr2_c_y = 0.0, 0.0
            if isinstance(s2, Arc):
                p2 = reg.get_point(s2.start_idx)
                dr2_c_x = -2 * (p2.x - c2.x)
                dr2_c_y = -2 * (p2.y - c2.y)
                add_grad(s2.start_idx, 2 * (p2.x - c2.x), 2 * (p2.y - c2.y))
            elif isinstance(s2, Circle):
                p2 = reg.get_point(s2.radius_pt_idx)
                dr2_c_x = -2 * (p2.x - c2.x)
                dr2_c_y = -2 * (p2.y - c2.y)
                add_grad(
                    s2.radius_pt_idx,
                    2 * (p2.x - c2.x),
                    2 * (p2.y - c2.y),
                )

            # Dist term for C1: -(-2(c2-c1)) = 2(c2-c1) = d_dist_x
            add_grad(s1.center_idx, d_dist_x + dr1_c_x, d_dist_y + dr1_c_y)
            # Dist term for C2: -(2(c2-c1)) = -d_dist_x
            add_grad(s2.center_idx, -d_dist_x + dr2_c_x, -d_dist_y + dr2_c_y)

        return grad

    def get_visuals(
        self,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
    ) -> Optional[Tuple[float, float, Optional[float], Optional[float]]]:
        """Calculates screen position and angles for visualization."""
        e1 = reg.get_entity(self.e1_id)
        e2 = reg.get_entity(self.e2_id)
        if not (e1 and e2):
            return None

        # --- Case 1: Line-Line ---
        if isinstance(e1, Line) and isinstance(e2, Line):
            return self._get_line_line_visuals(e1, e2, reg, to_screen)

        # --- Case 2: Line-Shape ---
        line, shape = (e1, e2) if isinstance(e1, Line) else (e2, e1)
        if isinstance(line, Line) and isinstance(shape, (Arc, Circle)):
            return self._get_line_shape_visuals(line, shape, reg, to_screen)

        # --- Case 3: Shape-Shape ---
        if isinstance(e1, (Arc, Circle)) and isinstance(e2, (Arc, Circle)):
            return self._get_shape_shape_visuals(e1, e2, reg, to_screen)

        return None

    def _get_line_line_visuals(self, l1, l2, reg, to_screen):
        p1 = reg.get_point(l1.p1_idx)
        p2 = reg.get_point(l1.p2_idx)
        p3 = reg.get_point(l2.p1_idx)
        p4 = reg.get_point(l2.p2_idx)
        pt = line_intersection(
            (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)
        )
        if not pt:
            m1x, m1y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
            m2x, m2y = (p3.x + p4.x) / 2, (p3.y + p4.y) / 2
            pt = ((m1x + m2x) / 2, (m1y + m2y) / 2)

        ix, iy = pt
        sx, sy = to_screen((ix, iy))
        s_p1, s_p2 = to_screen((p1.x, p1.y)), to_screen((p2.x, p2.y))
        s_p3, s_p4 = to_screen((p3.x, p3.y)), to_screen((p4.x, p4.y))

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

    def _get_line_shape_visuals(self, line, shape, reg, to_screen):
        center = reg.get_point(shape.center_idx)
        lp1, lp2 = reg.get_point(line.p1_idx), reg.get_point(line.p2_idx)
        dxL, dyL = lp2.x - lp1.x, lp2.y - lp1.y
        if math.hypot(dxL, dyL) < 1e-9:
            return None
        ux, uy = dxL / math.hypot(dxL, dyL), dyL / math.hypot(dxL, dyL)

        if isinstance(shape, Arc):
            sp = reg.get_point(shape.start_idx)
        else:
            sp = reg.get_point(shape.radius_pt_idx)
        radius = math.hypot(sp.x - center.x, sp.y - center.y)

        ix1, iy1 = center.x + radius * ux, center.y + radius * uy
        ix2, iy2 = center.x - radius * ux, center.y - radius * uy

        valid_points = []
        for ix, iy in [(ix1, iy1), (ix2, iy2)]:
            on_line = is_point_on_segment(
                (ix, iy), (lp1.x, lp1.y), (lp2.x, lp2.y)
            )
            on_arc = True
            if isinstance(shape, Arc):
                angle = math.atan2(iy - center.y, ix - center.x)
                on_arc = shape.is_angle_within_sweep(angle, reg)
            if on_line and on_arc:
                valid_points.append((ix, iy))

        if valid_points:
            best_pt = valid_points[0]
            if len(valid_points) > 1:
                lmx, lmy = (lp1.x + lp2.x) / 2, (lp1.y + lp2.y) / 2
                d1 = (best_pt[0] - lmx) ** 2 + (best_pt[1] - lmy) ** 2
                d2 = (valid_points[1][0] - lmx) ** 2 + (
                    valid_points[1][1] - lmy
                ) ** 2
                if d2 < d1:
                    best_pt = valid_points[1]
            sx, sy = to_screen(best_pt)
            return sx, sy, None, None

        sx, sy = to_screen((center.x, center.y))
        return sx, sy, None, None

    def _get_shape_shape_visuals(self, s1, s2, reg, to_screen):
        c1, c2 = reg.get_point(s1.center_idx), reg.get_point(s2.center_idx)
        r1 = math.hypot(
            reg.get_point(
                s1.start_idx if isinstance(s1, Arc) else s1.radius_pt_idx
            ).x
            - c1.x,
            reg.get_point(
                s1.start_idx if isinstance(s1, Arc) else s1.radius_pt_idx
            ).y
            - c1.y,
        )
        r2 = math.hypot(
            reg.get_point(
                s2.start_idx if isinstance(s2, Arc) else s2.radius_pt_idx
            ).x
            - c2.x,
            reg.get_point(
                s2.start_idx if isinstance(s2, Arc) else s2.radius_pt_idx
            ).y
            - c2.y,
        )

        intersections = circle_circle_intersection(
            (c1.x, c1.y), r1, (c2.x, c2.y), r2
        )
        if not intersections:
            return None

        valid_points = []
        for ix, iy in intersections:
            on_s1 = (
                s1.is_angle_within_sweep(math.atan2(iy - c1.y, ix - c1.x), reg)
                if isinstance(s1, Arc)
                else True
            )
            on_s2 = (
                s2.is_angle_within_sweep(math.atan2(iy - c2.y, ix - c2.x), reg)
                if isinstance(s2, Arc)
                else True
            )
            if on_s1 and on_s2:
                valid_points.append((ix, iy))

        if not valid_points:
            sx, sy = to_screen(intersections[0])
            return sx, sy, None, None

        best_pt = valid_points[0]
        if len(valid_points) > 1:
            m1 = s1.get_midpoint(reg) if isinstance(s1, Arc) else None
            m2 = s2.get_midpoint(reg) if isinstance(s2, Arc) else None
            if m1 and m2:
                d1 = (
                    (valid_points[0][0] - m1[0]) ** 2
                    + (valid_points[0][1] - m1[1]) ** 2
                    + (valid_points[0][0] - m2[0]) ** 2
                    + (valid_points[0][1] - m2[1]) ** 2
                )
                d2 = (
                    (valid_points[1][0] - m1[0]) ** 2
                    + (valid_points[1][1] - m1[1]) ** 2
                    + (valid_points[1][0] - m2[0]) ** 2
                    + (valid_points[1][1] - m2[1]) ** 2
                )
                if d2 < d1:
                    best_pt = valid_points[1]

        sx, sy = to_screen(best_pt)
        return sx, sy, None, None

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        data = self.get_visuals(reg, to_screen)
        if not data:
            return False

        cx, cy, ang1, ang2 = data

        if ang1 is not None and ang2 is not None:
            # Case 1: Line-Line (Angles are provided)
            # The marker is an arc with a dot. We hit-test the specific dot
            # location.
            visual_radius = 16.0  # Matches renderer.py
            diff = ang2 - ang1

            # Normalize angle difference to [-pi, pi]
            while diff <= -math.pi:
                diff += 2 * math.pi
            while diff > math.pi:
                diff -= 2 * math.pi

            mid_angle = ang1 + diff / 2
            # The dot is drawn at 0.6 * radius
            target_x = cx + math.cos(mid_angle) * visual_radius * 0.6
            target_y = cy + math.sin(mid_angle) * visual_radius * 0.6

            return math.hypot(sx - target_x, sy - target_y) < threshold
        else:
            # Case 2: Line-Arc / Arc-Arc (Box style)
            # The marker is a square box at the intersection/anchor.
            return math.hypot(sx - cx, sy - cy) < threshold

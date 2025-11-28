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
)
from .entities import Line, Arc, Circle
from .params import ParameterContext
from rayforge.core.geo.primitives import (
    line_intersection,
    circle_circle_intersection,
    is_point_on_segment,
)


if TYPE_CHECKING:
    from .entities import EntityRegistry


class Constraint:
    """Base class for all geometric constraints."""

    def error(
        self, reg: "EntityRegistry", params: ParameterContext
    ) -> Union[float, Tuple[float, ...], List[float]]:
        """Calculates the error of the constraint."""
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the constraint to a dictionary."""
        return {}  # Default for non-serializable constraints like Drag

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        """Checks if the constraint's visual representation is hit."""
        return False


class DistanceConstraint(Constraint):
    """Enforces distance between two points."""

    def __init__(self, p1: int, p2: int, value: Union[str, float]):
        self.p1 = p1
        self.p2 = p2
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "DistanceConstraint",
            "p1": self.p1,
            "p2": self.p2,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistanceConstraint":
        return cls(p1=data["p1"], p2=data["p2"], value=data["value"])

    def error(self, reg: "EntityRegistry", params: ParameterContext) -> float:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        target = params.evaluate(self.value)

        # Use squared distances to avoid sqrt, which is better for the solver
        dist_sq = (pt2.x - pt1.x) ** 2 + (pt2.y - pt1.y) ** 2
        return dist_sq - target**2

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        p1 = reg.get_point(self.p1)
        p2 = reg.get_point(self.p2)
        if p1 and p2:
            s1 = to_screen((p1.x, p1.y))
            s2 = to_screen((p2.x, p2.y))
            mx, my = (s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2
            return math.hypot(sx - mx, sy - my) < 15
        return False


class EqualDistanceConstraint(Constraint):
    """Enforces that distance(p1, p2) equals distance(p3, p4)."""

    def __init__(self, p1: int, p2: int, p3: int, p4: int):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "EqualDistanceConstraint",
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
            "p4": self.p4,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EqualDistanceConstraint":
        return cls(p1=data["p1"], p2=data["p2"], p3=data["p3"], p4=data["p4"])

    def error(self, reg: "EntityRegistry", params: ParameterContext) -> float:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        dist1_sq = (pt2.x - pt1.x) ** 2 + (pt2.y - pt1.y) ** 2

        pt3 = reg.get_point(self.p3)
        pt4 = reg.get_point(self.p4)
        dist2_sq = (pt4.x - pt3.x) ** 2 + (pt4.y - pt3.y) ** 2

        return dist1_sq - dist2_sq


class HorizontalConstraint(Constraint):
    """Enforces two points have the same Y coordinate."""

    def __init__(self, p1: int, p2: int):
        self.p1 = p1
        self.p2 = p2

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "HorizontalConstraint", "p1": self.p1, "p2": self.p2}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HorizontalConstraint":
        return cls(p1=data["p1"], p2=data["p2"])

    def error(self, reg: "EntityRegistry", params: ParameterContext) -> float:
        return reg.get_point(self.p1).y - reg.get_point(self.p2).y

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        p1 = reg.get_point(self.p1)
        p2 = reg.get_point(self.p2)
        if p1 and p2:
            s1 = to_screen((p1.x, p1.y))
            s2 = to_screen((p2.x, p2.y))

            t = 0.2
            mx = s1[0] + (s2[0] - s1[0]) * t
            my = s1[1] + (s2[1] - s1[1]) * t
            cx = mx
            cy = my - 10
            return math.hypot(sx - cx, sy - cy) < threshold
        return False


class VerticalConstraint(Constraint):
    """Enforces two points have the same X coordinate."""

    def __init__(self, p1: int, p2: int):
        self.p1 = p1
        self.p2 = p2

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "VerticalConstraint", "p1": self.p1, "p2": self.p2}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerticalConstraint":
        return cls(p1=data["p1"], p2=data["p2"])

    def error(self, reg: "EntityRegistry", params: ParameterContext) -> float:
        return reg.get_point(self.p1).x - reg.get_point(self.p2).x

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        p1 = reg.get_point(self.p1)
        p2 = reg.get_point(self.p2)
        if p1 and p2:
            s1 = to_screen((p1.x, p1.y))
            s2 = to_screen((p2.x, p2.y))

            t = 0.2
            mx = s1[0] + (s2[0] - s1[0]) * t
            my = s1[1] + (s2[1] - s1[1]) * t
            cx = mx + 10
            cy = my
            return math.hypot(sx - cx, sy - cy) < threshold
        return False


class CoincidentConstraint(Constraint):
    """Enforces two points are at the same location."""

    def __init__(self, p1: int, p2: int):
        self.p1 = p1
        self.p2 = p2

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "CoincidentConstraint", "p1": self.p1, "p2": self.p2}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoincidentConstraint":
        return cls(p1=data["p1"], p2=data["p2"])

    def error(
        self, reg: "EntityRegistry", params: ParameterContext
    ) -> Tuple[float, float]:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        # Return separate X and Y errors for a solver-friendly quadratic form
        return (pt1.x - pt2.x, pt1.y - pt2.y)

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        origin_id = getattr(element.sketch, "origin_id", -1)
        pid_to_check = self.p1
        if self.p1 == origin_id and origin_id != -1:
            pid_to_check = self.p2

        pt_to_check = reg.get_point(pid_to_check)
        if pt_to_check:
            s_pt = to_screen((pt_to_check.x, pt_to_check.y))
            return math.hypot(sx - s_pt[0], sy - s_pt[1]) < threshold
        return False


class PointOnLineConstraint(Constraint):
    """Enforces a point lies on the infinite geometry of a shape."""

    def __init__(self, point_id: int, shape_id: int):
        self.point_id = point_id
        self.shape_id = shape_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "PointOnLineConstraint",
            "point_id": self.point_id,
            "shape_id": self.shape_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PointOnLineConstraint":
        return cls(point_id=data["point_id"], shape_id=data["shape_id"])

    def error(self, reg: "EntityRegistry", params: ParameterContext) -> float:
        pt = reg.get_point(self.point_id)
        shape = reg.get_entity(self.shape_id)

        if isinstance(shape, Line):
            l1 = reg.get_point(shape.p1_idx)
            l2 = reg.get_point(shape.p2_idx)

            # Use the 2D cross-product of vectors (pt - l1) and (l2 - l1).
            return (l2.x - l1.x) * (pt.y - l1.y) - (pt.x - l1.x) * (
                l2.y - l1.y
            )

        elif isinstance(shape, (Arc, Circle)):
            center = reg.get_point(shape.center_idx)
            radius_sq = 0.0
            if isinstance(shape, Arc):
                start = reg.get_point(shape.start_idx)
                radius_sq = (start.x - center.x) ** 2 + (
                    start.y - center.y
                ) ** 2
            elif isinstance(shape, Circle):
                radius_pt = reg.get_point(shape.radius_pt_idx)
                radius_sq = (radius_pt.x - center.x) ** 2 + (
                    radius_pt.y - center.y
                ) ** 2

            dist_to_point_sq = (pt.x - center.x) ** 2 + (pt.y - center.y) ** 2
            # Error is diff in squared distances
            # (dist_from_center^2 - radius^2)
            return dist_to_point_sq - radius_sq

        return 0.0

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        pt = reg.get_point(self.point_id)
        if pt:
            s_pt = to_screen((pt.x, pt.y))
            return math.hypot(sx - s_pt[0], sy - s_pt[1]) < threshold
        return False


class RadiusConstraint(Constraint):
    """Enforces radius of an Arc or Circle."""

    def __init__(self, entity_id: int, radius: Union[str, float]):
        self.entity_id = entity_id
        self.value = radius

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "RadiusConstraint",
            "entity_id": self.entity_id,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RadiusConstraint":
        return cls(entity_id=data["entity_id"], radius=data["value"])

    def error(self, reg: "EntityRegistry", params: ParameterContext) -> float:
        entity = reg.get_entity(self.entity_id)
        target = params.evaluate(self.value)
        curr_r_sq = 0.0

        if isinstance(entity, Arc):
            center = reg.get_point(entity.center_idx)
            start = reg.get_point(entity.start_idx)
            curr_r_sq = (start.x - center.x) ** 2 + (start.y - center.y) ** 2
        elif isinstance(entity, Circle):
            center = reg.get_point(entity.center_idx)
            radius_pt = reg.get_point(entity.radius_pt_idx)
            curr_r_sq = (radius_pt.x - center.x) ** 2 + (
                radius_pt.y - center.y
            ) ** 2
        else:
            return 0.0

        return curr_r_sq - target**2

    def get_label_pos(
        self,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
    ):
        """Calculates screen position for Radius/Diameter constraint labels."""
        entity = reg.get_entity(self.entity_id)
        if not isinstance(entity, (Arc, Circle)):
            return None

        center = reg.get_point(entity.center_idx)
        if not center:
            return None

        radius, mid_angle = 0.0, 0.0

        if isinstance(entity, Arc):
            start = reg.get_point(entity.start_idx)
            if not start:
                return None
            radius = math.hypot(start.x - center.x, start.y - center.y)
            midpoint = entity.get_midpoint(reg)
            if not midpoint:
                return None
            mid_angle = math.atan2(
                midpoint[1] - center.y, midpoint[0] - center.x
            )

        elif isinstance(entity, Circle):
            radius_pt = reg.get_point(entity.radius_pt_idx)
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
        label_sx, label_sy = to_screen((label_mx, label_my))

        # Position on the arc for the leader line
        arc_mid_mx = center.x + radius * math.cos(mid_angle)
        arc_mid_my = center.y + radius * math.sin(mid_angle)
        arc_mid_sx, arc_mid_sy = to_screen((arc_mid_mx, arc_mid_my))

        return label_sx, label_sy, arc_mid_sx, arc_mid_sy

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        pos_data = self.get_label_pos(reg, to_screen, element)
        if pos_data:
            label_sx, label_sy, _, _ = pos_data
            return math.hypot(sx - label_sx, sy - label_sy) < 15
        return False


class DiameterConstraint(Constraint):
    """Enforces the diameter of a Circle."""

    def __init__(self, circle_id: int, diameter: Union[str, float]):
        self.circle_id = circle_id
        self.value = diameter

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "DiameterConstraint",
            "circle_id": self.circle_id,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiameterConstraint":
        return cls(circle_id=data["circle_id"], diameter=data["value"])

    def error(self, reg: "EntityRegistry", params: ParameterContext) -> float:
        circle_entity = reg.get_entity(self.circle_id)

        if not isinstance(circle_entity, Circle):
            return 0.0

        center = reg.get_point(circle_entity.center_idx)
        radius_pt = reg.get_point(circle_entity.radius_pt_idx)
        target_diameter = params.evaluate(self.value)

        # Error = 4 * r^2 - d^2
        curr_r_sq = (radius_pt.x - center.x) ** 2 + (
            radius_pt.y - center.y
        ) ** 2
        return 4 * curr_r_sq - target_diameter**2

    def get_label_pos(
        self,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
    ):
        # Delegate to RadiusConstraint's logic as it is identical
        temp_radius_constr = RadiusConstraint(self.circle_id, 0)
        return temp_radius_constr.get_label_pos(reg, to_screen, element)

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        pos_data = self.get_label_pos(reg, to_screen, element)
        if pos_data:
            label_sx, label_sy, _, _ = pos_data
            return math.hypot(sx - label_sx, sy - label_sy) < 15
        return False


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

    def error(self, reg: "EntityRegistry", params: ParameterContext) -> float:
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
            center = reg.get_point(shape.center_idx)

            # Use the 2D cross-product of vectors
            #   (lp2 - lp1) and (center - lp1).
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
        if data:
            vx, vy, _, _ = data
            return math.hypot(sx - vx, sy - vy) < 20
        return False


class TangentConstraint(Constraint):
    """
    Enforces tangency between a Line and an Arc/Circle.
    Logic: Distance from shape center to Line equals shape Radius.
    """

    def __init__(self, line_id: int, shape_id: int):
        self.line_id = line_id
        self.shape_id = shape_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "TangentConstraint",
            "line_id": self.line_id,
            "shape_id": self.shape_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TangentConstraint":
        return cls(line_id=data["line_id"], shape_id=data["shape_id"])

    def error(self, reg: "EntityRegistry", params: ParameterContext) -> float:
        line = reg.get_entity(self.line_id)
        shape = reg.get_entity(self.shape_id)

        if not isinstance(line, Line) or not isinstance(shape, (Arc, Circle)):
            return 0.0

        # Shape Data
        center = reg.get_point(shape.center_idx)
        radius_sq = 0.0
        if isinstance(shape, Arc):
            start = reg.get_point(shape.start_idx)
            radius_sq = (start.x - center.x) ** 2 + (start.y - center.y) ** 2
        elif isinstance(shape, Circle):
            radius_pt = reg.get_point(shape.radius_pt_idx)
            radius_sq = (radius_pt.x - center.x) ** 2 + (
                radius_pt.y - center.y
            ) ** 2

        # Line Data
        lp1 = reg.get_point(line.p1_idx)
        lp2 = reg.get_point(line.p2_idx)

        # Vector of the line
        line_dx = lp2.x - lp1.x
        line_dy = lp2.y - lp1.y
        line_len_sq = line_dx**2 + line_dy**2

        if line_len_sq < 1e-18:  # line has zero length
            # Fallback to error as distance from center to one of the line's
            # points
            dist_to_pt_sq = (lp1.x - center.x) ** 2 + (lp1.y - center.y) ** 2
            return dist_to_pt_sq - radius_sq

        # 2D Cross Product (Area of parallelogram)
        # magnitude = |line_len| * |dist_to_line|
        cross_product = (
            line_dx * (lp1.y - center.y) - (lp1.x - center.x) * line_dy
        )

        # We want dist_from_center_to_line^2 == radius^2.
        # (cross_product / line_len)^2 == radius_sq
        # cross_product^2 / line_len_sq == radius_sq
        #
        # To avoid division by line_len_sq which causes instability:
        # cross_product^2 - radius_sq * line_len_sq = 0
        return cross_product**2 - (radius_sq * line_len_sq)


class EqualLengthConstraint(Constraint):
    """
    Enforces that all entities in a set have the same characteristic length.
    - Line: Length
    - Arc/Circle: Radius
    """

    def __init__(self, entity_ids: List[int]):
        self.entity_ids = entity_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "EqualLengthConstraint",
            "entity_ids": self.entity_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EqualLengthConstraint":
        return cls(entity_ids=data["entity_ids"])

    def _get_length_sq(self, entity, reg: "EntityRegistry") -> float:
        if isinstance(entity, Line):
            p1 = reg.get_point(entity.p1_idx)
            p2 = reg.get_point(entity.p2_idx)
            return (p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2
        elif isinstance(entity, Arc):
            c = reg.get_point(entity.center_idx)
            s = reg.get_point(entity.start_idx)
            return (s.x - c.x) ** 2 + (s.y - c.y) ** 2
        elif isinstance(entity, Circle):
            c = reg.get_point(entity.center_idx)
            r = reg.get_point(entity.radius_pt_idx)
            return (r.x - c.x) ** 2 + (r.y - c.y) ** 2
        return 0.0

    def error(
        self, reg: "EntityRegistry", params: ParameterContext
    ) -> List[float]:
        if len(self.entity_ids) < 2:
            return []

        entities = [reg.get_entity(eid) for eid in self.entity_ids]
        if any(e is None for e in entities):
            return []

        # All lengths should equal the length of the first entity.
        base_len_sq = self._get_length_sq(entities[0], reg)
        errors = []
        for i in range(1, len(entities)):
            other_len_sq = self._get_length_sq(entities[i], reg)
            errors.append(other_len_sq - base_len_sq)
        return errors

    def _get_symbol_pos(
        self,
        entity,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
    ):
        """Calculates screen pos for an equality symbol on an entity."""
        # 1. Get anchor point (mid_x, mid_y) and normal_angle in MODEL space
        mid_x, mid_y, normal_angle = 0.0, 0.0, 0.0

        if isinstance(entity, Line):
            p1 = reg.get_point(entity.p1_idx)
            p2 = reg.get_point(entity.p2_idx)
            mid_x = (p1.x + p2.x) / 2.0
            mid_y = (p1.y + p2.y) / 2.0
            tangent_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            normal_angle = tangent_angle - (math.pi / 2.0)
        elif isinstance(entity, (Arc, Circle)):
            midpoint = entity.get_midpoint(reg)
            if not midpoint:
                return None
            mid_x, mid_y = midpoint
            center = reg.get_point(entity.center_idx)
            normal_angle = math.atan2(mid_y - center.y, mid_x - center.x)

        scale = 1.0
        if element.canvas and hasattr(element.canvas, "get_view_scale"):
            scale, _ = element.canvas.get_view_scale()
            scale = max(scale, 1e-9)
        offset_dist_model = 15.0 / scale
        final_x = mid_x + offset_dist_model * math.cos(normal_angle)
        final_y = mid_y + offset_dist_model * math.sin(normal_angle)
        return to_screen((final_x, final_y))

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        for entity_id in self.entity_ids:
            entity = reg.get_entity(entity_id)
            if not entity:
                continue
            pos = self._get_symbol_pos(entity, reg, to_screen, element)
            if pos:
                esx, esy = pos
                if math.hypot(sx - esx, sy - esy) < 15:
                    return True
        return False


class SymmetryConstraint(Constraint):
    """
    Enforces symmetry between two points (p1, p2) with respect to:
    1. A Center Point (Point Symmetry)
    2. An Axis Line (Line Symmetry)
    """

    def __init__(
        self,
        p1: int,
        p2: int,
        center: Optional[int] = None,
        axis: Optional[int] = None,
    ):
        self.p1 = p1
        self.p2 = p2
        self.center = center
        self.axis = axis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "SymmetryConstraint",
            "p1": self.p1,
            "p2": self.p2,
            "center": self.center,
            "axis": self.axis,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymmetryConstraint":
        return cls(
            p1=data["p1"],
            p2=data["p2"],
            center=data.get("center"),
            axis=data.get("axis"),
        )

    def error(
        self, reg: "EntityRegistry", params: ParameterContext
    ) -> List[float]:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)

        if self.center is not None:
            # Case 1: Point Symmetry
            # Constraint: Center is the midpoint of P1 and P2
            # (P1 + P2) / 2 = Center  =>  P1 + P2 - 2*Center = 0
            s = reg.get_point(self.center)
            return [
                (pt1.x + pt2.x) - 2 * s.x,
                (pt1.y + pt2.y) - 2 * s.y,
            ]

        elif self.axis is not None:
            # Case 2: Line Symmetry
            # Constraint A: The segment P1-P2 is perpendicular to the Axis Line
            # Constraint B: The midpoint of P1-P2 lies on the Axis Line
            line = reg.get_entity(self.axis)
            if not isinstance(line, Line):
                return [0.0, 0.0]

            l1 = reg.get_point(line.p1_idx)
            l2 = reg.get_point(line.p2_idx)

            # Vector of the Axis Line
            dx_l = l2.x - l1.x
            dy_l = l2.y - l1.y

            # 1. Perpendicularity: Dot product (P2 - P1) . (L2 - L1) = 0
            dx_p = pt2.x - pt1.x
            dy_p = pt2.y - pt1.y
            err_perp = dx_p * dx_l + dy_p * dy_l

            # 2. Midpoint on Line: Cross product (Mid - L1) x (L2 - L1) = 0
            mx = (pt1.x + pt2.x) * 0.5
            my = (pt1.y + pt2.y) * 0.5
            err_collinear = (mx - l1.x) * dy_l - (my - l1.y) * dx_l

            return [err_perp, err_collinear]

        return [0.0, 0.0]

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        p1 = reg.get_point(self.p1)
        p2 = reg.get_point(self.p2)
        if not (p1 and p2):
            return False
        s1 = to_screen((p1.x, p1.y))
        s2 = to_screen((p2.x, p2.y))
        mx = (s1[0] + s2[0]) / 2.0
        my = (s1[1] + s2[1]) / 2.0
        angle = math.atan2(s2[1] - s1[1], s2[0] - s1[0])
        offset = 12.0
        lx = mx - offset * math.cos(angle)
        ly = my - offset * math.sin(angle)
        rx = mx + offset * math.cos(angle)
        ry = my + offset * math.sin(angle)
        if math.hypot(sx - lx, sy - ly) < threshold:
            return True
        if math.hypot(sx - rx, sy - ry) < threshold:
            return True
        return False


class DragConstraint(Constraint):
    """
    A transient constraint used only during interaction.
    It pulls a point toward a target (mouse) coordinate.
    """

    def __init__(
        self,
        point_id: int,
        target_x: float,
        target_y: float,
        weight: float = 0.1,
    ):
        self.point_id = point_id
        self.target_x = target_x
        self.target_y = target_y
        # Weight controls how strongly this constraint pulls vs geometric
        # constraints. It should be << 1.0 to prevent breaking geometry,
        # as geometric constraints have an implicit weight of 1.0.
        self.weight = weight

    def error(
        self, reg: "EntityRegistry", params: ParameterContext
    ) -> Tuple[float, float]:
        p = reg.get_point(self.point_id)
        # Return separate errors for X and Y components. This creates a
        # quadratic objective function (sum of squares) which is much
        # friendlier to the solver than one based on hypot().
        err_x = (p.x - self.target_x) * self.weight
        err_y = (p.y - self.target_y) * self.weight
        return err_x, err_y

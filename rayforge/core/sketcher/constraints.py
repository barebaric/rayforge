from typing import Protocol, Union, Tuple
from .entities import EntityRegistry, Line, Arc
from .params import ParameterContext


class Constraint(Protocol):
    """Interface for all geometric constraints."""

    def error(
        self, reg: EntityRegistry, params: ParameterContext
    ) -> Union[float, Tuple[float, float]]: ...


class DistanceConstraint:
    """Enforces distance between two points."""

    def __init__(self, p1: int, p2: int, value: Union[str, float]):
        self.p1 = p1
        self.p2 = p2
        self.value = value

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        target = params.evaluate(self.value)

        # Use squared distances to avoid sqrt, which is better for the solver
        dist_sq = (pt2.x - pt1.x) ** 2 + (pt2.y - pt1.y) ** 2
        return dist_sq - target**2


class EqualDistanceConstraint:
    """Enforces that distance(p1, p2) equals distance(p3, p4)."""

    def __init__(self, p1: int, p2: int, p3: int, p4: int):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        dist1_sq = (pt2.x - pt1.x) ** 2 + (pt2.y - pt1.y) ** 2

        pt3 = reg.get_point(self.p3)
        pt4 = reg.get_point(self.p4)
        dist2_sq = (pt4.x - pt3.x) ** 2 + (pt4.y - pt3.y) ** 2

        return dist1_sq - dist2_sq


class HorizontalConstraint:
    """Enforces two points have the same Y coordinate."""

    def __init__(self, p1: int, p2: int):
        self.p1 = p1
        self.p2 = p2

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        return reg.get_point(self.p1).y - reg.get_point(self.p2).y


class VerticalConstraint:
    """Enforces two points have the same X coordinate."""

    def __init__(self, p1: int, p2: int):
        self.p1 = p1
        self.p2 = p2

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        return reg.get_point(self.p1).x - reg.get_point(self.p2).x


class CoincidentConstraint:
    """Enforces two points are at the same location."""

    def __init__(self, p1: int, p2: int):
        self.p1 = p1
        self.p2 = p2

    def error(
        self, reg: EntityRegistry, params: ParameterContext
    ) -> Tuple[float, float]:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        # Return separate X and Y errors for a solver-friendly quadratic form
        return (pt1.x - pt2.x, pt1.y - pt2.y)


class PointOnLineConstraint:
    """Enforces a point lies on the infinite line defined by a Line entity."""

    def __init__(self, point_id: int, line_id: int):
        self.point_id = point_id
        self.line_id = line_id

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        pt = reg.get_point(self.point_id)
        line = reg.get_entity(self.line_id)

        if not isinstance(line, Line):
            return 0.0

        l1 = reg.get_point(line.p1_idx)
        l2 = reg.get_point(line.p2_idx)

        # Use the 2D cross-product of vectors (pt - l1) and (l2 - l1).
        # This value is 0 when the point is on the infinite line passing
        # through l1 and l2. It is also twice the signed area of the
        # triangle formed by the three points.
        # This formulation avoids sqrt and abs(), making it much friendlier
        # for the non-linear solver.
        return (l2.x - l1.x) * (pt.y - l1.y) - (pt.x - l1.x) * (l2.y - l1.y)


class RadiusConstraint:
    """Enforces distance between center and start point of an arc."""

    def __init__(self, arc_id: int, radius: Union[str, float]):
        self.arc_id = arc_id
        self.value = radius

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        # O(1) lookup
        arc_entity = reg.get_entity(self.arc_id)

        if not isinstance(arc_entity, Arc):
            return 0.0

        center = reg.get_point(arc_entity.center_idx)
        start = reg.get_point(arc_entity.start_idx)
        target = params.evaluate(self.value)

        # Use squared distances to avoid sqrt, which is better for the solver
        curr_r_sq = (start.x - center.x) ** 2 + (start.y - center.y) ** 2
        return curr_r_sq - target**2


class PerpendicularConstraint:
    """Enforces two lines are perpendicular (dot product is 0)."""

    def __init__(self, l1_id: int, l2_id: int):
        self.l1_id = l1_id
        self.l2_id = l2_id

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        l1 = reg.get_entity(self.l1_id)
        l2 = reg.get_entity(self.l2_id)

        if not isinstance(l1, Line) or not isinstance(l2, Line):
            return 0.0

        p1 = reg.get_point(l1.p1_idx)
        p2 = reg.get_point(l1.p2_idx)
        p3 = reg.get_point(l2.p1_idx)
        p4 = reg.get_point(l2.p2_idx)

        dx1, dy1 = p2.x - p1.x, p2.y - p1.y
        dx2, dy2 = p4.x - p3.x, p4.y - p3.y

        # Dot product
        return dx1 * dx2 + dy1 * dy2


class TangentConstraint:
    """
    Enforces tangency between a Line and an Arc/Circle.
    Logic: Distance from Arc center to Line equals Arc Radius.
    """

    def __init__(self, line_id: int, arc_id: int):
        self.line_id = line_id
        self.arc_id = arc_id

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        line = reg.get_entity(self.line_id)
        arc = reg.get_entity(self.arc_id)

        if not isinstance(line, Line) or not isinstance(arc, Arc):
            return 0.0

        # Arc Data
        center = reg.get_point(arc.center_idx)
        start = reg.get_point(arc.start_idx)
        radius_sq = (start.x - center.x) ** 2 + (start.y - center.y) ** 2

        # Line Data
        lp1 = reg.get_point(line.p1_idx)
        lp2 = reg.get_point(line.p2_idx)

        # We want dist_from_center_to_line^2 == radius^2.
        # This avoids sqrt() and abs() for better solver performance.
        line_dx = lp2.x - lp1.x
        line_dy = lp2.y - lp1.y
        line_len_sq = line_dx**2 + line_dy**2

        if line_len_sq < 1e-18:  # line has zero length
            # Error is distance from center to one of the line's points
            dist_to_pt_sq = (lp1.x - center.x) ** 2 + (lp1.y - center.y) ** 2
            return dist_to_pt_sq - radius_sq

        # This is the squared numerator of the point-to-line distance formula
        cross_product = (
            line_dx * (lp1.y - center.y) - (lp1.x - center.x) * line_dy
        )

        dist_to_line_sq = cross_product**2 / line_len_sq
        return dist_to_line_sq - radius_sq


class DragConstraint:
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
        self, reg: EntityRegistry, params: ParameterContext
    ) -> Tuple[float, float]:
        p = reg.get_point(self.point_id)
        # Return separate errors for X and Y components. This creates a
        # quadratic objective function (sum of squares) which is much
        # friendlier to the solver than one based on hypot().
        err_x = (p.x - self.target_x) * self.weight
        err_y = (p.y - self.target_y) * self.weight
        return err_x, err_y

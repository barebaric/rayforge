from typing import Protocol, Union, Tuple, Dict, Any, List
from .entities import EntityRegistry, Line, Arc, Circle
from .params import ParameterContext


class Constraint(Protocol):
    """Interface for all geometric constraints."""

    def error(
        self, reg: EntityRegistry, params: ParameterContext
    ) -> Union[float, Tuple[float, ...], List[float]]: ...

    def to_dict(self) -> Dict[str, Any]: ...


class DistanceConstraint:
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

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "HorizontalConstraint", "p1": self.p1, "p2": self.p2}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HorizontalConstraint":
        return cls(p1=data["p1"], p2=data["p2"])

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        return reg.get_point(self.p1).y - reg.get_point(self.p2).y


class VerticalConstraint:
    """Enforces two points have the same X coordinate."""

    def __init__(self, p1: int, p2: int):
        self.p1 = p1
        self.p2 = p2

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "VerticalConstraint", "p1": self.p1, "p2": self.p2}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerticalConstraint":
        return cls(p1=data["p1"], p2=data["p2"])

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        return reg.get_point(self.p1).x - reg.get_point(self.p2).x


class CoincidentConstraint:
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
        self, reg: EntityRegistry, params: ParameterContext
    ) -> Tuple[float, float]:
        pt1 = reg.get_point(self.p1)
        pt2 = reg.get_point(self.p2)
        # Return separate X and Y errors for a solver-friendly quadratic form
        return (pt1.x - pt2.x, pt1.y - pt2.y)


class PointOnLineConstraint:
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
        # Handle legacy "line_id" key for backward compatibility
        shape_id = data.get("shape_id", data.get("line_id"))
        if shape_id is None:
            raise KeyError(
                "PointOnLineConstraint data missing 'shape_id' or 'line_id'"
            )
        return cls(point_id=data["point_id"], shape_id=shape_id)

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
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


class RadiusConstraint:
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
        # Handle legacy "arc_id" key for backward compatibility
        entity_id = data.get("entity_id", data.get("arc_id"))
        if entity_id is None:
            raise KeyError(
                "RadiusConstraint data missing 'entity_id' or 'arc_id'"
            )
        return cls(entity_id=entity_id, radius=data["value"])

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
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


class DiameterConstraint:
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

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
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


class PerpendicularConstraint:
    """Enforces two lines are perpendicular (dot product is 0)."""

    def __init__(self, l1_id: int, l2_id: int):
        self.l1_id = l1_id
        self.l2_id = l2_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "PerpendicularConstraint",
            "l1_id": self.l1_id,
            "l2_id": self.l2_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerpendicularConstraint":
        return cls(l1_id=data["l1_id"], l2_id=data["l2_id"])

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
        # Handle legacy "arc_id" key for backward compatibility
        shape_id = data.get("shape_id", data.get("arc_id"))
        if shape_id is None:
            raise KeyError(
                "TangentConstraint data missing 'shape_id' or 'arc_id'"
            )
        return cls(line_id=data["line_id"], shape_id=shape_id)

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
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


class EqualLengthConstraint:
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
        # Handle legacy binary constraint for backward compatibility
        e1 = data.get("e1_id")
        e2 = data.get("e2_id")
        if e1 is not None and e2 is not None:
            return cls(entity_ids=[e1, e2])
        return cls(entity_ids=data["entity_ids"])

    def _get_length_sq(self, entity, reg: EntityRegistry) -> float:
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
        self, reg: EntityRegistry, params: ParameterContext
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

    def to_dict(self) -> Dict[str, Any]:
        """Drag constraints are transient and should not be serialized."""
        return {}

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

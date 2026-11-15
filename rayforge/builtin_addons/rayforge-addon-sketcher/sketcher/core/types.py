from typing import TypeAlias

from raygeo.sketch.constraints.angle import AngleConstraint
from raygeo.sketch.constraints.aspect_ratio import AspectRatioConstraint
from raygeo.sketch.constraints.coincident import CoincidentConstraint
from raygeo.sketch.constraints.collinear import CollinearConstraint
from raygeo.sketch.constraints.diameter import DiameterConstraint
from raygeo.sketch.constraints.distance import DistanceConstraint
from raygeo.sketch.constraints.drag import DragConstraint
from raygeo.sketch.constraints.equal_distance import EqualDistanceConstraint
from raygeo.sketch.constraints.equal_length import EqualLengthConstraint
from raygeo.sketch.constraints.horizontal import HorizontalConstraint
from raygeo.sketch.constraints.parallelogram import ParallelogramConstraint
from raygeo.sketch.constraints.perpendicular import PerpendicularConstraint
from raygeo.sketch.constraints.point_on_curve import PointOnCurveConstraint
from raygeo.sketch.constraints.point_on_line import PointOnLineConstraint
from raygeo.sketch.constraints.radius import RadiusConstraint
from raygeo.sketch.constraints.rotational import RotationalConstraint
from raygeo.sketch.constraints.symmetry import SymmetryConstraint
from raygeo.sketch.constraints.tangent import TangentConstraint
from raygeo.sketch.constraints.vertical import VerticalConstraint
from raygeo.sketch.entities.arc import Arc
from raygeo.sketch.entities.bezier import Bezier
from raygeo.sketch.entities.circle import Circle
from raygeo.sketch.entities.ellipse import Ellipse
from raygeo.sketch.entities.line import Line
from raygeo.sketch.entities.polygon import Polygon
from raygeo.sketch.entities.text_box import TextBox

EntityID: TypeAlias = int

Entity: TypeAlias = Line | Arc | Circle | Ellipse | Bezier | Polygon | TextBox

Constraint: TypeAlias = (
    AngleConstraint
    | AspectRatioConstraint
    | CoincidentConstraint
    | CollinearConstraint
    | DiameterConstraint
    | DistanceConstraint
    | DragConstraint
    | EqualDistanceConstraint
    | EqualLengthConstraint
    | HorizontalConstraint
    | ParallelogramConstraint
    | PerpendicularConstraint
    | PointOnCurveConstraint
    | PointOnLineConstraint
    | RadiusConstraint
    | RotationalConstraint
    | SymmetryConstraint
    | TangentConstraint
    | VerticalConstraint
)

HitTarget: TypeAlias = Entity | Constraint | EntityID | None

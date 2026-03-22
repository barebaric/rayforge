"""
Geometric constraints for the 2D CAD sketcher.
"""

from typing import Dict, Type
from .base import Constraint, ConstraintStatus
from .angle import AngleConstraint, ANGLE_WEIGHT
from .aspect_ratio import AspectRatioConstraint
from .coincident import CoincidentConstraint
from .collinear import CollinearConstraint
from .diameter import DiameterConstraint
from .distance import DistanceConstraint
from .drag import DragConstraint
from .equal_distance import EqualDistanceConstraint
from .equal_length import EqualLengthConstraint
from .horizontal import HorizontalConstraint
from .parallelogram import ParallelogramConstraint
from .perpendicular import PerpendicularConstraint
from .point_on_line import PointOnLineConstraint
from .radius import RadiusConstraint
from .symmetry import SymmetryConstraint
from .tangent import TangentConstraint
from .vertical import VerticalConstraint


CONSTRAINT_TYPE_MAP: Dict[str, Type[Constraint]] = {
    "horiz": HorizontalConstraint,
    "vert": VerticalConstraint,
    "dist": DistanceConstraint,
    "radius": RadiusConstraint,
    "diameter": DiameterConstraint,
    "perp": PerpendicularConstraint,
    "tangent": TangentConstraint,
    "equal": EqualLengthConstraint,
    "coincident": CoincidentConstraint,
    "point_on_line": PointOnLineConstraint,
    "symmetry": SymmetryConstraint,
    "aspect_ratio": AspectRatioConstraint,
    "angle": AngleConstraint,
}


__all__ = [
    "ANGLE_WEIGHT",
    "CONSTRAINT_TYPE_MAP",
    "Constraint",
    "ConstraintStatus",
    "AngleConstraint",
    "AspectRatioConstraint",
    "CoincidentConstraint",
    "CollinearConstraint",
    "DiameterConstraint",
    "DistanceConstraint",
    "DragConstraint",
    "EqualDistanceConstraint",
    "EqualLengthConstraint",
    "HorizontalConstraint",
    "ParallelogramConstraint",
    "PerpendicularConstraint",
    "PointOnLineConstraint",
    "RadiusConstraint",
    "SymmetryConstraint",
    "TangentConstraint",
    "VerticalConstraint",
]

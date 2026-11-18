import enum
from enum import Enum
from typing import TypeAlias

EntityID: TypeAlias = int


class ConstraintStatus(enum.IntEnum):
    """The validation status of a constraint."""

    VALID = 1
    EXPRESSION_BASED = 2
    ERROR = 3
    CONFLICTING = 4


class WaypointType(str, Enum):
    """Bezier waypoint role."""

    SHARP = "sharp"
    SMOOTH = "smooth"
    SYMMETRIC = "symmetric"

"""
The path module contains shared, stateless utility functions for manipulating
path-like data structures (such as Ops and Geometry). These functions are
generic and have no knowledge of the high-level objects that use them.
"""

from . import analysis
from . import contours
from . import fitting
from . import intersect
from . import linearize
from . import primitives
from . import query
from . import transform
from .constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
    COL_I,
    COL_J,
    COL_CW,
    GEO_ARRAY_COLS,
)
from .geometry import (
    Geometry,
)

__all__ = [
    "analysis",
    "contours",
    "fitting",
    "intersect",
    "linearize",
    "primitives",
    "query",
    "transform",
    "Geometry",
    "CMD_TYPE_MOVE",
    "CMD_TYPE_LINE",
    "CMD_TYPE_ARC",
    "COL_TYPE",
    "COL_X",
    "COL_Y",
    "COL_Z",
    "COL_I",
    "COL_J",
    "COL_CW",
    "GEO_ARRAY_COLS",
]

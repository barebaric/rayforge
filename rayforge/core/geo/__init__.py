"""
The path module contains shared, stateless utility functions for manipulating
path-like data structures (such as Ops and Geometry). These functions are
generic and have no knowledge of the high-level objects that use them.
"""

from . import analysis
from . import linearize
from . import primitives
from . import query
from .geometry import (
    Geometry,
    Command,
    MovingCommand,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
)

__all__ = [
    "analysis",
    "linearize",
    "primitives",
    "query",
    "Geometry",
    "Command",
    "MovingCommand",
    "MoveToCommand",
    "LineToCommand",
    "ArcToCommand",
]

"""
Job origin ("Start From") settings of a document.

The job origin decides where a job runs on the bed. With ``ABSOLUTE``
the job runs exactly where it is placed on the canvas. With
``USER_ORIGIN`` and ``CURRENT_POSITION`` the whole job is translated so
that one of nine points of its bounding box (the anchor) lands on the
zero point of the active work coordinate system or on the reported
head position.
"""

from dataclasses import dataclass
from enum import Enum
from gettext import gettext as _
from typing import Any

from raygeo.geo.types import Point

# A world-space bounding box as (min_x, min_y, max_x, max_y).
BBox = tuple[float, float, float, float]


class StartFrom(Enum):
    """Where a job is placed on the machine."""

    ABSOLUTE = "absolute"
    CURRENT_POSITION = "current_position"
    USER_ORIGIN = "user_origin"

    @property
    def label(self) -> str:
        labels = {
            StartFrom.ABSOLUTE: _("Absolute Coordinates"),
            StartFrom.CURRENT_POSITION: _("Current Position"),
            StartFrom.USER_ORIGIN: _("User Origin"),
        }
        return labels[self]


class JobAnchor(Enum):
    """
    One of nine points on the job's bounding box.

    The value holds the fractions along X (left to right) and Y
    (bottom to top) in world space, which is Y-up.
    """

    TOP_LEFT = (0.0, 1.0)
    TOP = (0.5, 1.0)
    TOP_RIGHT = (1.0, 1.0)
    LEFT = (0.0, 0.5)
    CENTER = (0.5, 0.5)
    RIGHT = (1.0, 0.5)
    BOTTOM_LEFT = (0.0, 0.0)
    BOTTOM = (0.5, 0.0)
    BOTTOM_RIGHT = (1.0, 0.0)

    @property
    def label(self) -> str:
        labels = {
            JobAnchor.TOP_LEFT: _("Top left"),
            JobAnchor.TOP: _("Top center"),
            JobAnchor.TOP_RIGHT: _("Top right"),
            JobAnchor.LEFT: _("Middle left"),
            JobAnchor.CENTER: _("Center"),
            JobAnchor.RIGHT: _("Middle right"),
            JobAnchor.BOTTOM_LEFT: _("Bottom left"),
            JobAnchor.BOTTOM: _("Bottom center"),
            JobAnchor.BOTTOM_RIGHT: _("Bottom right"),
        }
        return labels[self]


@dataclass(frozen=True)
class JobOrigin:
    """The "Start From" mode and job anchor of a document."""

    start_from: StartFrom = StartFrom.ABSOLUTE
    anchor: JobAnchor = JobAnchor.BOTTOM_LEFT

    @property
    def is_absolute(self) -> bool:
        return self.start_from == StartFrom.ABSOLUTE

    def to_dict(self) -> dict[str, str]:
        return {
            "start_from": self.start_from.value,
            "anchor": self.anchor.name.lower(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "JobOrigin":
        if not data:
            return cls()
        try:
            start_from = StartFrom(data.get("start_from"))
        except ValueError:
            start_from = StartFrom.ABSOLUTE
        anchor_name = str(data.get("anchor", "")).upper()
        anchor = JobAnchor.__members__.get(anchor_name, JobAnchor.BOTTOM_LEFT)
        return cls(start_from=start_from, anchor=anchor)


def anchor_point(rect: BBox, anchor: JobAnchor) -> Point:
    """The world-space position of *anchor* on the bounding box."""
    min_x, min_y, max_x, max_y = rect
    fx, fy = anchor.value
    return (min_x + fx * (max_x - min_x), min_y + fy * (max_y - min_y))


def placement_offset(
    rect: BBox, anchor: JobAnchor, start_point: Point
) -> Point:
    """The translation that moves *anchor* of *rect* onto *start_point*."""
    ax, ay = anchor_point(rect, anchor)
    return (start_point[0] - ax, start_point[1] - ay)

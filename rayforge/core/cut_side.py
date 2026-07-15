from __future__ import annotations

from enum import Enum, auto
from gettext import gettext as _


class CutSide(Enum):
    """Defines which side of a path the cut should be on."""

    CENTERLINE = auto()
    """The center of the beam follows the path directly."""
    INSIDE = auto()
    """The final cut will be inside the original path."""
    OUTSIDE = auto()
    """The final cut will be outside the original path."""

    def label(self) -> str:
        """Return a translatable label for this cut side."""
        labels = {
            self.CENTERLINE: _("Centerline"),
            self.INSIDE: _("Inside"),
            self.OUTSIDE: _("Outside"),
        }
        return labels[self]


class CutOrder(Enum):
    """Defines the processing order for nested paths."""

    INSIDE_OUTSIDE = auto()
    OUTSIDE_INSIDE = auto()

    def label(self) -> str:
        """Return a translatable label for this cut order."""
        labels = {
            self.INSIDE_OUTSIDE: _("Inside-Outside"),
            self.OUTSIDE_INSIDE: _("Outside-Inside"),
        }
        return labels[self]

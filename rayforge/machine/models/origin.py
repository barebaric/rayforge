"""The physical corner a machine treats as coordinate zero."""

from enum import Enum
from gettext import gettext as _


class Origin(Enum):
    TOP_LEFT = "top_left"
    BOTTOM_LEFT = "bottom_left"
    TOP_RIGHT = "top_right"
    BOTTOM_RIGHT = "bottom_right"


#: Labels for the coordinate origins, in display order.
ORIGIN_LABELS = {
    Origin.BOTTOM_LEFT: _("Bottom Left"),
    Origin.TOP_LEFT: _("Top Left"),
    Origin.TOP_RIGHT: _("Top Right"),
    Origin.BOTTOM_RIGHT: _("Bottom Right"),
}

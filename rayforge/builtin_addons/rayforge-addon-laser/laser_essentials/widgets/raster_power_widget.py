"""Custom varset widget for the raster "Power" section.

Titles the min/max power rows after the lightest and darkest areas of
the image, and hides the section when no row is visible.
"""

from gettext import gettext as _
from typing import cast

from gi.repository import Adw

from rayforge.ui_gtk.varset.adapter import escape_title
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget


class RasterPowerWidget(VarSetWidget):
    """The raster dialog's "Power" section.

    Visibility itself is handled by the base machinery, which calls
    :meth:`_update_visibility` after populate, value changes and context
    pushes.
    """

    def _update_visibility(self):
        super()._update_visibility()
        self._update_power_labels()
        # Modes without power modulation or levels (constant power)
        # leave every row hidden; hide the empty section entirely.
        self.set_visible(
            any(row.get_visible() for row, _var in self.widget_map.values())
        )

    def _update_power_labels(self):
        """Title the min/max rows after the areas they drive."""
        self._set_row_text(
            "min_power_level",
            _("Min Power (White)"),
            _(
                "Power for lightest areas, as a percentage of the step's "
                "main power"
            ),
        )
        self._set_row_text(
            "max_power_level",
            _("Max Power (Black)"),
            _(
                "Power for darkest areas, as a percentage of the step's "
                "main power"
            ),
        )

    def _set_row_text(self, key: str, title: str, subtitle: str):
        row = self.row_for(key)
        if row is not None:
            row.set_title(escape_title(title))
            cast(Adw.ActionRow, row).set_subtitle(escape_title(subtitle))

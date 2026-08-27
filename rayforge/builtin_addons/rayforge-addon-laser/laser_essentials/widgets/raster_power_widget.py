"""Custom varset widget for the raster "Power" section.

Owns the invert-dependent min/max power labels: the min/max rows
describe the lightest/darkest areas, and which is which flips with the
``invert`` setting. The invert row lives in the Engrave section, so
its value arrives here through :meth:`set_context_values`; this widget
re-titles its rows whenever visibility is re-evaluated.
"""

from gettext import gettext as _
from typing import cast

from gi.repository import Adw

from rayforge.ui_gtk.varset.adapter import escape_title
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget


class RasterPowerWidget(VarSetWidget):
    """The raster dialog's "Power" section.

    Re-titles the min/max power rows based on the ``invert`` context
    value. Visibility itself is handled by the base machinery, which
    calls :meth:`_update_visibility` after populate, value changes and
    context pushes.
    """

    def _update_visibility(self):
        super()._update_visibility()
        self._update_power_labels(
            bool(self._context_values.get("invert", False))
        )
        # Modes without power modulation or levels (constant power)
        # leave every row hidden; hide the empty section entirely.
        self.set_visible(
            any(row.get_visible() for row, _var in self.widget_map.values())
        )

    def _update_power_labels(self, invert: bool):
        """Update min/max power labels based on invert setting."""
        lightest_subtitle = _(
            "Power for lightest areas, as a percentage of the step's "
            "main power"
        )
        darkest_subtitle = _(
            "Power for darkest areas, as a percentage of the step's main power"
        )

        if invert:
            min_title = _("Min Power (Black)")
            min_subtitle = darkest_subtitle
            max_title = _("Max Power (White)")
            max_subtitle = lightest_subtitle
        else:
            min_title = _("Min Power (White)")
            min_subtitle = lightest_subtitle
            max_title = _("Max Power (Black)")
            max_subtitle = darkest_subtitle

        min_row = self.row_for("min_power_level")
        if min_row is not None:
            min_row.set_title(escape_title(min_title))
            cast(Adw.ActionRow, min_row).set_subtitle(
                escape_title(min_subtitle)
            )
        max_row = self.row_for("max_power_level")
        if max_row is not None:
            max_row.set_title(escape_title(max_title))
            cast(Adw.ActionRow, max_row).set_subtitle(
                escape_title(max_subtitle)
            )

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import GLib

from rayforge.core.varset import VarSet
from rayforge.machine.models.laser import LaserHead

from .laser_step_page import LaserStepSettingsPage
from .levels_adapter import LevelsAdapter
from .raster_power_widget import RasterPowerWidget

#: Engrave-section keys (mode, geometry, multi-pass) vs. Power-section
#: keys (histogram, brightness range). The recipe varset keeps them in
#: one group; the dialog splits them into two sections.
_ENGRAVE_KEYS = {
    "depth_mode",
    "threshold",
    "dither_algorithm",
    "scan_angle",
    "cross_hatch",
    "scan_mode",
    "line_interval_mm",
    "sample_interval_mm",
    "dot_width_correction_mm",
    "bidir_x_offset_mm",
    "invert",
    "num_depth_levels",
    "z_step_down",
    "angle_increment",
}

_POWER_KEYS = {
    "auto_levels",
    "black_point",
    "white_point",
    "min_power_level",
    "max_power_level",
    "num_power_levels",
}

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class RasterSettingsPage(LaserStepSettingsPage):
    """UI for configuring the EngraveStep."""

    cut_speed_title = _("Engrave Speed")

    def __init__(
        self,
        editor: "DocEditor",
        step: Any,
    ):
        super().__init__(editor, step)
        groups = self.step.recipe_varset_groups()
        step_vars = groups[-1][1] if len(groups) > 1 else None
        if step_vars is None:
            return
        engrave_vs = VarSet(
            vars=[v for v in step_vars if v.key in _ENGRAVE_KEYS]
        )
        power_vs = VarSet(vars=[v for v in step_vars if v.key in _POWER_KEYS])
        self.engrave_widget = self.add_varset_section(
            _("Engrave"),
            engrave_vs,
            description=_("Raster the image onto the material."),
        )
        self.power_widget = self.add_varset_section(
            _("Power"),
            power_vs,
            description=_("Power modulation and brightness range."),
            widget_cls=RasterPowerWidget,
        )

        # The Power section's visible_when predicates and histogram
        # depend on depth_mode and invert, whose rows live in the
        # Engrave section. Feed them in as context; the adapters fire
        # changed synchronously on user interaction and on resync.
        self._sync_power_context()
        for key in ("depth_mode", "invert"):
            adapter = self.engrave_widget.adapter_for(key)
            if adapter is not None:
                adapter.changed.connect(
                    lambda sender, k=key: self._sync_power_context(),
                    weak=False,
                )

        levels = self.power_widget.adapter_for("black_point")
        if isinstance(levels, LevelsAdapter):
            levels.set_histogram_source(step)
            GLib.idle_add(levels.compute_histogram)

        self._push_head_defaults()

    def _push_head_defaults(self):
        """Show head-derived values for the auto (None) interval rows.

        The step keeps ``None`` (auto) until the user edits the row;
        the old dialog displayed the laser spot-size default instead.
        """
        head = self.get_selected_head()
        if not isinstance(head, LaserHead):
            return
        spot_x, spot_y = head.spot_size_mm
        values = {}
        if self.step.line_interval_mm is None:
            values["line_interval_mm"] = spot_y
        if self.step.sample_interval_mm is None:
            values["sample_interval_mm"] = spot_x / 2.0
        if self.step.dot_width_correction_mm is None:
            values["dot_width_correction_mm"] = spot_x / 2.0
        if values:
            self.engrave_widget.set_values(values)

    def _sync_power_context(self):
        """Feed the current depth_mode/invert into the Power section."""
        values = self.engrave_widget.get_values()
        self.power_widget.set_context_values(
            {
                "depth_mode": values.get("depth_mode"),
                "invert": values.get("invert"),
            }
        )

    def _on_varset_data_changed(self, widget, key):
        if key in ("min_power_level", "max_power_level"):
            self._commit_power_range(widget, key)
            return
        super()._on_varset_data_changed(widget, key)

    def _commit_power_range(self, widget, moved_key):
        """Commit min/max power together, keeping min <= max.

        When one slider is dragged past the other, the other follows
        (max follows min up, min follows max down), mirroring the old
        dialog behavior.
        """
        min_p = widget.adapter_for("min_power_level").get_value()
        max_p = widget.adapter_for("max_power_level").get_value()
        if min_p is None or max_p is None:
            return
        if moved_key == "min_power_level" and min_p > max_p:
            max_p = min_p
        elif moved_key == "max_power_level" and max_p < min_p:
            min_p = max_p
        with self.history_manager.transaction(_("Change Power Range")):
            self.set_step_property("min_power_level", min_p)
            self.set_step_property("max_power_level", max_p)

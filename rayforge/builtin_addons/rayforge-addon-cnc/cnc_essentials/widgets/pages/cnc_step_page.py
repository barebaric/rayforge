"""CNC step settings pages.

The "CNC" companion page (spindle, cooling, depth, feed) and the CNC
step page base. Both render their rows from the step's recipe varsets
via the varset machinery.
"""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from rayforge.core.undo import ChangePropertyCommand
from rayforge.core.varset import VarSet
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep

_SPINDLE_KEYS = ["selected_head_uid", "spindle_rpm", "tool_diameter"]
_COOLANT_KEY = "coolant_method"
_DEPTH_KEYS = ["target_depth", "depth_per_pass", "safe_z"]
_FEED_KEYS = ["cut_speed", "travel_speed", "plunge_speed"]


class CncSettingsPage(StepSettingsPage):
    """The CNC process settings page (spindle, cooling, depth, feed)."""

    show_identity = False

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(editor, step)
        producer_type = step.ASSEMBLER_NAME or "unknown"
        self.key = f"{producer_type.lower()}/cnc"

        cnc_group = self._cnc_group()
        if cnc_group is None:
            return
        spindle_keys = _SPINDLE_KEYS + [_COOLANT_KEY]
        self.spindle_widget = self.add_varset_section(
            _("Spindle"),
            self._varset_for_keys(cnc_group, spindle_keys),
            description=_("Spindle head, speed, tool geometry, and cooling."),
        )
        self.add_varset_section(
            _("Depth"),
            self._varset_for_keys(cnc_group, _DEPTH_KEYS),
            description=_("Cut depth, depth per pass, and safe height."),
        )
        self.add_varset_section(
            _("Feed"),
            self._varset_for_keys(cnc_group, _FEED_KEYS),
            description=_("Cutting, plunging, and travel feed rates."),
        )

        # The head row needs a machine to list heads.
        head_row = self.spindle_widget.row_for("selected_head_uid")
        if head_row is not None:
            head_row.set_visible(self.get_machine() is not None)
        self._update_machine_bounds()

    def _cnc_group(self) -> VarSet | None:
        """The domain varset group holding the common CNC settings."""
        groups = self.step.recipe_varset_groups()
        return groups[0][1] if groups else None

    def _on_machine_changed(self):
        """Update head row visibility after a machine switch."""
        super()._on_machine_changed()
        head_row = self.spindle_widget.row_for("selected_head_uid")
        if head_row is not None:
            head_row.set_visible(self.get_machine() is not None)

    def _on_varset_data_changed(self, widget, key):
        if key == "selected_head_uid":
            self._on_head_changed(widget.get_values().get("selected_head_uid"))
            return
        super()._on_varset_data_changed(widget, key)

    def _on_head_changed(self, head_uid):
        step = self.step
        if head_uid == step.selected_head_uid:
            return
        with self.history_manager.transaction(_("Change Spindle")) as t:
            t.execute(
                ChangePropertyCommand(
                    target=step,
                    property_name="selected_head_uid",
                    new_value=head_uid,
                    setter_method_name="set_selected_head_uid",
                )
            )


class CncStepSettingsPage(StepSettingsPage):
    """Base page for CNC step settings.

    Shows the step's own settings; the common CNC process settings
    (spindle, cooling, depth, feed) live on a second
    ``CncSettingsPage`` opened from the settings dialog. Subclasses
    override ``_add_step_sections``.
    """

    extra_pages = (("cnc_page", _("CNC"), "tool-change-symbolic"),)

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(editor, step)
        self._add_step_sections()

    def _add_step_sections(self):
        """Add step-specific sections right after the General section."""

    def _step_specific_group(self) -> VarSet | None:
        """The concrete step's own settings group, if any."""
        groups = self.step.recipe_varset_groups()
        return groups[-1][1] if len(groups) > 1 else None

    def cnc_page(self) -> CncSettingsPage:
        """Build the companion CNC process settings page."""
        return CncSettingsPage(self.editor, self.step)

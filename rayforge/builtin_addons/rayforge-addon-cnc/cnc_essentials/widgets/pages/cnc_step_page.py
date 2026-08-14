"""CNC step settings widget base."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.core.varset import VarSet
from rayforge.machine.models.spindle import SpindleHead
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep

_SPINDLE_KEYS = {"spindle_rpm", "tool_diameter"}
_DEPTH_KEYS = {"target_depth", "depth_per_pass", "safe_z"}
_FEED_KEYS = {"cut_speed", "travel_speed", "plunge_speed"}


class CncStepSettingsPage(StepSettingsPage):
    """Base page for CNC step settings.

    Renders the common CNC sections (cooling, spindle, depth, feed)
    from the step's recipe varset. Subclasses add their step-specific
    sections.
    """

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(editor, step)
        self._add_cooling_section()
        cnc_group = self._cnc_group()
        if cnc_group is None:
            return
        self.add_varset_section(
            _("Spindle"),
            self._varset_for_keys(cnc_group, _SPINDLE_KEYS),
            description=_("Spindle speed and tool geometry."),
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

    def _cnc_group(self) -> VarSet | None:
        """The domain varset group holding the common CNC settings."""
        groups = self.step.recipe_varset_groups()
        return groups[0][1] if groups else None

    def _step_specific_group(self) -> VarSet | None:
        """The concrete step's own settings group, if any."""
        groups = self.step.recipe_varset_groups()
        return groups[-1][1] if len(groups) > 1 else None

    def _add_cooling_section(self):
        """Add the coolant section, hidden unless a spindle head is used."""
        cnc_group = self._cnc_group()
        if cnc_group is None:
            return
        coolant_var = next(
            (var for var in cnc_group if var.key == "coolant_method"), None
        )
        if coolant_var is None:
            return
        self.coolant_section = self.add_varset_section(
            _("Cooling"),
            VarSet(vars=[coolant_var]),
            description=_("Coolant used while this operation runs."),
        )
        self.step.updated.connect(self._update_cooling_section_visibility)
        self._update_cooling_section_visibility()

    def _update_cooling_section_visibility(self, *args):
        self.coolant_section.set_visible(
            isinstance(self.get_selected_head(), SpindleHead)
        )

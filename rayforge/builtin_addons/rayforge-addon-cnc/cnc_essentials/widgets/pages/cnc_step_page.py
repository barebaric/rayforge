"""CNC step settings widget base."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.core.varset import VarSet
from rayforge.machine.models.spindle import SpindleHead
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.doceditor.step_settings.rows import (
    CutSpeedRow,
    TravelSpeedRow,
)

from ..rows.depth_per_pass_row import DepthPerPassRow
from ..rows.plunge_speed_row import PlungeSpeedRow
from ..rows.safe_z_row import SafeZRow
from ..rows.spindle_rpm_row import SpindleRpmRow
from ..rows.target_depth_row import TargetDepthRow
from ..rows.tool_diameter_row import ToolDiameterRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class CncStepSettingsPage(StepSettingsPage):
    """Base page for CNC step settings.

    Adds the common CNC sections (cooling, spindle, depth, feed).
    Subclasses add their step-specific sections.
    """

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(editor, step)
        self._add_cooling_section()
        self.add_section(
            _("Spindle"),
            SpindleRpmRow,
            ToolDiameterRow,
            description=_("Spindle speed and tool geometry."),
        )
        self.add_section(
            _("Depth"),
            TargetDepthRow,
            DepthPerPassRow,
            SafeZRow,
            description=_("Cut depth, depth per pass, and safe height."),
        )
        self.add_section(
            _("Feed"),
            CutSpeedRow(editor, step, title=_("Feed Rate")),
            TravelSpeedRow,
            PlungeSpeedRow,
            description=_("Cutting, plunging, and travel feed rates."),
        )

    def _add_cooling_section(self):
        """Add the coolant section, hidden unless a spindle head is used."""
        coolant_var = next(
            (
                var
                for var in self.step.recipe_varset()
                if var.key == "coolant_method"
            ),
            None,
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

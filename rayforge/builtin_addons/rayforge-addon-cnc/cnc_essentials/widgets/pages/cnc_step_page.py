"""CNC step settings widget base."""

from gettext import gettext as _
from typing import TYPE_CHECKING

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

    Adds the common CNC sections (spindle, depth, feed). Subclasses
    add their step-specific sections.
    """

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(editor, step)
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

"""Laser step settings pages."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from rayforge.core.undo import ChangePropertyCommand
from rayforge.machine.models.laser import LaserHead
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.doceditor.step_settings.rows import (
    CutSpeedRow,
    HeadRow,
    TravelSpeedRow,
)

from ..rows.air_assist_row import AirAssistRow
from ..rows.power_row import PowerRow
from ..rows.pwm_row import FrequencyRow, PulseWidthRow
from ..rows.tab_power_row import TabPowerRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class LaserSettingsPage(StepSettingsPage):
    """The laser process settings page (head, power, speed, PWM)."""

    show_identity = False

    def __init__(
        self,
        editor: "DocEditor",
        step: Any,
        include_tab_power: bool = False,
        include_process: bool = True,
        cut_speed_title: str = _("Cut Speed"),
    ):
        super().__init__(editor, step)
        producer_type = step.ASSEMBLER_NAME or "unknown"
        self.key = f"{producer_type.lower()}/laser"
        self.head_row = HeadRow(editor, step)
        self.head_row.head_changed.connect(self._on_head_changed)
        if include_process:
            rows = [
                self.head_row,
                PowerRow,
                CutSpeedRow(editor, step, title=cut_speed_title),
                TravelSpeedRow,
                AirAssistRow,
            ]
            if include_tab_power:
                rows.append(TabPowerRow)
        else:
            rows = [self.head_row, AirAssistRow]
        self.add_section(
            _("Laser"),
            *rows,
            description=_(
                "Laser power, speed, and head selection for this operation."
            ),
        )
        self.machine_section = self.add_section(
            _("Machine"),
            FrequencyRow,
            PulseWidthRow,
            description=_(
                "Settings provided by the machine's hardware for this head."
            ),
        )
        step.updated.connect(self._update_machine_section_visibility)
        self._update_machine_section_visibility()

    def _update_machine_section_visibility(self, *args):
        machine = self.get_machine()
        head = self.get_selected_head()
        supported = bool(machine and head and machine.get_pwm_params(head))
        self.machine_section.set_visible(supported)

    def _on_head_changed(self, sender, head_uid):
        step = self.step
        if head_uid == step.selected_head_uid:
            return
        machine = self.get_machine()
        head = None
        if machine:
            head = next((h for h in machine.heads if h.uid == head_uid), None)
        with self.history_manager.transaction(_("Change Head")) as t:
            t.execute(
                ChangePropertyCommand(
                    target=step,
                    property_name="selected_head_uid",
                    new_value=head_uid,
                    setter_method_name="set_selected_head_uid",
                )
            )
            if isinstance(head, LaserHead):
                params = machine.get_pwm_params(head) if machine else None
                if params is not None:
                    t.execute(
                        ChangePropertyCommand(
                            target=step,
                            property_name="frequency",
                            new_value=params.frequency,
                            setter_method_name="set_frequency",
                        )
                    )
                    t.execute(
                        ChangePropertyCommand(
                            target=step,
                            property_name="pulse_width",
                            new_value=params.pulse_width,
                            setter_method_name="set_pulse_width",
                        )
                    )


class LaserStepSettingsPage(StepSettingsPage):
    """Base page for laser step settings.

    Shows the step's own settings; the laser process settings live on
    a second ``LaserSettingsPage`` opened from the settings dialog.
    Subclasses override ``_add_step_sections`` and the laser options
    class attributes.
    """

    include_tab_power = False
    include_process = True
    cut_speed_title = _("Cut Speed")

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(editor, step)
        self._add_step_sections()

    def _add_step_sections(self):
        """Add step-specific sections right after the General section."""

    def laser_page(self) -> LaserSettingsPage:
        """Build the companion laser process settings page."""
        return LaserSettingsPage(
            self.editor,
            self.step,
            include_tab_power=self.include_tab_power,
            include_process=self.include_process,
            cut_speed_title=self.cut_speed_title,
        )

"""Laser step settings pages.

The "Laser" companion page and the laser step page base. Both render
their rows from the step's recipe varsets via the varset machinery.
"""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from rayforge.core.undo import ChangePropertyCommand
from rayforge.core.varset import VarSet
from rayforge.machine.models.laser import LaserHead
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.varset.adapter import escape_title

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

_MACHINE_KEYS = ("frequency", "pulse_width")


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

        full = step.recipe_varset()
        laser_keys = {
            "selected_head_uid",
            "power",
            "cut_speed",
            "travel_speed",
            "air_assist",
        }
        if include_tab_power:
            laser_keys.add("tab_power")
        if not include_process:
            laser_keys = {"selected_head_uid", "air_assist"}
        laser_vs = VarSet(
            vars=[v for v in full if v.key in laser_keys],
            description=_(
                "Laser power, speed, and head selection for this operation."
            ),
        )
        machine_vs = VarSet(
            vars=[v for v in full if v.key in _MACHINE_KEYS],
            description=_(
                "Settings provided by the machine's hardware for this head."
            ),
        )
        self.laser_widget = self.add_varset_section(_("Laser"), laser_vs)
        self.machine_widget = self.add_varset_section(_("Machine"), machine_vs)
        if cut_speed_title != _("Cut Speed"):
            row = self.laser_widget.row_for("cut_speed")
            if row is not None:
                row.set_title(escape_title(cut_speed_title))

        # The head row needs a machine to list heads.
        head_row = self.laser_widget.row_for("selected_head_uid")
        if head_row is not None:
            head_row.set_visible(self.get_machine() is not None)

        self.step.updated.connect(self._update_machine_section_visibility)
        self._update_machine_section_visibility()

    def _update_machine_section_visibility(self, *args):
        machine = self.get_machine()
        head = self.get_selected_head()
        supported = bool(machine and head and machine.get_pwm_params(head))
        self.machine_widget.set_visible(supported)

    def _on_machine_changed(self):
        """Update head row and PWM section after a machine switch."""
        head_row = self.laser_widget.row_for("selected_head_uid")
        if head_row is not None:
            head_row.set_visible(self.get_machine() is not None)
        self._update_machine_section_visibility()

    def _on_varset_data_changed(self, widget, key):
        if key == "selected_head_uid":
            self._on_head_changed(widget.get_values().get("selected_head_uid"))
            return
        super()._on_varset_data_changed(widget, key)

    def _on_head_changed(self, head_uid):
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

    extra_pages = (("laser_page", _("Laser"), "laser-on-symbolic"),)

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

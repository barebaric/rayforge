"""Step 9 — Rotary module (optional).

Pick rotary type (jaws / rollers), axis (A / B / C), mode (true 4th
axis vs axis replacement), geometry (mm-per-rotation, default
diameter, max length, roller Ø), mount position, reverse-direction
flag and an optional 3D model path.

The wizard seeds one ``RotaryModule``. Skipping the page (via the
wizard's "Skip" button) means no entry is written to
``MachineConfig.rotary_modules``. The page is intentionally written
against ``DeviceProfile`` rather than a live ``Machine`` so it can
plug into the wizard's in-memory model.

Multiples: per the design "A machine may define multiple rotary
modules; the wizard seeds one and offers an `add another` affordance."
We keep things simple here — multiple modules can be added through
the machine settings page (``RotaryModulePage``) once the machine is
created.
"""

from gettext import gettext as _

from gi.repository import Adw, Gtk
from raygeo.ops.axis import Axis

from ....machine.device.profile import DeviceProfile
from ....machine.models.rotary_module import (
    RotaryMode,
    RotaryModule,
    RotaryType,
)
from . import WizardPage, _makePreferencesGroup

_AXIS_NAMES = ("A", "B", "C")
_AXIS_NAMED = {"A": Axis.A, "B": Axis.B, "C": Axis.C}
_MODE_TRUE_4TH = 0
_TYPE_JAWS = 0
_TYPE_ROLLERS = 1


class RotaryPage(WizardPage):
    step_number = 9
    title = _("Rotary Module")
    subtitle = _(
        "Optional. Set up a rotary attachment now or skip this "
        "step to add one later from machine settings."
    )

    def __init__(self, wizard, **kwargs):
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        self.details_group = _makePreferencesGroup(
            title=_("Module"),
            description=_("Pick rotary type, axis, mode, and geometry."),
        )
        self.content.append(self.details_group)

        type_store = Gtk.StringList()
        type_store.append(_("Jaws / chuck"))
        type_store.append(_("Rollers"))
        self.type_row = Adw.ComboRow(
            title=_("Rotary Type"),
            subtitle=_("How the workpiece is held"),
            model=type_store,
        )
        self.type_row.connect("notify::selected", self._on_type_changed)
        self.details_group.add(self.type_row)

        axis_store = Gtk.StringList()
        for name in _AXIS_NAMES:
            axis_store.append(name)
        self.axis_row = Adw.ComboRow(
            title=_("Rotary Axis"),
            subtitle=_("Which axis the rotary uses"),
            model=axis_store,
        )
        self.details_group.add(self.axis_row)

        mode_store = Gtk.StringList()
        mode_store.append(_("True 4th Axis (keeps X/Y/Z)"))
        mode_store.append(_("Axis Replacement (swaps e.g. Y for A)"))
        self.mode_row = Adw.ComboRow(title=_("Mode"), model=mode_store)
        self.mode_row.connect("notify::selected", self._on_mode_changed)
        self.details_group.add(self.mode_row)

        # Geometry fields
        self.mu_per_rotation_row = Adw.SpinRow(
            title=_("Length per Rotation"),
            subtitle=_("Auto-fetched from GRBL $101/$103 if probing"),
            adjustment=Gtk.Adjustment(
                lower=0, upper=100000, step_increment=0.1, page_increment=10
            ),
            digits=3,
        )
        self.mu_per_rotation_row.set_value(0)
        self.details_group.add(self.mu_per_rotation_row)

        self.default_diameter_row = Adw.SpinRow(
            title=_("Default Workpiece Ø"),
            adjustment=Gtk.Adjustment(
                lower=0, upper=1000, step_increment=1, page_increment=10
            ),
            digits=2,
        )
        self.default_diameter_row.set_value(25.0)
        self.details_group.add(self.default_diameter_row)

        self.max_length_row = Adw.SpinRow(
            title=_("Max Workpiece Length"),
            adjustment=Gtk.Adjustment(
                lower=0, upper=10000, step_increment=1, page_increment=10
            ),
            digits=2,
        )
        self.max_length_row.set_value(300.0)
        self.details_group.add(self.max_length_row)

        self.roller_diameter_row = Adw.SpinRow(
            title=_("Roller Ø"),
            subtitle=_("Required when using roller-type rotary"),
            adjustment=Gtk.Adjustment(
                lower=0, upper=1000, step_increment=1, page_increment=10
            ),
            digits=2,
        )
        self.roller_diameter_row.set_value(0.0)
        self.details_group.add(self.roller_diameter_row)

        self.reverse_row = Adw.SwitchRow(
            title=_("Reverse Axis Direction"),
            subtitle=_("Invert the rotary's rotation direction"),
        )
        self.details_group.add(self.reverse_row)

        # The page is always ready; "Skip" on the footer is how the
        # user opts out entirely.
        self.set_ready(True)
        self._on_type_changed(self.type_row, None)
        self._on_mode_changed(self.mode_row, None)

    def _on_type_changed(self, row, _param) -> None:
        # Roller Ø is only meaningful for roller-type rotational.
        is_rollers = row.get_selected() == _TYPE_ROLLERS
        self.roller_diameter_row.set_visible(is_rollers)

    def _on_mode_changed(self, row, _param) -> None:
        # In axis-replacement mode, mm-per-rotation is unused — the
        # axis inherits the replaced linear axis's settings.
        is_axis_replacement = row.get_selected() != _MODE_TRUE_4TH
        self.mu_per_rotation_row.set_visible(not is_axis_replacement)

    # ----- profile binding -----------------------------------------------

    def enter(self, profile: DeviceProfile) -> None:
        mc = profile.machine_config
        modules = mc.rotary_modules or []
        if not modules:
            return
        first = modules[0]
        self.type_row.set_selected(
            _TYPE_JAWS
            if (first.get("rotary_type", "jaws") == "jaws")
            else _TYPE_ROLLERS
        )
        axis_name = first.get("axis", "A")
        try:
            self.axis_row.set_selected(_AXIS_NAMES.index(axis_name))
        except ValueError:
            self.axis_row.set_selected(0)
        mode_val = first.get("mode", "true_4th_axis")
        self.mode_row.set_selected(
            0 if mode_val == RotaryMode.TRUE_4TH_AXIS.value else 1
        )
        self.mu_per_rotation_row.set_value(first.get("mm_per_rotation", 0))
        self.default_diameter_row.set_value(
            first.get("default_diameter", 25.0)
        )
        self.max_length_row.set_value(first.get("max_workpiece_length", 300.0))
        self.roller_diameter_row.set_value(first.get("roller_diameter", 0.0))
        self.reverse_row.set_active(bool(first.get("reverse_axis", False)))

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        mc = profile.machine_config

        axis_idx = self.axis_row.get_selected()
        axis_name = _AXIS_NAMES[axis_idx] if axis_idx >= 0 else "A"
        mode_val = (
            RotaryMode.TRUE_4TH_AXIS.value
            if self.mode_row.get_selected() == _MODE_TRUE_4TH
            else RotaryMode.AXIS_REPLACEMENT.value
        )
        type_val = (
            RotaryType.JAWS.value
            if self.type_row.get_selected() == _TYPE_JAWS
            else RotaryType.ROLLERS.value
        )

        # Build a fresh RotaryModule so the model canonicalizes the
        # data and fills defaults we don't surface in the wizard.
        module = RotaryModule()
        module.name = _("Rotary Module")
        module.axis = _AXIS_NAMED.get(axis_name, Axis.A)
        module.mode = RotaryMode(mode_val)
        module.rotary_type = RotaryType(type_val)
        module.mu_per_rotation = self.mu_per_rotation_row.get_value()
        module.default_diameter = self.default_diameter_row.get_value()
        module.max_workpiece_length = self.max_length_row.get_value()
        module.roller_diameter = self.roller_diameter_row.get_value()
        module.reverse_axis = self.reverse_row.get_active()

        # Stash uid so when the orchestrator materializes the machine
        # it can avoid duplicate-pathology: we hand back the dict form
        # for MachineConfig.rotary_modules.
        mc.rotary_modules = [module.to_dict()]
        return True


__all__ = ["RotaryPage"]

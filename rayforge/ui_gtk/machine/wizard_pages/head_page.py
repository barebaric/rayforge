"""Step 8 — Head configuration.

Lets the user declare whether the machine has a laser or a spindle head
and capture the key head-specific params (max power, spot size, PWM
freq, focal distance, framing, max/min RPM for spindle, etc.). Mirrors
the most important widgets from the live
:mod:`~rayforge.ui_gtk.machine.head_preferences_page` page but binds to
the wizard's working ``DeviceProfile``.

The wizard seeds a single head; if the user has multiple heads they'll
add more via machine settings later.
"""

from gettext import gettext as _
from typing import Any, Optional

from gi.repository import Adw, Gtk

from ....machine.device.profile import DeviceProfile
from ....machine.driver import get_driver_cls
from ....machine.driver.dummy import NoDeviceDriver
from ....machine.models.laser import LaserHead
from ...shared.pref_rows.base import SpinRow
from ...shared.pref_rows.length_spin_row import LengthSpinRow
from . import WizardPage, _makePreferencesGroup

# Index 0 == laser, 1 == spindle.
_HEAD_LASER = 0
_HEAD_SPINDLE = 1


def _is_spindle_head_dict(head: Optional[dict[str, Any]]) -> bool:
    if not head:
        return False
    cls = (head.get("head_class") or "").lower()
    return "spindle" in cls or "max_rpm" in head


class HeadPage(WizardPage):
    step_number = 8
    title = _("Head")
    subtitle = _("What's attached to the gantry: a laser, a spindle, or both?")

    def __init__(self, wizard, **kwargs):
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        head_group = _makePreferencesGroup(
            title=_("Head Type"),
            description=_("Pick the primary head for this machine."),
        )
        self.content.append(head_group)

        store = Gtk.StringList()
        store.append(_("Laser Head"))
        store.append(_("Spindle Head"))
        self.head_type_row = Adw.ComboRow(
            title=_("Head Type"),
            subtitle=_("Type of tool attached to this machine"),
            model=store,
        )
        self.head_type_row.connect(
            "notify::selected", self._on_head_type_changed
        )
        head_group.add(self.head_type_row)

        self.head_name_row = Adw.EntryRow(title=_("Head Name"))
        head_group.add(self.head_name_row)

        # ----- shared / laser fields -------------------------------
        self.laser_group = _makePreferencesGroup(title=_("Laser Settings"))
        self.content.append(self.laser_group)

        self.max_power_row = SpinRow(
            _("Max Power (S-value)"),
            _("Max laser power value in GCode"),
            lower=1,
            upper=100000,
            step_increment=100,
            value=1000,
        )
        self.laser_group.add(self.max_power_row)

        self.spot_x_row = LengthSpinRow(
            _("Spot Size X"),
            _("Laser beam width on X axis"),
            upper=10,
            digits=3,
            value_in_base=0.1,
        )
        self.laser_group.add(self.spot_x_row)

        self.spot_y_row = LengthSpinRow(
            _("Spot Size Y"),
            _("Laser beam width on Y axis"),
            upper=10,
            digits=3,
            value_in_base=0.1,
        )
        self.laser_group.add(self.spot_y_row)

        self.pwm_freq_row = SpinRow(
            _("PWM Frequency (Hz)"),
            _("Laser modulation frequency"),
            lower=1,
            upper=100000,
            step_increment=100,
            value=500,
        )
        self.laser_group.add(self.pwm_freq_row)

        self.focal_distance_row = LengthSpinRow(
            _("Focal Distance"),
            _("Lens-to-workpiece distance"),
            upper=1000,
            value_in_base=0,
        )
        self.laser_group.add(self.focal_distance_row)

        # ----- spindle fields ----------------------------------
        self.spindle_group = _makePreferencesGroup(title=_("Spindle"))
        self.content.append(self.spindle_group)

        self.max_rpm_row = SpinRow(
            _("Max RPM"),
            lower=1,
            upper=100000,
            step_increment=100,
            value=20000,
        )
        self.spindle_group.add(self.max_rpm_row)

        self.min_rpm_row = SpinRow(
            _("Min RPM"),
            lower=1,
            upper=100000,
            step_increment=100,
            value=1000,
        )
        self.spindle_group.add(self.min_rpm_row)

        self._on_head_type_changed(self.head_type_row, None)
        self.set_ready(True)

    def _on_head_type_changed(self, row, _param) -> None:
        is_spindle = row.get_selected() == _HEAD_SPINDLE
        self.laser_group.set_visible(not is_spindle)
        self.spindle_group.set_visible(is_spindle)
        if not is_spindle:
            self._update_pwm_visibility()

    def _update_pwm_visibility(self) -> None:
        """Hide PWM fields when the chosen driver doesn't support PWM."""
        driver_name = self.wizard.profile.machine_config.driver
        if not driver_name:
            self.pwm_freq_row.set_visible(False)
            return
        driver_cls = get_driver_cls(driver_name)
        if driver_cls is NoDeviceDriver:
            self.pwm_freq_row.set_visible(False)
            return
        # ``supports_pwm`` is an instance method but the overrides don't
        # use ``self``; build a bare instance to query without a live
        # machine.
        probe_driver = driver_cls.__new__(driver_cls)
        probe_head = LaserHead()
        self.pwm_freq_row.set_visible(probe_driver.supports_pwm(probe_head))

    # ----- profile binding --------------------------------------------

    def enter(self, profile: DeviceProfile) -> None:
        heads = profile.machine_config.heads or []
        head: dict[str, Any] = heads[0] if heads else {}
        if _is_spindle_head_dict(head):
            self.head_type_row.set_selected(_HEAD_SPINDLE)
            self.max_rpm_row.set_value(head.get("max_rpm", 20000))
            self.min_rpm_row.set_value(head.get("min_rpm", 1000))
        else:
            self.head_type_row.set_selected(_HEAD_LASER)
            self.max_power_row.set_value(head.get("max_power", 1000))
            spot = head.get("spot_size_mm") or (0.1, 0.1)
            self.spot_x_row.set_value_in_base_units(spot[0])
            self.spot_y_row.set_value_in_base_units(spot[1])
            self.pwm_freq_row.set_value(head.get("pwm_frequency", 500))
            self.focal_distance_row.set_value_in_base_units(
                head.get("focal_distance", 0)
            )
            self._update_pwm_visibility()
        self.head_name_row.set_text(head.get("name", ""))

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        head: dict[str, Any] = {"name": self.head_name_row.get_text() or ""}
        if self.head_type_row.get_selected() == _HEAD_SPINDLE:
            head["head_class"] = "SpindleHead"
            head["max_rpm"] = int(self.max_rpm_row.get_value())
            head["min_rpm"] = int(self.min_rpm_row.get_value())
        else:
            head["head_class"] = "LaserHead"
            head["max_power"] = int(self.max_power_row.get_value())
            head["spot_size_mm"] = [
                self.spot_x_row.get_value_in_base_units(),
                self.spot_y_row.get_value_in_base_units(),
            ]
            if self.pwm_freq_row.get_visible():
                head["pwm_frequency"] = int(self.pwm_freq_row.get_value())
            head["focal_distance"] = (
                self.focal_distance_row.get_value_in_base_units()
            )
        profile.machine_config.heads = [head]
        return True


__all__ = ["HeadPage"]

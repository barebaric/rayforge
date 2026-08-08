"""Step 3 — Connection parameters.

Always required: even after picking a known profile or importing a
snapshot, the user must supply host-specific values like the USB
device path, IP address, hostname, or OctoPrint API key. The page
prefills any defaults the profile supplies and leaves the rest blank.

Reuses :class:`~rayforge.ui_gtk.varset.varsetwidget.VarSetWidget` to
render the driver's ``get_setup_vars()`` definition, mirroring the
pattern in the legacy ``config_wizard.py``.
"""

from gettext import gettext as _

from gi.repository import Adw, Gtk

from ....machine.device.profile import DeviceProfile
from ....machine.driver import get_driver_cls
from ....machine.driver.driver import Driver
from ...varset.varsetwidget import VarSetWidget
from . import WizardPage, _makePreferencesGroup


class ConnectionPage(WizardPage):
    step_number = 3
    title = _("Connection")
    subtitle = _("Enter the connection parameters for your device.")

    def __init__(self, wizard, **kwargs):
        self._driver_cls: type[Driver] | None = None
        self._required_keys: set = set()
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        self.group = _makePreferencesGroup(
            title=_("Connection"),
            description=_(
                "Enter the connection parameters your machine "
                "requires. The exact fields depend on the controller "
                "you chose in the previous step."
            ),
        )
        self.content.append(self.group)

        self.driver_row = Adw.ActionRow(
            title=_("Driver"),
            subtitle=_("Fixed by the chosen profile"),
        )
        self.group.add(self.driver_row)

        self.connect_widget = VarSetWidget()
        self.connect_widget.data_changed.connect(self._on_data_changed)
        self.content.append(self.connect_widget)

        # Spacer to keep the surrounding layout from looking cramped.
        self.content.append(Gtk.Box(vexpand=True))

    def enter(self, profile: DeviceProfile) -> None:
        """Repopulate the form from the working profile."""
        driver_name = profile.machine_config.driver
        if not driver_name:
            self.driver_row.set_title(_("Driver"))
            self.driver_row.set_subtitle(_("None — G-code export only"))
            self.connect_widget.clear_dynamic_rows()
            self.set_ready(True)
            self._driver_cls = None
            return

        driver_cls = get_driver_cls(driver_name)
        self._driver_cls = driver_cls
        self.driver_row.set_title(driver_cls.label)
        self.driver_row.set_subtitle(driver_cls.subtitle or "")

        var_set = driver_cls.get_setup_vars()
        # Vars without a usable default are the host-specific values the
        # user must supply (USB path, hostname, API key, …). The page
        # stays unready until every one of them is filled.
        self._required_keys = {
            var.key for var in var_set if var.default in (None, "")
        }
        # If the working profile carries saved driver_args (e.g. via
        # import), prefill the var set before rendering.
        saved_args = profile.machine_config.driver_args or {}
        if saved_args:
            for var in var_set:
                if saved_args.get(var.key):
                    var.value = saved_args[var.key]
        self.connect_widget.populate(var_set)
        self._refresh_ready()

    def _on_data_changed(self, sender, **kwargs) -> None:
        self._refresh_ready()

    def _refresh_ready(self) -> None:
        """Ready when there's no driver or all required vars are set."""
        if self._driver_cls is None:
            self.set_ready(True)
            return
        try:
            values = self.connect_widget.get_values()
        except ValueError:
            self.set_ready(False)
            return
        for key in self._required_keys:
            if values.get(key) in (None, ""):
                self.set_ready(False)
                return
        self.set_ready(True)

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        if self._driver_cls is None:
            profile.machine_config.driver_args = None
            return True
        try:
            values = self.connect_widget.get_values()
        except ValueError as exc:
            self.wizard.show_error(_("Invalid input"), str(exc))
            return False
        # Drop empty-string / None values so we don't blur defaults.
        cleaned: dict = {}
        for key, value in values.items():
            if value in (None, ""):
                continue
            cleaned[key] = value
        profile.machine_config.driver_args = cleaned or None
        return True


__all__ = ["ConnectionPage"]

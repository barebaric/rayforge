"""Step 11 — Review & name.

Summarizes every value the wizard has collected so far, runs a
static config sanity pass, and lets the user pick a final name for
the new machine. The "Create Machine" button at the wizard footer
is the user's commit point — the orchestrator calls
``DeviceProfile.create_machine()`` and hands the live ``Machine``
back to the caller via the ``machine_created`` signal.
"""

from gettext import gettext as _

from gi.repository import Adw

from ....machine.device.profile import DeviceProfile
from ....machine.driver import get_driver_cls
from ....machine.models.machine import Origin
from ....shared.units.system import UnitSystem
from . import WizardPage, _makePreferencesGroup


def _format_tuple(value) -> str:
    if value is None:
        return _("—")
    try:
        return " × ".join(str(v) for v in value)
    except TypeError:
        return str(value)


def _format(value) -> str:
    if value is None:
        return _("—")
    if isinstance(value, bool):
        return _("Yes") if value else _("No")
    return str(value)


def _format_bool(value) -> str:
    """Boolean with an unset (None) state collapsed to "No"."""
    return _("Yes") if value else _("No")


_ORIGIN_LABELS = {
    Origin.BOTTOM_LEFT: _("Bottom Left"),
    Origin.TOP_LEFT: _("Top Left"),
    Origin.TOP_RIGHT: _("Top Right"),
    Origin.BOTTOM_RIGHT: _("Bottom Right"),
}

_UNIT_SYSTEM_LABELS = {
    UnitSystem.METRIC: _("Metric (mm)"),
    UnitSystem.IMPERIAL: _("Imperial (inches)"),
}

_SECRET_ARG_KEYS = ("api_key", "password", "secret", "token")


def _format_connection(mc) -> str:
    """Human-readable summary of the connection arguments.

    Renders ``driver_args`` as "Label: value" pairs using the
    driver's own setup-var labels instead of dumping the raw dict,
    and masks secret-ish values (API keys, passwords).
    """
    args = mc.driver_args or {}
    if not args:
        return _("—")
    labels: dict[str, str] = {}
    if mc.driver:
        try:
            d = get_driver_cls(mc.driver)
            for var in d.get_setup_vars():
                labels[var.key] = var.label
        except (ValueError, TypeError):
            labels = {}
    parts: list[str] = []
    for key, value in args.items():
        label = labels.get(key) or key
        if any(s in key.lower() for s in _SECRET_ARG_KEYS):
            value = "••••••••"
        else:
            value = _format(value)
        parts.append(f"{label}: {value}")
    return ", ".join(parts)


def _prefill_name(profile: DeviceProfile) -> str:
    """Machine-name suggestion for the review page.

    Keeps an explicit profile name, otherwise composes one from the
    vendor + model entered on the AI lookup page (e.g. "Sculpfun S30
    Pro") and falls back to the default placeholder.
    """
    name = (profile.meta.name or "").strip()
    if name and name != _("New Machine"):
        return name
    parts = [p for p in (profile.meta.vendor, profile.meta.model) if p]
    return " ".join(parts) if parts else _("New Machine")


class ReviewPage(WizardPage):
    step_number = 11
    title = _("Review & Name")
    subtitle = _("Final name and sanity check before creating the machine.")

    def __init__(self, wizard, **kwargs):
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        name_group = _makePreferencesGroup(
            title=_("Name"), description=_("A friendly name for this machine.")
        )
        self.content.append(name_group)

        self.name_row = Adw.EntryRow(title=_("Machine Name"))
        name_group.add(self.name_row)

        self.summary_group = _makePreferencesGroup(title=_("Summary"))
        self.content.append(self.summary_group)
        self._summary_rows: list[Adw.ActionRow] = []

        # Warnings surface config issues (e.g. driver missing) but
        # do not block machine creation — they round-trip into the
        # settings dialog where the user can fix them.
        self.warnings_group = _makePreferencesGroup(title=_("Warnings"))
        self.warnings_group.set_visible(False)
        self.content.append(self.warnings_group)
        self._warning_rows: list[Adw.ActionRow] = []

        self.set_ready(True)

    def enter(self, profile: DeviceProfile) -> None:
        self.name_row.set_text(_prefill_name(profile))
        self._populate_summary(profile)
        self._populate_warnings(profile)

    def _populate_summary(self, profile: DeviceProfile) -> None:
        for row in self._summary_rows:
            self.summary_group.remove(row)
        self._summary_rows.clear()

        mc = profile.machine_config

        driver_label = _("None (G-code export only)")
        if mc.driver:
            d = get_driver_cls(mc.driver)
            # get_driver_cls returns NoDeviceDriver when the class
            # name is not in the registry; use the class name as a
            # fallback label so the user knows the configured driver
            # couldn't be looked up.
            if d.__name__ == "NoDeviceDriver" and d.__name__ != mc.driver:
                driver_label = _("Unknown driver: {}").format(mc.driver)
            else:
                driver_label = d.label

        origin = mc.origin if mc.origin is not None else Origin.BOTTOM_LEFT
        origin_label = _ORIGIN_LABELS[origin]
        unit_system = mc.unit_system or UnitSystem.METRIC
        unit_system_label = _UNIT_SYSTEM_LABELS[unit_system]
        rows_data = [
            (_("Driver"), driver_label),
            (_("Connection"), _format_connection(mc)),
            (_("Work Area X×Y"), _format_tuple(mc.axis_extents)),
            (_("Origin"), origin_label),
            (_("Unit System"), unit_system_label),
            (_("Max Travel Speed"), _format(mc.max_travel_speed)),
            (_("Max Cut Speed"), _format(mc.max_cut_speed)),
            (_("Acceleration"), _format(mc.acceleration)),
            (_("Home on Start"), _format_bool(mc.home_on_start)),
            (_("Heads"), str(len(mc.heads or []))),
            (_("Rotary Modules"), str(len(mc.rotary_modules or []))),
            (_("Cameras"), str(len(mc.cameras or []))),
        ]
        for label, value in rows_data:
            row = Adw.ActionRow(title=label, subtitle=value)
            self.summary_group.add(row)
            self._summary_rows.append(row)

    def _populate_warnings(self, profile: DeviceProfile) -> None:
        warnings: list[str] = self._check_profile(profile)
        for row in self._warning_rows:
            self.warnings_group.remove(row)
        self._warning_rows.clear()
        if not warnings:
            self.warnings_group.set_visible(False)
            return
        for text in warnings:
            row = Adw.ActionRow(title=text)
            row.add_css_class("warning")
            self.warnings_group.add(row)
            self._warning_rows.append(row)
        self.warnings_group.set_visible(True)

    def _check_profile(self, profile: DeviceProfile) -> list[str]:
        warnings: list[str] = []
        mc = profile.machine_config

        if not mc.driver:
            warnings.append(
                _(
                    "No driver selected — this machine will only "
                    "export G-code to files; it cannot run jobs."
                )
            )

        toplefts = mc.axis_extents or (0, 0)
        if not toplefts or min(toplefts) <= 0:
            warnings.append(
                _("Work area dimensions are unset or non-positive.")
            )

        if not (mc.heads or []):
            warnings.append(_("No head is configured for this machine."))
        else:
            for idx, head in enumerate(mc.heads):
                cls = (head.get("head_class") or "").lower()
                if "laser" in cls and "max_power" not in head:
                    warnings.append(
                        _(
                            "Head #{n} looks like a laser but has "
                            "no max_power setting."
                        ).format(n=idx + 1)
                    )
                if "spindle" in cls and "max_rpm" not in head:
                    warnings.append(
                        _(
                            "Head #{n} looks like a spindle but has "
                            "no max_rpm setting."
                        ).format(n=idx + 1)
                    )

        if not profile.meta.name or not profile.meta.name.strip():
            warnings.append(_("Machine name is blank."))

        return warnings

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        name = self.name_row.get_text().strip()
        if not name:
            self.wizard.show_error(
                _("Missing name"), _("Please enter a name.")
            )
            return False
        profile.meta.name = name
        return True


__all__ = ["ReviewPage"]

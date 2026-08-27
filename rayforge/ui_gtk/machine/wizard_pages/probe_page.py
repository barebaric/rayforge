"""Step 4 — Connect + auto-detect.

Only shown when the chosen driver's ``supports_probing`` is True
(GRBL, Marlin, Smoothie, and OctoPrint today). Reuses the driver's
classmethod ``probe()`` plumbing originally developed for the legacy
``ConfigWizard``.

The page offers:

* **Probe now** — attempts to connect to the device using the
  Step-3 connection parameters and read its configuration
  ($I, axis extents $130/$131, speeds $110/$111, accel, laser mode,
  RX buffer size). On success the resulting ``DeviceProfile`` fields
  are merged into the working profile and Step 6/7 inputs are
  prefilled and marked verified.
* **Skip** — the user forgoes probing now; wizard falls through to
  Step 5 (AI lookup, if applicable) or Step 6 manual entry.
"""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from blinker import Signal
from gi.repository import Adw, Gtk

from ....context import get_context
from ....machine.device.profile import DeviceProfile
from ....machine.driver import Driver, get_driver_cls
from ....machine.driver.driver import DriverPrecheckError
from ....shared.tasker import Task, task_mgr
from ....shared.tasker.context import ExecutionContext
from . import WizardPage, _makePreferencesGroup

if TYPE_CHECKING:
    from ..unified_wizard import UnifiedWizard


class ProbePage(WizardPage):
    step_number = 5
    title = _("Probe Device")
    subtitle = _(
        "Connect to the device and read its configuration, "
        "or skip to enter the values manually."
    )

    # Sent after a successful probe. Payload ``(profile, warnings)``
    # where ``profile`` is the working profile with probed values
    # merged in; warnings is a list of human-readable strings.
    def __init__(self, wizard: "UnifiedWizard", **kwargs: Any) -> None:
        self._driver_cls: type[Driver] | None = None
        # True once probing has been attempted on this page instance,
        # so re-entering via Back does not auto-restart a probe.
        self._probed: bool = False
        self.probe_succeeded = Signal()
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        self.group = _makePreferencesGroup(
            title=_("Probing"),
            description=_(
                "Auto-discover the machine's working area, "
                "speeds, and firmware capabilities by reading its "
                "settings over the connection."
            ),
        )
        self.content.append(self.group)

        self.status_row = Adw.ActionRow(
            title=_("Status"),
            subtitle=_("Idle"),
        )
        self.group.add(self.status_row)

        self.spinner = Gtk.Spinner()
        self.spinner.set_halign(Gtk.Align.CENTER)
        self.spinner.set_size_request(32, 32)
        self.spinner.set_visible(False)
        self.content.append(self.spinner)

        # Probe / Retry live on the wizard footer button bar (see
        # footer_buttons()). A single button relabels itself between
        # "Probe Now" and "Retry" so there is never a redundant pair.
        self.probe_button = Gtk.Button(label=_("Probe Now"))
        self.probe_button.add_css_class("suggested-action")
        self.probe_button.connect("clicked", lambda *_: self._start_probe())

        # The page is considered ready by default so the orchestrator's
        # Next button stays sensitive. Skip is the minimal action.
        # The orchestrator may disable Next while a probe is in
        # flight via set_ready.
        self.set_ready(True)

    def footer_buttons(self) -> list[Gtk.Button]:
        return [self.probe_button]

    def enter(self, profile: DeviceProfile) -> None:
        driver_name = profile.machine_config.driver
        self._driver_cls = get_driver_cls(driver_name) if driver_name else None
        self._reset_status()
        # Auto-start probing the first time the page is shown, but not
        # when the user navigates back to it.
        if not self._probed:
            self._probed = True
            self._start_probe()

    def _reset_status(self) -> None:
        self.status_row.set_title(_("Status"))
        self.status_row.set_subtitle(_("Idle"))
        self.status_row.remove_css_class("error")
        self.spinner.stop()
        self.spinner.set_visible(False)
        self.probe_button.set_label(_("Probe Now"))
        self.probe_button.set_sensitive(True)
        self.set_ready(True)

    # ----- probe coroutine ----------------------------------------------

    def _start_probe(self) -> None:
        driver_cls = self._driver_cls
        if driver_cls is None:
            return
        profile = self.wizard.profile
        params = dict(profile.machine_config.driver_args or {})

        # Lightweight precheck (var_set validation, port format, ...).
        try:
            driver_cls.precheck(**params)
        except DriverPrecheckError as exc:
            self._show_error(str(exc))
            return

        self.status_row.set_title(_("Probing…"))
        self.status_row.set_subtitle(
            _("Connecting to device and reading settings")
        )
        self.status_row.remove_css_class("error")
        self.spinner.set_visible(True)
        self.spinner.start()
        self.probe_button.set_sensitive(False)
        self.probe_button.set_label(_("Probing…"))
        self.set_ready(False)

        context = get_context()

        async def _coroutine(
            exec_ctx: ExecutionContext,
        ) -> tuple[DeviceProfile, list[str]]:
            return await driver_cls.probe(context, **params)

        task_mgr.add_coroutine(
            _coroutine,
            key="unified_wizard_probe",
            when_done=self._on_probe_done,
        )

    def _on_probe_done(self, task: Task) -> None:
        # Marshal back to the GTK main thread before touching widgets.
        def _update():
            try:
                result = task.result()
            except Exception as exc:  # noqa: BLE001 - async task boundary
                self._show_error(str(exc) or _("Probe failed"))
                return

            if task.get_status() != "completed":
                msg = _("Probe failed")
                self._show_error(msg)
                return

            profile, warnings = result
            self.status_row.set_title(_("Probe succeeded"))
            self.status_row.set_subtitle(
                _("Working area and speeds auto-detected.")
            )
            self.spinner.stop()
            self.spinner.set_visible(False)
            self.probe_button.set_label(_("Probe Now"))
            self.probe_button.set_sensitive(False)
            self.set_ready(True)
            self.probe_succeeded.send(self, profile=profile, warnings=warnings)

        task_mgr.schedule_on_main_thread(_update)

    def _show_error(self, message: str) -> None:
        self.status_row.set_title(_("Error"))
        self.status_row.set_subtitle(message)
        self.status_row.add_css_class("error")
        self.spinner.stop()
        self.spinner.set_visible(False)
        self.probe_button.set_label(_("Retry"))
        self.probe_button.set_sensitive(True)
        self.set_ready(True)


__all__ = ["ProbePage"]

"""Step 1 — Automatic device discovery.

Starts scanning for machines the moment the wizard opens — before
the user has selected anything. Every driver that declares a
``DISCOVERY`` recognizer participates (see
:mod:`rayforge.machine.driver.discovery`); drivers that declare
``MDNS_SERVICES`` are found over the network the same way. Found
devices are listed as rows the user can activate to adopt them.

Discovery and data collection are fully automatic: as soon as a
device is identified, its driver is asked to probe it (firmware
build info, working area, speeds) and the row is updated with what
the device reported — e.g. "Sculpfun iCube, 120×120 mm". The user
never interacts with this page beyond picking a row.

The page keeps polling while it is visible so devices plugged in
later show up on their own. Ports of already-found devices are
excluded from rescans: re-opening them would reset some boards
(DTR toggle) and fight with an in-flight probe. The footer's Next
button is relabeled "Configure Manually" and leads to the regular
profile-first flow.
"""

import logging
from dataclasses import replace
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from blinker import Signal
from gi.repository import Adw, GLib, Gtk

from ....context import get_context
from ....machine.device.profile import DeviceProfile
from ....machine.driver import (
    DiscoveredDevice,
    find_all_devices,
    get_driver_cls,
)
from ....machine.transport import SerialTransport
from ....shared.tasker import task_mgr
from ....shared.tasker.context import ExecutionContext
from . import WizardPage, _makePreferencesGroup

if TYPE_CHECKING:
    from ..unified_wizard import UnifiedWizard

logger = logging.getLogger(__name__)

# Seconds between automatic re-scans while the page is visible.
_RESCAN_INTERVAL_S = 5


class _DeviceRow(Adw.ActionRow):
    """A custom row to hold a reference to its discovered device."""

    def __init__(self, device: DiscoveredDevice, **kwargs: Any):
        super().__init__(**kwargs)
        self.device: DiscoveredDevice = device
        # Rows are not obviously clickable; an explicit Select
        # button states the action. The whole row stays activatable.
        self.select_button = Gtk.Button(label=_("Select"))
        self.select_button.set_valign(Gtk.Align.CENTER)
        self.select_button.add_css_class("suggested-action")
        self.select_button.connect("clicked", lambda *_: self.activate())
        self.add_suffix(self.select_button)


class DiscoverPage(WizardPage):
    step_number = 1
    title = _("Add a Machine")
    subtitle = _(
        "Connect your machine via USB or network — we will find it "
        "automatically."
    )
    next_label = _("Configure Manually")

    # Sent when the user activates a discovered-device row. Payload
    # ``device`` is the picked DiscoveredDevice, including any probe
    # data collected by then.
    device_selected = Signal()

    def __init__(self, wizard: "UnifiedWizard", **kwargs: Any) -> None:
        # Bumped on every scan start; stale completion callbacks are
        # ignored by comparing against this.
        self._scan_generation: int = 0
        self._device_rows: dict[str, _DeviceRow] = {}
        # Ports of found devices; excluded from rescans (see module
        # docstring) and monitored for unplug.
        self._held_ports: set[str] = set()
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        self.group = _makePreferencesGroup(
            title=_("Detected Devices"),
            description=_(
                "Plug in and power on your machine. Devices found on "
                "USB or network connections appear here automatically."
            ),
        )
        self.content.append(self.group)

        self.spinner = Gtk.Spinner()
        self.status_label = Gtk.Label(label=_("Searching for devices…"))
        self.status_label.add_css_class("dim-label")
        self.status_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            halign=Gtk.Align.CENTER,
            margin_top=6,
            margin_bottom=6,
        )
        self.status_box.append(self.spinner)
        self.status_box.append(self.status_label)
        self.content.append(self.status_box)

        self.set_ready(True)

    def enter(self, profile: DeviceProfile) -> None:
        # (Re-)start discovery from scratch every time the page is
        # shown; the generation bump invalidates stale callbacks.
        self._start_scan()

    def footer_buttons(self) -> list[Gtk.Button]:
        return []

    # ----- scanning --------------------------------------------------------

    def _start_scan(self) -> None:
        self._scan_generation += 1
        generation = self._scan_generation
        self._show_scanning()

        async def _coroutine(exec_ctx: ExecutionContext) -> Any:
            return await find_all_devices(exclude_ports=self._held_ports)

        task_mgr.add_coroutine(
            _coroutine,
            key=f"wizard-device-discovery-{id(self.wizard)}",
            when_done=lambda task: self._on_scan_done(task, generation),
        )

    def _on_scan_done(self, task: Any, generation: int) -> None:
        # Marshal back to the GTK main thread before touching widgets.
        def _update() -> None:
            if generation != self._scan_generation or not self.get_mapped():
                return
            try:
                devices = task.result()
            except Exception:
                logger.exception("Device discovery failed")
                devices = []
            if task.get_status() != "completed":
                return
            self._prune_unplugged_ports()
            self._update_devices(devices or [])
            GLib.timeout_add_seconds(_RESCAN_INTERVAL_S, self._on_rescan_timer)

        task_mgr.schedule_on_main_thread(_update)

    def _on_rescan_timer(self) -> bool:
        """Periodic re-scan while the page is visible."""
        if not self.get_mapped():
            return False
        self._start_scan()
        return False

    def _show_scanning(self) -> None:
        self.status_label.set_text(_("Searching for devices…"))
        self.spinner.start()
        self.status_box.set_visible(True)

    def _update_devices(self, devices: list[DiscoveredDevice]) -> None:
        """Reconciles the visible rows with the scan results."""
        keep: set[str] = set()
        for device in devices:
            keep.add(device.key)
            port = device.params.get("port")
            if isinstance(port, str):
                self._held_ports.add(port)
            existing = self._device_rows.get(device.key)
            if existing is not None:
                continue
            row = _DeviceRow(
                device=device,
                title=device.label,
                subtitle=_row_subtitle(device),
                activatable=True,
            )
            row.connect("activated", self._on_row_activated)
            self._device_rows[device.key] = row
            self.group.add(row)
            self._start_probe(device)

        for key, row in list(self._device_rows.items()):
            if key in keep:
                continue
            # Devices on held ports are excluded from rescans, so
            # their absence from the results is expected; they are
            # dropped by _prune_unplugged_ports instead.
            port = row.device.params.get("port")
            if isinstance(port, str) and port in self._held_ports:
                continue
            self._remove_row(key)

        if self._device_rows:
            self.spinner.stop()
            self.status_box.set_visible(False)
        else:
            self.status_label.set_text(_("No devices detected yet"))
            self.spinner.start()
            self.status_box.set_visible(True)

    def _remove_row(self, key: str) -> None:
        row = self._device_rows.pop(key, None)
        if row is not None:
            self.group.remove(row)

    def _prune_unplugged_ports(self) -> None:
        """Drops rows whose port has vanished from the system."""
        try:
            present = {i.device for i in SerialTransport.list_port_info()}
        except Exception:
            logger.debug(
                "Could not enumerate ports for pruning", exc_info=True
            )
            return
        for key, row in list(self._device_rows.items()):
            port = row.device.params.get("port")
            if isinstance(port, str) and port not in present:
                self._held_ports.discard(port)
                self._remove_row(key)

    # ----- probing ---------------------------------------------------------

    def _start_probe(self, device: DiscoveredDevice) -> None:
        """Asks the device itself for its specs, without interaction.

        The row is already usable when the probe finishes; the probe
        only enriches it (machine name, working area) and equips the
        wizard to match a device profile automatically.
        """
        driver_cls = get_driver_cls(device.driver_name)
        if driver_cls is None or not driver_cls.supports_probing:
            return
        context = get_context()
        params = dict(device.params)

        async def _coroutine(exec_ctx: ExecutionContext) -> Any:
            return await driver_cls.probe(context, **params)

        task_mgr.add_coroutine(
            _coroutine,
            key=f"wizard-device-probe-{device.key}",
            when_done=lambda task: self._on_probe_done(task, device),
        )

    def _on_probe_done(self, task: Any, device: DiscoveredDevice) -> None:
        def _update() -> None:
            row = self._device_rows.get(device.key)
            if row is None:
                return
            try:
                profile, warnings = task.result()
            except Exception:
                logger.debug(
                    "Probe of discovered device %s failed",
                    device.key,
                    exc_info=True,
                )
                return
            for text in warnings:
                logger.info("Probing warning: %s", text)
            enriched = replace(device, probe_profile=profile)
            row.device = enriched
            name = enriched.probe_name
            if name:
                row.set_title(name)
            row.set_subtitle(_row_subtitle(enriched))

        task_mgr.schedule_on_main_thread(_update)

    def _on_row_activated(self, row: _DeviceRow) -> None:
        self.device_selected.send(self, device=row.device)

    def _on_select_clicked(self, _button: Gtk.Button, row: _DeviceRow) -> None:
        self._on_row_activated(row)


def _row_subtitle(device: DiscoveredDevice) -> str:
    """The two-line description under a device row."""
    parts = [device.detail]
    if device.probe_profile is not None:
        extents = device.probe_profile.machine_config.axis_extents
        if extents is not None:
            width, height = extents
            parts.append(_("{w}×{h} mm work area").format(w=width, h=height))
    elif device.identity.banner:
        parts.append(device.identity.banner)
    return "\n".join(parts)


__all__ = ["DiscoverPage"]

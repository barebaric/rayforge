"""Step 1 — Automatic device discovery.

Starts scanning for machines the moment the wizard opens — before
the user has selected anything. Every driver that declares a
``DISCOVERY`` spec participates (see
:mod:`rayforge.machine.discovery`). Found devices are listed as
rows the user can activate to adopt them.

Discovery and data collection are fully automatic: as soon as a
device is identified, its driver is asked to probe it (firmware
build info, working area, speeds) and the row is updated with what
the device reported — e.g. "Sculpfun iCube, 120×120 mm". The user
never interacts with this page beyond picking a row.

All scan/probe state lives in the GTK-free
:class:`~rayforge.machine.discovery.DiscoverySession`; this page
only feeds results in and maps them onto rows.

The page keeps polling while it is visible so devices plugged in
later show up on their own. Ports of already-found devices are
excluded from rescans: re-opening them would reset some boards
(DTR toggle) and fight with an in-flight probe. The footer's Next
button is relabeled "Configure Manually" and leads to the regular
profile-first flow.
"""

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from blinker import Signal
from gi.repository import Adw, GLib, Gtk

from ....context import get_context
from ....machine.device.profile import DeviceProfile
from ....machine.discovery import (
    DiscoveredDevice,
    DiscoverySession,
    device_key,
    find_all_devices,
)
from ....machine.driver import get_driver_cls
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

    select_button: Gtk.Button | None

    def __init__(
        self, device: DiscoveredDevice, configured: bool = False, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.device: DiscoveredDevice = device
        self.configured: bool = configured
        if configured:
            # Already set up in the app: show a badge instead of a
            # Select button and make the row non-activatable so the
            # user cannot pick it.
            self.set_activatable(False)
            badge = Gtk.Label(label=_("Configured"))
            badge.add_css_class("dim-label")
            badge.set_valign(Gtk.Align.CENTER)
            self.add_suffix(badge)
            self.select_button = None
        else:
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
        self.session = DiscoverySession()
        self._device_rows: dict[str, _DeviceRow] = {}
        # Connection keys of machines already configured in the app,
        # refreshed every time the page is entered so newly added or
        # removed machines are reflected. Discovered devices whose key
        # appears here are shown as read-only "Configured" rows.
        self._configured_keys: set[str] = set()
        # Key -> configured Machine, used to enrich the row with the
        # machine's real name and work area instead of the generic
        # discovery label.
        self._configured_machines: dict[str, Any] = {}
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
        # (Re-)start scanning every time the page is shown; the
        # generation bump invalidates stale callbacks. Previously
        # found devices stay listed (their ports remain held).
        self._refresh_configured_keys()
        self._start_scan()

    def _refresh_configured_keys(self) -> None:
        """Recomputes the set of connection keys already claimed by
        configured machines, so discovered devices matching one are
        shown as read-only."""
        self._configured_keys = set()
        self._configured_machines = {}
        try:
            machines = get_context().machine_mgr.get_machines()
        except Exception:
            logger.debug("Could not load machines", exc_info=True)
            return
        for machine in machines:
            # Placeholder machines are auto-created stubs, not real
            # configurations, so they never block discovery.
            if getattr(machine, "placeholder", False):
                continue
            if not machine.driver_name:
                continue
            key = device_key(machine.driver_name, machine.driver_args)
            self._configured_keys.add(key)
            self._configured_machines[key] = machine

    def footer_buttons(self) -> list[Gtk.Button]:
        return []

    # ----- scanning --------------------------------------------------------

    def _start_scan(self) -> None:
        self._scan_generation += 1
        generation = self._scan_generation
        self._show_scanning()

        async def _coroutine(exec_ctx: ExecutionContext) -> Any:
            return await find_all_devices(
                exclude_ports=self.session.held_ports
            )

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
            present = self._present_ports()
            if present is not None:
                self._prune_absent_ports(present)
            self._apply_scan_result(devices or [])
            GLib.timeout_add_seconds(_RESCAN_INTERVAL_S, self._on_rescan_timer)

        task_mgr.schedule_on_main_thread(_update)

    def _apply_scan_result(self, devices: list[DiscoveredDevice]) -> None:
        """Feeds a completed scan into the session and reconciles the
        visible rows with it."""
        added, removed_keys = self.session.apply_scan(devices)
        self._remove_rows(removed_keys)
        for device in added:
            self._add_row(device)
            # Probing an already-configured device is pointless: the
            # user cannot select it and the probe would only enrich a
            # row that stays read-only.
            if device.key not in self._configured_keys:
                self._start_probe(device)
        self._update_status()

    def _present_ports(self) -> set[str] | None:
        """Serial ports currently present on the system, or None
        when they could not be enumerated (pruning is skipped)."""
        try:
            return {i.device for i in SerialTransport.list_port_info()}
        except Exception:
            logger.debug(
                "Could not enumerate ports for pruning", exc_info=True
            )
            return None

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

    def _update_status(self) -> None:
        if self._device_rows:
            self.spinner.stop()
            self.status_box.set_visible(False)
        else:
            self.status_label.set_text(_("No devices detected yet"))
            self.spinner.start()
            self.status_box.set_visible(True)

    # ----- rows ------------------------------------------------------------

    def _add_row(self, device: DiscoveredDevice) -> None:
        machine = self._configured_machines.get(device.key)
        configured = machine is not None
        if configured:
            title = machine.name or device.label
            subtitle = _configured_row_subtitle(device, machine)
        else:
            title = device.label
            subtitle = _row_subtitle(device)
        row = _DeviceRow(
            device=device,
            configured=configured,
            title=title,
            subtitle=subtitle,
            activatable=not configured,
        )
        row.connect("activated", self._on_row_activated)
        self._device_rows[device.key] = row
        self.group.add(row)

    def _remove_rows(self, keys: list[str]) -> None:
        for key in keys:
            self._remove_row(key)

    def _prune_absent_ports(self, present_ports: set[str]) -> list[str]:
        """Drops rows whose port has vanished from the system."""
        removed = self.session.prune_absent_ports(present_ports)
        self._remove_rows(removed)
        return removed

    def _remove_row(self, key: str) -> None:
        row = self._device_rows.pop(key, None)
        if row is not None:
            self.group.remove(row)

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
            enriched = self.session.apply_probe(device.key, profile)
            if enriched is None:
                return
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


def _configured_row_subtitle(device: DiscoveredDevice, machine: Any) -> str:
    """Subtitle for a row backed by an already-configured machine:
    leads with the connection detail, then the configured work area
    (which is more trustworthy than a fresh probe of the same board)."""
    parts = [device.detail]
    try:
        width, height = machine.axis_extents
    except Exception:
        logger.debug("Could not read axis_extents", exc_info=True)
        width = height = None
    if width and height:
        parts.append(_("{w}×{h} mm work area").format(w=width, h=height))
    return "\n".join(parts)


__all__ = ["DiscoverPage"]

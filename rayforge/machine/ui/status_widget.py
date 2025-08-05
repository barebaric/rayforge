from gi.repository import Gtk  # type: ignore
from typing import Optional
from ...machine.driver.driver import DeviceStatus, DeviceState
from ...machine.models.machine import Machine


class MachineStatusIconWidget(Gtk.Box):
    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        # Create an image widget to display the status icon
        self.status_image = Gtk.Image()
        self.append(self.status_image)

        # Set the initial status
        self.set_status(DeviceStatus.UNKNOWN)

    def set_status(self, status):
        """Update the status icon based on the given status."""
        icon_name = self._get_icon_name_for_status(status)
        self.status_image.set_from_icon_name(icon_name)

    def _get_icon_name_for_status(self, status):
        """Map the status to an appropriate symbolic icon name."""
        if status == DeviceStatus.UNKNOWN:
            return "dialog-question-symbolic"
        elif status == DeviceStatus.IDLE:
            return "emblem-ok-symbolic"
        elif status == DeviceStatus.RUN:
            return "media-playback-start-symbolic"
        elif status == DeviceStatus.HOLD:
            return "media-playback-pause-symbolic"
        elif status == DeviceStatus.JOG:
            return "media-seek-forward-symbolic"
        elif status == DeviceStatus.ALARM:
            return "dialog-warning-symbolic"
        elif status == DeviceStatus.DOOR:
            return "system-lock-screen-symbolic"
        elif status == DeviceStatus.CHECK:
            return "system-search-symbolic"
        elif status == DeviceStatus.HOME:
            return "go-home-symbolic"
        elif status == DeviceStatus.SLEEP:
            return "system-suspend-symbolic"
        elif status == DeviceStatus.TOOL:
            return "preferences-system-symbolic"
        elif status == DeviceStatus.QUEUE:
            return "view-list-bullet-symbolic"
        elif status == DeviceStatus.LOCK:
            return "system-lock-screen-symbolic"
        elif status == DeviceStatus.UNLOCK:
            return "process-working-symbolic"
        elif status == DeviceStatus.CYCLE:
            return "view-refresh-symbolic"
        elif status == DeviceStatus.TEST:
            return "utilities-terminal-symbolic"
        else:
            return "network-offline-symbolic"


class MachineStatusWidget(Gtk.Box):
    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.machine: Optional[Machine] = None

        self.label = Gtk.Label()
        self.append(self.label)

        self.icon = MachineStatusIconWidget()
        self.append(self.icon)

        self._update_display(DeviceState())  # Initial default state

    def set_machine(self, machine: Optional[Machine]):
        if self.machine:
            try:
                self.machine.state_changed.disconnect(self._on_state_changed)
            except TypeError:
                pass  # Was not connected

        self.machine = machine

        if self.machine:
            self.machine.state_changed.connect(self._on_state_changed)
            self._update_display(self.machine.device_state)
        else:
            self._update_display(None)

    def _on_state_changed(self, machine: Machine, state: DeviceState):
        self._update_display(state)

    def _update_display(self, state: Optional[DeviceState]):
        is_nodriver = not self.machine or not self.machine.driver
        status = state.status if state else DeviceStatus.UNKNOWN

        if is_nodriver:
            self.label.set_label(_("No driver selected"))
            self.icon.set_status(DeviceStatus.UNKNOWN)
        else:
            self.label.set_label(status.name)
            self.icon.set_status(status)

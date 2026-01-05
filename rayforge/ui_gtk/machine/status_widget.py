from gi.repository import Gtk
from typing import Optional
from ..icons import get_icon
from ...machine.driver.driver import (
    DeviceStatus,
    DeviceState,
    DEVICE_STATUS_LABELS,
)
from ...machine.models.machine import Machine


class MachineStatusIconWidget(Gtk.Box):
    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        # Placeholder for the image widget
        self.status_image: Optional[Gtk.Widget] = None

        # Set the initial status
        self.set_status(DeviceStatus.UNKNOWN)

    def set_status(self, status):
        """Update the status icon based on the given status."""
        icon_name = self._get_icon_name_for_status(status)

        # Get the new image widget from the helper
        new_image = get_icon(icon_name)

        # Remove the old image if it exists
        if self.status_image is not None:
            self.remove(self.status_image)

        # Set and add the new image
        self.status_image = new_image
        if self.status_image:
            self.append(self.status_image)

    def _get_icon_name_for_status(self, status):
        """Map the status to an appropriate symbolic icon name."""
        if status == DeviceStatus.UNKNOWN:
            return "question-box-symbolic"
        elif status == DeviceStatus.IDLE:
            return "status-idle-symbolic"
        elif status == DeviceStatus.RUN:
            return "play-arrow-symbolic"
        elif status == DeviceStatus.HOLD:
            return "pause-symbolic"
        elif status == DeviceStatus.JOG:
            return "jog-symbolic"
        elif status == DeviceStatus.ALARM:
            return "alarm-symbolic"
        elif status == DeviceStatus.DOOR:
            return "door-symbolic"
        elif status == DeviceStatus.CHECK:
            return "status-check-symbolic"
        elif status == DeviceStatus.HOME:
            return "home-symbolic"
        elif status == DeviceStatus.SLEEP:
            return "sleep-symbolic"
        elif status == DeviceStatus.TOOL:
            return "machine-settings-general-symbolic"
        elif status == DeviceStatus.QUEUE:
            return "batch-symbolic"
        elif status == DeviceStatus.LOCK:
            return "lock-symbolic"
        elif status == DeviceStatus.UNLOCK:
            return "lock-open-symbolic"
        elif status == DeviceStatus.CYCLE:
            return "refresh-symbolic"
        elif status == DeviceStatus.TEST:
            return "test-symbolic"
        else:
            return "status-offline-symbolic"


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
            self.label.set_label(
                DEVICE_STATUS_LABELS.get(status, _("Unknown"))
            )
            self.icon.set_status(status)

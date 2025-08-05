from gi.repository import Gtk  # type: ignore
from typing import Optional
from blinker import Signal
from ...machine.transport import TransportStatus
from ...machine.driver.driver import driver_mgr, DeviceState, DeviceStatus
from ...machine.driver.dummy import NoDeviceDriver


class ConnectionStatusIconWidget(Gtk.Box):
    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        # Create an image widget to display the status icon
        self.status_image = Gtk.Image()
        self.append(self.status_image)

        # Set the initial status
        self.set_status(TransportStatus.DISCONNECTED)

    def set_status(self, status):
        """Update the status icon based on the given status."""
        icon_name = self._get_icon_name_for_status(status)
        self.status_image.set_from_icon_name(icon_name)

    def _get_icon_name_for_status(self, status):
        """Map the status to an appropriate icon name."""
        if status == TransportStatus.UNKNOWN:
            return "network-error-symbolic"
        elif status == TransportStatus.IDLE:
            return "network-idle-symbolic"
        elif status == TransportStatus.CONNECTING:
            return "network-transmit-receive-symbolic"
        elif status == TransportStatus.CONNECTED:
            return "network-wired-symbolic"
        elif status == TransportStatus.ERROR:
            return "network-error-symbolic"
        elif status == TransportStatus.CLOSING:
            return "network-offline-symbolic"
        elif status == TransportStatus.DISCONNECTED:
            return "network-offline-symbolic"
        elif status == TransportStatus.SLEEPING:
            return "network-offline-symbolic"
        else:
            return "network-offline-symbolic"  # Default icon


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


class StatusWidget(Gtk.Box):
    def __init__(self, icon_widget, default_status):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        self.label = Gtk.Label()
        self.append(self.label)

        self.icon = icon_widget
        self.append(self.icon)

        self.set_status(default_status)

    def set_status(self, status):
        self.icon.set_status(status)
        if status is None:
            self.label.set_label(_("No driver selected"))
        else:
            self.label.set_label(status.name)


class ConnectionStatusMonitor(StatusWidget):
    def __init__(self):
        self.changed = Signal()
        self.status = TransportStatus.UNKNOWN
        super().__init__(ConnectionStatusIconWidget(), self.status)

        driver_mgr.changed.connect(self.on_driver_changed)
        self.on_driver_changed(driver_mgr, driver_mgr.driver)

    def on_driver_changed(self, manager, driver):
        nodriver = driver is None or driver.__class__ == NoDeviceDriver
        self.set_status(None if nodriver else DeviceStatus.UNKNOWN)
        if driver is None:
            return

        # The driver may be new, or it may just have been reconfigured.
        # So we disconnect the signal in case it was already connected.
        driver.connection_status_changed.disconnect(
            self.on_connection_status_changed
        )
        driver.connection_status_changed.connect(
            self.on_connection_status_changed
        )

    def on_connection_status_changed(self,
                                     driver,
                                     status: TransportStatus,
                                     message: Optional[str] = None):
        nodriver = driver_mgr.driver.__class__ == NoDeviceDriver
        self.set_status(None if nodriver else status)

    def set_status(self, status):
        self.status = status
        super().set_status(status)
        self.changed.send(self)

    def get_status(self):
        return self.status


class MachineStatusMonitor(StatusWidget):
    def __init__(self):
        self.changed = Signal()
        self.status = DeviceStatus.UNKNOWN
        self.state = None
        super().__init__(MachineStatusIconWidget(), self.status)

        driver_mgr.changed.connect(self.on_driver_changed)
        self.on_driver_changed(self, driver_mgr.driver)  # trigger update

    def on_driver_changed(self, sender, driver):
        nodriver = driver is None or driver.__class__ == NoDeviceDriver
        self.set_status(None if nodriver else DeviceStatus.UNKNOWN)
        if driver is None:
            return

        # The driver may be new, or it may just have been reconfigured.
        # So we disconnect the signal in case it was already connected.
        driver.state_changed.disconnect(self.on_driver_state_changed)
        driver.state_changed.connect(self.on_driver_state_changed)

    def on_driver_state_changed(self,
                                driver,
                                state: DeviceState):
        self.state = state
        nodriver = driver_mgr.driver.__class__ == NoDeviceDriver
        self.set_status(None if nodriver else state.status)

    def set_status(self, status):
        self.status = status
        super().set_status(status)
        self.changed.send(self)

    def get_status(self):
        return self.status

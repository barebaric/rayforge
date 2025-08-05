from gi.repository import Gtk  # type: ignore
from typing import Optional
from ...machine.transport import TransportStatus
from ...machine.models.machine import Machine


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


class ConnectionStatusWidget(Gtk.Box):
    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.machine: Optional[Machine] = None

        self.label = Gtk.Label()
        self.append(self.label)

        self.icon = ConnectionStatusIconWidget()
        self.append(self.icon)

        self._update_display(TransportStatus.DISCONNECTED)

    def set_machine(self, machine: Optional[Machine]):
        if self.machine:
            try:
                self.machine.connection_status_changed.disconnect(
                    self._on_connection_status_changed
                )
            except TypeError:
                pass  # Was not connected

        self.machine = machine

        if self.machine:
            self.machine.connection_status_changed.connect(
                self._on_connection_status_changed
            )
            # Set initial state from the machine object
            self._update_display(self.machine.connection_status)
        else:
            self._update_display(None)

    def _on_connection_status_changed(
        self,
        machine: Machine,
        status: TransportStatus,
        message: Optional[str] = None,
    ):
        self._update_display(status)

    def _update_display(self, status: Optional[TransportStatus]):
        is_nodriver = not self.machine or not self.machine.driver

        if is_nodriver or status is None:
            self.label.set_label(_("No driver selected"))
            self.icon.set_status(TransportStatus.DISCONNECTED)
        else:
            self.label.set_label(status.name)
            self.icon.set_status(status)

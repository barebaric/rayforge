from gi.repository import Gtk  # type: ignore
from typing import Optional
from blinker import Signal
from ...machine.ui.connection_status_widget import ConnectionStatusWidget
from ...machine.ui.status_widget import MachineStatusWidget
from ...machine.models.machine import Machine
from .progress_bar import ProgressBar


class TaskBar(Gtk.Box):
    """
    A comprehensive status widget that combines a progress bar with machine and
    connection status displays.
    """

    log_requested = Signal()

    def __init__(self, task_mgr):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)

        # Create a two-row progress and status widget.
        self.progress_widget = ProgressBar(task_mgr)
        self.progress_widget.add_css_class("statusbar")
        self.append(self.progress_widget)

        # Get the top row of the widget to add status monitors to it.
        status_row = self.progress_widget.status_box
        status_row.set_margin_start(12)
        status_row.set_margin_end(12)

        # Monitor machine status
        label = Gtk.Label()
        label.set_markup(_("<b>Machine status:</b>"))
        status_row.append(label)

        self.machine_status_widget = MachineStatusWidget()
        status_row.append(self.machine_status_widget)

        # Monitor connection status
        label = Gtk.Label()
        label.set_markup(_("<b>Connection status:</b>"))
        label.set_margin_start(12)
        status_row.append(label)

        self.connection_status_widget = ConnectionStatusWidget()
        status_row.append(self.connection_status_widget)

        # Open machine log if the status row is clicked.
        gesture = Gtk.GestureClick()
        gesture.connect("pressed", lambda *args: self.log_requested.send(self))
        status_row.add_controller(gesture)

    def set_machine(self, machine: Optional[Machine]):
        """
        Sets the machine to be monitored by the status widgets.

        Args:
            machine: The Machine instance to monitor, or None.
        """
        self.machine_status_widget.set_machine(machine)
        self.connection_status_widget.set_machine(machine)

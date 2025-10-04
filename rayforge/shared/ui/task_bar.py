import logging
from gi.repository import Gtk
from typing import Optional
from blinker import Signal
from ...machine.ui.connection_status_widget import ConnectionStatusWidget
from ...machine.ui.status_widget import MachineStatusWidget
from ...machine.models.machine import Machine
from ...core.ops import Ops
from .progress_bar import ProgressBar


logger = logging.getLogger(__name__)


class TaskBar(Gtk.Box):
    """
    A comprehensive status widget that combines a progress bar with machine and
    connection status displays.
    """

    log_requested = Signal()

    def __init__(self, task_mgr):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.task_mgr = task_mgr

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

        # Estimated machining time
        label = Gtk.Label()
        label.set_markup(_("<b>Est. time:</b>"))
        label.set_margin_start(12)
        status_row.append(label)

        self.estimated_time_label = Gtk.Label()
        self.estimated_time_label.set_text(_("No operations"))
        status_row.append(self.estimated_time_label)

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

    def set_estimated_time(self, time_seconds: float):
        """
        Updates the estimated machining time display.

        Args:
            time_seconds: The estimated time in seconds.
        """
        if time_seconds <= 0:
            self.estimated_time_label.set_text(_("No operations"))
        elif time_seconds < 60:
            self.estimated_time_label.set_text(
                _("{:.1f}s").format(time_seconds)
            )
        elif time_seconds < 3600:
            minutes = int(time_seconds // 60)
            seconds = int(time_seconds % 60)
            self.estimated_time_label.set_text(
                _("{}m {}s").format(minutes, seconds)
            )
        else:
            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            self.estimated_time_label.set_text(
                _("{}h {}m").format(hours, minutes)
            )

    def update_estimated_time_from_ops(self, ops: Ops):
        """
        Updates the estimated machining time display from Ops.

        Args:
            ops: The Ops object to calculate time from.
        """
        logger.debug(f"update_estimated_time_from_ops called with ops: {ops}")

        # Show "Calculating..." while processing
        self.estimated_time_label.set_text(_("Calculating..."))

        if not ops:
            logger.debug("Ops is None")
            self.set_estimated_time(0.0)
            return

        if ops.is_empty():
            logger.debug("Ops is empty")
            self.set_estimated_time(0.0)
            return

        logger.debug(f"Ops has {len(ops.commands)} commands")

        try:
            estimated_time = ops.estimate_time()
            logger.debug(
                f"Calculated estimated time: {estimated_time} seconds"
            )
            self.set_estimated_time(estimated_time)
        except Exception:
            logger.error(
                "Failed to calculate estimated time from Ops", exc_info=True
            )
            self.set_estimated_time(0.0)

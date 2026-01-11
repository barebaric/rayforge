import logging
from gi.repository import Gtk, Pango
from typing import Optional, Dict, Any
from blinker import Signal
from .machine.connection_status_widget import ConnectionStatusWidget
from .machine.status_widget import MachineStatusWidget
from ..machine.models.machine import Machine
from .shared.progress_bar import ProgressBar


logger = logging.getLogger(__name__)


def _format_time_for_display(time_seconds: float) -> str:
    """Formats a positive number of seconds for display."""
    if time_seconds < 60:
        return _("{:.0f}s").format(time_seconds)
    elif time_seconds < 3600:
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        return _("{}m {}s").format(minutes, seconds)
    else:
        hours = int(time_seconds // 3600)
        minutes = int((time_seconds % 3600) // 60)
        return _("{}h {}m").format(hours, minutes)


class TaskBar(Gtk.Box):
    """
    A comprehensive status bar with a three-column layout for machine status,
    task messages, and job progress, plus an overall progress bar.
    """

    log_requested = Signal()

    def __init__(self, task_mgr):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.task_mgr = task_mgr

        # --- Top Row: Status Information using CenterBox for robust layout ---
        status_row = Gtk.CenterBox()
        status_row.set_margin_start(12)
        status_row.set_margin_end(12)
        status_row.add_css_class("statusbar")
        self.append(status_row)

        # --- Left Column (Start Widget) ---
        left_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=3)
        status_row.set_start_widget(left_box)

        label = Gtk.Label()
        label.set_markup(_("<b>Machine:</b>"))
        left_box.append(label)
        self.machine_status_widget = MachineStatusWidget()
        left_box.append(self.machine_status_widget)

        label = Gtk.Label()
        label.set_markup(_("<b>Connection:</b>"))
        label.set_margin_start(12)
        left_box.append(label)
        self.connection_status_widget = ConnectionStatusWidget()
        left_box.append(self.connection_status_widget)

        # --- Middle Column (Center Widget) ---
        self.task_message_label = Gtk.Label(
            valign=Gtk.Align.CENTER,
            ellipsize=Pango.EllipsizeMode.END,
        )
        status_row.set_center_widget(self.task_message_label)

        # --- Right Column (End Widget) ---
        right_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        status_row.set_end_widget(right_box)

        # Container for Live Progress view
        self.live_progress_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        self.live_eta_label = Gtk.Label(
            valign=Gtk.Align.CENTER, width_chars=15, xalign=0
        )
        self.live_progress_box.append(self.live_eta_label)
        self.live_progress_bar = Gtk.ProgressBar(valign=Gtk.Align.CENTER)
        self.live_progress_box.append(self.live_progress_bar)
        self.live_progress_label = Gtk.Label(
            label="0%", valign=Gtk.Align.CENTER
        )
        self.live_progress_box.append(self.live_progress_label)
        right_box.append(self.live_progress_box)

        # Container for "Job Sent" view
        self.job_sent_label = Gtk.Label(
            label=_("Job Sent"), valign=Gtk.Align.CENTER
        )
        right_box.append(self.job_sent_label)

        # Separator between live and static views
        self.live_progress_separator = Gtk.Separator(
            orientation=Gtk.Orientation.VERTICAL,
            margin_top=6,
            margin_bottom=6,
        )
        right_box.append(self.live_progress_separator)

        # Container for Static Estimated Time view
        self.static_time_box = Gtk.Box(spacing=3, valign=Gtk.Align.CENTER)
        est_time_label_title = Gtk.Label()
        est_time_label_title.set_markup(_("<b>Est. time:</b>"))
        self.static_time_box.append(est_time_label_title)
        self.estimated_time_label = Gtk.Label(label=_("No operations"))
        self.static_time_box.append(self.estimated_time_label)
        right_box.append(self.static_time_box)

        # --- Bottom Row: Overall Task Progress Bar ---
        self.overall_progress_bar = ProgressBar(task_mgr)
        self.append(self.overall_progress_bar)

        # Set initial visibility
        self.task_message_label.set_visible(False)
        self.static_time_box.set_visible(True)
        self.live_progress_box.set_visible(False)
        self.job_sent_label.set_visible(False)
        self.live_progress_separator.set_visible(False)

        # Connect signals
        gesture = Gtk.GestureClick()
        gesture.connect("pressed", lambda *args: self.log_requested.send(self))
        status_row.add_controller(gesture)
        self.task_mgr.tasks_updated.connect(self._on_tasks_updated)

    def set_machine(self, machine: Optional[Machine]):
        self.machine_status_widget.set_machine(machine)
        self.connection_status_widget.set_machine(machine)

    def start_live_view(self, is_granular: bool):
        """
        Shows the live job progress indicators.
        """
        if is_granular:
            self.live_progress_box.set_visible(True)
            self.job_sent_label.set_visible(False)
            self.live_progress_separator.set_visible(True)
            self.live_progress_bar.set_fraction(0)
            self.live_progress_label.set_text("0%")
            self.live_eta_label.set_markup(
                f"<b>ETA:</b> {_('Calculating...')}"
            )
        else:
            self.live_progress_box.set_visible(False)
            self.job_sent_label.set_visible(True)
            self.live_progress_separator.set_visible(True)

    def stop_live_view(self):
        """
        Hides the live job progress indicators.
        """
        self.live_progress_box.set_visible(False)
        self.job_sent_label.set_visible(False)
        self.live_progress_separator.set_visible(False)

    def update_live_progress(self, metrics: Dict[str, Any]):
        """Updates the progress bar and percentage label during a live job."""
        fraction = metrics.get("progress_fraction", 0.0)
        self.live_progress_bar.set_fraction(fraction)
        self.live_progress_label.set_text(f"{fraction:.0%}")

        eta_seconds = metrics.get("eta_seconds")
        if eta_seconds is not None and eta_seconds > 0:
            eta_str = _format_time_for_display(eta_seconds)
            self.live_eta_label.set_markup(f"<b>ETA:</b> {eta_str}")
        else:
            self.live_eta_label.set_markup(
                f"<b>ETA:</b> {_('Calculating...')}"
            )

    def set_estimated_time(self, time_seconds: Optional[float]):
        if time_seconds is None:
            self.estimated_time_label.set_text(_("Calculating..."))
        elif time_seconds <= 0:
            self.estimated_time_label.set_text(_("No operations"))
        else:
            self.estimated_time_label.set_text(
                _format_time_for_display(time_seconds)
            )

    def _on_tasks_updated(self, sender, tasks, progress):
        """Updates the task message label."""
        has_tasks = bool(tasks)
        self.task_message_label.set_visible(has_tasks)

        if not has_tasks:
            self.task_message_label.set_text("")
            return

        oldest_task = tasks[0]
        message = oldest_task.get_message()
        status_text = message if message is not None else ""

        if status_text and len(tasks) > 1:
            status_text += _(" (+{tasks} more)").format(tasks=len(tasks) - 1)
        elif len(tasks) > 1:
            status_text = _("{tasks} tasks").format(tasks=len(tasks))

        self.task_message_label.set_text(status_text)

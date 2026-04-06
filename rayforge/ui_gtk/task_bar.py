import logging
from gi.repository import Gtk
from typing import Optional
from blinker import Signal
from ..machine.models.machine import Machine
from .shared.progress_bar import ProgressBar


logger = logging.getLogger(__name__)


class TaskBar(Gtk.Box):
    """
    A status bar with an overall progress bar.
    """

    log_requested = Signal()

    def __init__(self, task_mgr):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.task_mgr = task_mgr
        self.add_css_class("statusbar")

        # Overall Task Progress Bar
        self.overall_progress_bar = ProgressBar(task_mgr)
        self.append(self.overall_progress_bar)

        gesture = Gtk.GestureClick()
        gesture.connect("pressed", lambda *args: self.log_requested.send(self))
        self.add_controller(gesture)

    def set_machine(self, machine: Optional[Machine]):
        pass

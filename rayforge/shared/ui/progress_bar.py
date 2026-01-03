from gi.repository import Gtk
from .gtk import apply_css


class ProgressBar(Gtk.ProgressBar):
    """
    A simple, self-contained progress bar that automatically reflects the
    status of a TaskManager. It is a single thin bar that fades in when
    tasks are running and fades out when idle.
    """

    def __init__(self, task_manager):
        super().__init__(
            hexpand=True,
            halign=Gtk.Align.FILL,
            valign=Gtk.Align.CENTER,
        )
        self.task_manager = task_manager

        self.add_css_class("thin-progress-bar")
        apply_css(
            """
            progressbar.thin-progress-bar {
                min-height: 5px;
                transition: opacity 0.25s;
            }
            """
        )

        self.task_manager.tasks_updated.connect(self._on_tasks_updated)
        self.set_opacity(0)  # Start faded out

    def _on_tasks_updated(self, sender, tasks, progress):
        """
        Updates the progress bar's fraction and visibility based on the
        state of the TaskManager.
        """
        has_tasks = bool(tasks)
        self.set_opacity(1 if has_tasks else 0)

        if has_tasks:
            self.set_fraction(progress)

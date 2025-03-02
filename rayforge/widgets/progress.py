from gi.repository import Gtk


class TaskProgressBar(Gtk.Box):
    def __init__(self, task_manager):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self.task_manager = task_manager

        # Create the progress bar
        self.progress_bar = Gtk.ProgressBar()
        self.progress_bar.set_hexpand(True)
        self.progress_bar.set_halign(Gtk.Align.FILL)
        self.progress_bar.set_valign(Gtk.Align.CENTER)

        # Create an overlay to place the text on top of the progress bar
        self.overlay = Gtk.Overlay()
        self.overlay.set_child(self.progress_bar)

        # Create the label for text
        self.label = Gtk.Label()
        self.label.set_halign(Gtk.Align.CENTER)
        self.label.set_valign(Gtk.Align.CENTER)
        self.label.set_ellipsize(3)  # (3 = END)

        # Add the label as an overlay on top of the progress bar
        self.overlay.add_overlay(self.label)

        # Add the overlay to the box
        self.append(self.overlay)

        # Connect to TaskManager signals
        self._connect_signals()
        self.set_opacity(0)  # Initially hidden

    def _connect_signals(self):
        """Connect to TaskManager signals."""
        self.task_manager.overall_progress_changed.connect(
            self._on_overall_progress_changed
        )
        self.task_manager.running_tasks_changed.connect(
            self._on_running_tasks_changed
        )

    def _on_overall_progress_changed(self, task_mgr, progress):
        """Update the progress bar."""
        self.progress_bar.set_fraction(progress)

    def _on_running_tasks_changed(self, sender, tasks):
        """
        Update the status text with the oldest task's
        status and a count of others.
        """
        if not tasks:
            # self.label.set_text("No tasks running")
            self.set_opacity(0)  # Hide when no tasks are running
            return

        # Show the progress bar when tasks are running
        self.set_opacity(1)

        # Find the oldest task (first in the list)
        oldest_task = tasks[0]
        status_text = oldest_task.get_status()

        # Add (+N more) if there are additional tasks
        if len(tasks) > 1:
            status_text += f" (+{len(tasks) - 1} more)"

        # Update the label text
        # TODO: self.label.set_text(status_text)

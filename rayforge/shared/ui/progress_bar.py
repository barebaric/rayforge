from gi.repository import Gtk, Gdk  # type: ignore


class ProgressBar(Gtk.Box):
    """
    A two-row progress widget.

    The top row is a status bar (self.status_box) that contains a message
    label. Users can append their own widgets to self.status_box.

    The bottom row contains a thin progress bar that shows the overall
    progress of all running tasks.
    """

    def __init__(self, task_manager):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.task_manager = task_manager

        # --- Top Row: Status Box for Label and other widgets ---
        # This box is made public so users can add their own widgets to it.
        self.status_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=3
        )

        # Create the label for the status message
        self.label = Gtk.Label(
            halign=Gtk.Align.START,
            valign=Gtk.Align.CENTER,
            hexpand=False,
            ellipsize=3,  # (3 = Pango.EllipsizeMode.END)
        )
        self.status_box.append(self.label)
        self.append(self.status_box)

        # A spacer widget that will expand and push all subsequent
        # widgets to the right, ensuring they are always right-aligned.
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        self.status_box.append(spacer)

        # --- Bottom Row: Progress Bar ---
        self.progress_bar = Gtk.ProgressBar(
            hexpand=True,
            halign=Gtk.Align.FILL,
            valign=Gtk.Align.CENTER,
        )
        self.progress_bar.add_css_class("thin-progress-bar")
        self.append(self.progress_bar)

        self._apply_css()
        self._connect_signals()

        self.label.set_visible(False)
        # CHANGED: Use opacity for the progress bar to reserve its space.
        self.progress_bar.set_opacity(0)

    def _apply_css(self):
        """Applies custom CSS to style the widget."""
        css_provider = Gtk.CssProvider()
        css = """
        progressbar.thin-progress-bar {
            min-height: 5px;
            /* Add a transition for a smooth fade in/out effect */
            transition: opacity 0.25s;
        }
        """
        css_provider.load_from_string(css)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _connect_signals(self):
        """Connect to the single, consolidated TaskManager signal."""
        self.task_manager.tasks_updated.connect(self._on_tasks_updated)

    def _on_tasks_updated(self, sender, tasks, progress):
        """
        Update the progress bar and status text from a single event.
        """
        has_tasks = bool(tasks)

        # Show/hide the text label
        self.label.set_visible(has_tasks)
        # CHANGED: Fade the progress bar in or out by changing its opacity.
        self.progress_bar.set_opacity(1 if has_tasks else 0)

        if not has_tasks:
            return

        self.progress_bar.set_fraction(progress)

        # Find the oldest task (first in the list)
        oldest_task = tasks[0]
        message = oldest_task.get_message()
        status_text = message if message is not None else ""

        # Add (+N more) if there are additional tasks
        if status_text and len(tasks) > 1:
            status_text += _(" (+{tasks} more)").format(tasks=len(tasks) - 1)
        elif len(tasks) > 1:
            status_text = _("{tasks} tasks").format(tasks=len(tasks))

        # Update the label text
        self.label.set_text(status_text)

"""Window shown while a remote client waits for its server."""

from gettext import gettext as _

from gi.repository import Adw, Gtk


class ClientWaitWindow(Adw.ApplicationWindow):
    """Tells the user the app is waiting for a rayforge-server."""

    def __init__(self, app, host: str, port: int, **kwargs):
        super().__init__(application=app, **kwargs)
        self.set_title(_("Waiting for server"))
        self.set_default_size(420, 140)

        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            valign=Gtk.Align.CENTER,
        )
        box.set_margin_top(24)
        box.set_margin_bottom(24)
        box.set_margin_start(24)
        box.set_margin_end(24)

        spinner = Gtk.Spinner()
        spinner.start()
        box.append(spinner)

        label = Gtk.Label(
            label=_(
                "Waiting for the rayforge-server at {host}:{port}…"
            ).format(host=host, port=port),
            wrap=True,
            justify=Gtk.Justification.CENTER,
        )
        box.append(label)

        self.set_content(box)

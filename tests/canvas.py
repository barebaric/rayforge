# flake8: noqa: E402
import gi
import logging
import gettext
from pathlib import Path
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # type: ignore

base_path = Path(__file__).parent
gettext.install("canvas", base_path / "rayforge" / "locale")
logging.basicConfig(level=logging.DEBUG)


from rayforge.workbench.canvas import Canvas, CanvasElement


class CanvasApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.CanvasApp")

    def do_activate(self):
        win = Gtk.ApplicationWindow(application=self)
        win.set_default_size(800, 800)

        canvas = Canvas()
        win.set_child(canvas)

        elem = CanvasElement(100, 120, 100, 100, background=(0.5, 1, 0.5, 1))
        canvas.add(elem)

        group = CanvasElement(150, 50, 400, 300, background=(0, 1, 1, 1))
        group.add(
            CanvasElement(
                50, 50, 200, 150, background=(0, 0, 1, 1), selectable=False
            )
        )
        # Buffered element to test threaded updates
        group.add(
            CanvasElement(
                100, 100, 150, 150, background=(0, 1, 0, 1), buffered=True
            )
        )
        group.add(
            CanvasElement(50, 100, 250, 250, background=(1, 0, 1, 1))
        )
        canvas.add(group)
        win.present()

app = CanvasApp()
app.run([])

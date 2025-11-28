#!/usr/bin/python
# flake8: noqa: E402
import gi
import logging
import gettext
from pathlib import Path
from typing import Optional

# -- Setup Logging --
logging.basicConfig(
    level=logging.DEBUG, format="[%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sketcherapp")

base_path = Path(__file__).parent
gettext.install("canvas", base_path / "rayforge" / "locale")

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Adw

from rayforge.workbench.sketcher.studio import SketchStudio
from rayforge.core.sketcher import Sketch


class SketcherApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id="com.example.SketcherApp")
        self.studio: Optional[SketchStudio] = None
        self.window: Optional[Gtk.ApplicationWindow] = None

    def do_activate(self):
        self.window = Gtk.ApplicationWindow(application=self)
        self.window.set_default_size(1200, 800)

        # Initialize Studio
        self.studio = SketchStudio(parent_window=self.window)
        self.window.set_child(self.studio)

        # Connect signals for testing
        self.studio.finished.connect(self.on_studio_finished)
        self.studio.cancelled.connect(self.on_studio_cancelled)

        # Setup initial Element via Studio
        self.add_initial_sketch()

        self.window.present()

    def add_initial_sketch(self):
        """Creates and adds the first sketch with demo geometry."""
        if not self.studio:
            return

        sketch = Sketch()
        origin_id = sketch.origin_id

        # Build a floating shape first
        p1 = sketch.add_point(0, 0)
        p2 = sketch.add_point(200, 0)
        p3 = sketch.add_point(100, 150)

        # Then, constrain it to the origin
        sketch.constrain_coincident(p1, origin_id)

        sketch.add_line(p1, p2)
        sketch.add_line(p2, p3)
        sketch.add_line(p3, p1)

        sketch.constrain_distance(p1, p2, 200.0)
        sketch.constrain_horizontal(p1, p2)

        # Add a construction line for reference
        mid_p = sketch.add_point(100, 0)
        if sketch.registry.entities:
            line_id = sketch.registry.entities[0].id
            sketch.constrain_point_on_line(mid_p, line_id)

        sketch.add_line(mid_p, p3, construction=True)
        sketch.constrain_vertical(mid_p, p3)

        sketch.solve()

        # Load the sketch into the studio
        self.studio.set_sketch(sketch)

    def on_studio_finished(self, sender, sketch):
        print(
            f"Studio Finished! Sketch has {len(sketch.registry.entities)} entities."
        )

        if not self.studio:
            return

        # Reset with a fresh sketch to demonstrate lifecycle
        new_sketch = Sketch()
        print("Resetting studio with empty sketch...")
        self.studio.set_sketch(new_sketch)

    def on_studio_cancelled(self, sender):
        print("Studio Cancelled!")

        if not self.studio:
            return

        # Reset with a fresh sketch to demonstrate lifecycle
        new_sketch = Sketch()
        print("Resetting studio with empty sketch...")
        self.studio.set_sketch(new_sketch)


if __name__ == "__main__":
    app = SketcherApp()
    app.run([])

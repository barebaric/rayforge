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

from rayforge.workbench.sketcher.sketchcanvas import SketchCanvas
from rayforge.workbench.sketcher.sketchelement import SketchElement


class SketcherApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id="com.example.SketcherApp")
        self.sketch_elem: Optional[SketchElement] = None
        self.canvas: Optional[SketchCanvas] = None
        self.window: Optional[Gtk.ApplicationWindow] = None

    def do_activate(self):
        self.window = Gtk.ApplicationWindow(application=self)
        self.window.set_default_size(1200, 800)

        # Main layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.set_child(vbox)

        # Header with button and label
        header_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=10
        )
        header_box.set_margin_top(10)
        header_box.set_margin_bottom(10)
        header_box.set_margin_start(10)
        header_box.set_margin_end(10)
        vbox.append(header_box)

        reset_button = Gtk.Button(label="Reset Sketch")
        reset_button.connect("clicked", self.on_reset_clicked)
        header_box.append(reset_button)

        header_label = Gtk.Label(
            label="Right-click for Tools/Constraints | Scroll to Zoom | Middle-click to Pan"
        )
        header_label.add_css_class("dim-label")
        header_box.append(header_label)

        # Canvas
        self.canvas = SketchCanvas(parent_window=self.window)
        self.canvas.set_vexpand(True)
        vbox.append(self.canvas)

        # Get the element that the canvas created for itself
        self.sketch_elem = self.canvas.sketch_element

        # Setup initial Element
        self.add_initial_sketch()

        self.window.present()

    def add_initial_sketch(self):
        """Creates and adds the first sketch with demo geometry."""
        if not self.canvas or not self.sketch_elem:
            return

        # Initialize demo geometry on the canvas's internal sketch element
        sketch = self.sketch_elem.sketch
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
        self.sketch_elem.update_bounds_from_sketch()

        # Center the new element on the canvas
        canvas_w, canvas_h = self.canvas.get_size_mm()
        elem_w, elem_h = self.sketch_elem.width, self.sketch_elem.height
        self.sketch_elem.set_pos(
            (canvas_w - elem_w) / 2.0, (canvas_h - elem_h) / 2.0
        )

        # Center the view on the geometry
        self.canvas.reset_view()

    def on_reset_clicked(self, button: Gtk.Button):
        """Clears the sketch and resets the view."""
        if not self.canvas or not self.sketch_elem:
            return

        # Tell the canvas to perform a full, safe reset
        new_sketch = self.canvas.reset_sketch()

        # The new sketch is created empty; update its bounds to get a size.
        new_sketch.update_bounds_from_sketch()

        # Center the new element on the canvas before resetting the view
        canvas_w, canvas_h = self.canvas.get_size_mm()
        elem_w, elem_h = new_sketch.width, new_sketch.height
        new_sketch.set_pos(
            (canvas_w - elem_w) / 2.0, (canvas_h - elem_h) / 2.0
        )

        # Update the app's reference and reset the view to center it
        self.sketch_elem = new_sketch
        self.canvas.reset_view()


if __name__ == "__main__":
    app = SketcherApp()
    app.run([])

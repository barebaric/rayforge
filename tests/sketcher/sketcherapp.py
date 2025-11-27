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
from gi.repository import Gtk

from rayforge.workbench.sketcher.sketchcanvas import SketchCanvas
from rayforge.workbench.sketcher import SketchElement


class SketcherApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.SketcherApp")
        self.sketch_elem: Optional[SketchElement] = None

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
            label="Right-click on canvas for Tools & Constraints"
        )
        header_label.add_css_class("dim-label")
        header_box.append(header_label)

        # Canvas
        self.canvas = SketchCanvas(parent_window=self.window)
        self.canvas.set_vexpand(True)
        vbox.append(self.canvas)

        # Setup initial Element
        self.add_initial_sketch()

        self.window.present()

    def add_initial_sketch(self):
        """Creates and adds the first sketch with demo geometry."""
        self.sketch_elem = SketchElement(x=100, y=100, width=1, height=1)

        # Initialize demo geometry
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

        # Connect the constraint edit signal
        self.sketch_elem.constraint_edit_requested.connect(
            self.on_edit_constraint_val
        )
        self.canvas.add(self.sketch_elem)

        # Set the active edit context so the pie menu has a target
        self.canvas.enter_edit_mode(self.sketch_elem)

    def on_reset_clicked(self, button: Gtk.Button):
        """Removes the old sketch and creates a new, empty one."""
        if not self.canvas or not self.sketch_elem:
            return

        # Cleanly exit edit mode for the old sketch
        if self.canvas.edit_context is self.sketch_elem:
            self.canvas.leave_edit_mode()

        # Remove the old element from the canvas
        self.canvas.remove(self.sketch_elem)

        # Create a new, empty sketch element at the center of the canvas
        canvas_width, canvas_height = self.canvas.size()
        new_sketch = SketchElement(x=canvas_width / 2, y=canvas_height / 2)

        # The new sketch already has an origin. Just update its bounds.
        new_sketch.update_bounds_from_sketch()

        # Re-connect signals for the new sketch
        new_sketch.constraint_edit_requested.connect(
            self.on_edit_constraint_val
        )

        # Add to canvas and set as the active context for interaction
        self.canvas.add(new_sketch)
        self.canvas.enter_edit_mode(new_sketch)

        # Update the app's reference to the current sketch element
        self.sketch_elem = new_sketch

    def on_edit_constraint_val(self, sender, constraint):
        """Opens a dialog to edit a constraint value."""
        dialog = Gtk.Window(
            transient_for=self.window, modal=True, title="Edit Constraint"
        )
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(10)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_bottom(10)
        dialog.set_child(box)

        entry = Gtk.Entry()
        entry.set_text(str(constraint.value))
        box.append(entry)

        def on_confirm(*args):
            try:
                val = float(entry.get_text())
                constraint.value = val
                if self.sketch_elem:
                    self.sketch_elem.sketch.solve()
                    self.sketch_elem.update_bounds_from_sketch()
                    if self.canvas:
                        self.canvas.queue_draw()
                dialog.close()
            except ValueError:
                pass

        entry.connect("activate", on_confirm)
        dialog.present()


if __name__ == "__main__":
    app = SketcherApp()
    app.run([])

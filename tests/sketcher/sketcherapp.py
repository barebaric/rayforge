# flake8: noqa: E402
import gi
import logging
import gettext
from pathlib import Path
from typing import cast, Any

base_path = Path(__file__).parent
gettext.install("canvas", base_path / "rayforge" / "locale")
logging.basicConfig(level=logging.DEBUG)

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk

from rayforge.workbench.canvas import Canvas
from rayforge.workbench.sketcher import SketchElement


class SketchCanvas(Canvas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._edit_key_ctrl = Gtk.EventControllerKey.new()
        self._edit_key_ctrl.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        self._edit_key_ctrl.connect("key-pressed", self.on_edit_key_pressed)
        self.add_controller(self._edit_key_ctrl)

    def on_edit_key_pressed(self, controller, keyval, keycode, state):
        if self.edit_context and isinstance(self.edit_context, SketchElement):
            if keyval == Gdk.KEY_Delete:
                self.edit_context.delete_selection()
                return True
        return False

    def on_button_press(self, gesture, n_press: int, x: float, y: float):
        if self.edit_context:
            self.grab_focus()
            self._was_dragging = False
            world_x, world_y = self._get_world_coords(x, y)

            if isinstance(self.edit_context, SketchElement):
                ctx = cast(Any, self.edit_context)
                handled = ctx.handle_edit_press(world_x, world_y, n_press)
            else:
                handled = self.edit_context.handle_edit_press(world_x, world_y)

            if not handled and self._hovered_elem is None:
                self.leave_edit_mode()
            self.queue_draw()
            return

        super().on_button_press(gesture, n_press, x, y)


class SketcherApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.SketcherApp")
        self.sketch_elem = None

    def do_activate(self):
        self.window = Gtk.ApplicationWindow(application=self)
        self.window.set_default_size(1200, 800)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.set_child(vbox)

        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar.set_margin_top(6)
        toolbar.set_margin_start(6)
        vbox.append(toolbar)

        # Tools Group
        btn_select = Gtk.ToggleButton(label="Select")
        btn_select.set_active(True)
        btn_select.connect("toggled", self.on_tool, "select")
        toolbar.append(btn_select)

        btn_line = Gtk.ToggleButton(label="Line")
        btn_line.set_group(btn_select)
        btn_line.connect("toggled", self.on_tool, "line")
        toolbar.append(btn_line)

        btn_arc = Gtk.ToggleButton(label="Arc")
        btn_arc.set_group(btn_select)
        btn_arc.connect("toggled", self.on_tool, "arc")
        toolbar.append(btn_arc)

        toolbar.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))

        # Constraints Group
        btn_dist = Gtk.Button(label="Dist")
        btn_dist.connect(
            "clicked",
            lambda x: self.sketch_elem.add_distance_constraint()
            if self.sketch_elem
            else None,
        )
        toolbar.append(btn_dist)

        btn_rad = Gtk.Button(label="Rad")
        btn_rad.connect(
            "clicked",
            lambda x: self.sketch_elem.add_radius_constraint()
            if self.sketch_elem
            else None,
        )
        toolbar.append(btn_rad)

        btn_horiz = Gtk.Button(label="Horiz")
        btn_horiz.connect(
            "clicked",
            lambda x: self.sketch_elem.add_horizontal_constraint()
            if self.sketch_elem
            else None,
        )
        toolbar.append(btn_horiz)

        btn_vert = Gtk.Button(label="Vert")
        btn_vert.connect(
            "clicked",
            lambda x: self.sketch_elem.add_vertical_constraint()
            if self.sketch_elem
            else None,
        )
        toolbar.append(btn_vert)

        btn_align = Gtk.Button(label="Constrain")
        btn_align.connect(
            "clicked",
            lambda x: self.sketch_elem.add_alignment_constraint()
            if self.sketch_elem
            else None,
        )
        toolbar.append(btn_align)

        btn_perp = Gtk.Button(label="Perp")
        btn_perp.connect(
            "clicked",
            lambda x: self.sketch_elem.add_perpendicular()
            if self.sketch_elem
            else None,
        )
        toolbar.append(btn_perp)

        btn_tan = Gtk.Button(label="Tan")
        btn_tan.connect(
            "clicked",
            lambda x: self.sketch_elem.add_tangent()
            if self.sketch_elem
            else None,
        )
        toolbar.append(btn_tan)

        # Geometry Toggles
        toolbar.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))

        btn_constr = Gtk.Button(label="Toggle Construction Geometry")
        btn_constr.connect(
            "clicked",
            lambda x: self.sketch_elem.toggle_construction_on_selection()
            if self.sketch_elem
            else None,
        )
        toolbar.append(btn_constr)

        # Canvas
        self.canvas = SketchCanvas()
        self.canvas.set_vexpand(True)
        vbox.append(self.canvas)

        # Setup Element
        self.sketch_elem = SketchElement(x=100, y=100, width=1, height=1)

        # Initialize demo geometry (moved from SketchElement)
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
        # Constraint to line p1-p2 (assumed to be the first entity added)
        if sketch.registry.entities:
            line_id = sketch.registry.entities[0].id
            sketch.constrain_point_on_line(mid_p, line_id)

        sketch.add_line(mid_p, p3, construction=True)
        sketch.constrain_vertical(mid_p, p3)

        sketch.solve()
        self.sketch_elem.update_bounds_from_sketch()

        self.sketch_elem.constraint_edit_requested.connect(
            self.on_edit_constraint_val
        )
        self.canvas.add(self.sketch_elem)

        self.window.present()

    def on_tool(self, btn, name):
        if btn.get_active() and self.sketch_elem:
            self.sketch_elem.set_tool(name)

    def on_edit_constraint_val(self, sender, constraint):
        dialog = Gtk.Window(
            transient_for=self.window, modal=True, title="Edit Constraint"
        )
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(10)
        box.set_margin_start(10)
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

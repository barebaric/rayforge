# flake8: noqa: E402
import gi
import logging
import gettext
from pathlib import Path
from typing import cast, Any

# -- Setup Logging --
logging.basicConfig(
    level=logging.DEBUG, format="[%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sketcherapp")

base_path = Path(__file__).parent
gettext.install("canvas", base_path / "rayforge" / "locale")

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk

from rayforge.workbench.canvas import Canvas
from rayforge.workbench.sketcher import SketchElement
from rayforge.workbench.sketcher.piemenu import SketchPieMenu


class SketchCanvas(Canvas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1. Edit Key Controller (Delete key)
        self._edit_key_ctrl = Gtk.EventControllerKey.new()
        self._edit_key_ctrl.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        self._edit_key_ctrl.connect("key-pressed", self.on_edit_key_pressed)
        self.add_controller(self._edit_key_ctrl)

        # 2. Pie Menu Setup
        self.pie_menu = SketchPieMenu(self)

        # Connect signals
        self.pie_menu.tool_selected.connect(self.on_tool_selected)
        self.pie_menu.constraint_selected.connect(self.on_constraint_selected)
        self.pie_menu.action_triggered.connect(self.on_action_triggered)

        # 3. Right Click Gesture for Pie Menu
        right_click = Gtk.GestureClick()
        right_click.set_button(3)  # Right Mouse Button
        right_click.connect("pressed", self.on_right_click)
        self.add_controller(right_click)

    def on_right_click(self, gesture, n_press, x, y):
        """Open the pie menu at the cursor location."""
        logger.info(f"Opening Pie Menu at {x}, {y}")
        self.pie_menu.popup_at_location(x, y)
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)

    def on_tool_selected(self, sender, tool):
        logger.info(f"Tool activated: {tool}")
        if self.edit_context and isinstance(self.edit_context, SketchElement):
            self.edit_context.set_tool(tool)

    def on_constraint_selected(self, sender, constraint_type):
        logger.info(f"Constraint activated: {constraint_type}")
        ctx = self.edit_context
        if not (ctx and isinstance(ctx, SketchElement)):
            return

        if constraint_type == "dist":
            ctx.add_distance_constraint()
        elif constraint_type == "horiz":
            ctx.add_horizontal_constraint()
        elif constraint_type == "vert":
            ctx.add_vertical_constraint()

    def on_action_triggered(self, sender, action):
        logger.info(f"Action activated: {action}")
        ctx = self.edit_context
        if not (ctx and isinstance(ctx, SketchElement)):
            return

        if action == "construction":
            ctx.toggle_construction_on_selection()
        elif action == "delete":
            ctx.delete_selection()

    def on_edit_key_pressed(self, controller, keyval, keycode, state):
        if self.edit_context and isinstance(self.edit_context, SketchElement):
            if keyval == Gdk.KEY_Delete:
                self.edit_context.delete_selection()
                return True
        return False

    def on_button_press(self, gesture, n_press: int, x: float, y: float):
        # Ignore right clicks for the standard canvas logic
        if gesture.get_current_button() == 3:
            return

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

        # Main layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.set_child(vbox)

        # Info Header
        header = Gtk.Label(
            label="Right-click on canvas for Tools & Constraints"
        )
        header.set_margin_top(10)
        header.set_margin_bottom(10)
        header.add_css_class("dim-label")
        vbox.append(header)

        # Canvas
        self.canvas = SketchCanvas()
        self.canvas.set_vexpand(True)
        vbox.append(self.canvas)

        # Setup Element
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
        self.canvas.edit_context = self.sketch_elem

        self.window.present()

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

#!/usr/bin/python
# flake8: noqa: E402
import gi
import logging
import gettext
import json
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
from gi.repository import Gtk, Adw, Gio, GLib

from rayforge.workbench.sketcher.studio import SketchStudio
from rayforge.core.sketcher import Sketch


class SketcherApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id="com.example.SketcherApp")
        self.studio: Optional[SketchStudio] = None
        self.window: Optional[Gtk.ApplicationWindow] = None

        # Create and register the "quit" action
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", self.on_quit_action)
        self.add_action(quit_action)

        # Now, bind the accelerator to the action we just created.
        self.set_accels_for_action("app.quit", ["<Control>q"])

    def on_quit_action(self, action, param):
        """Handler for the 'quit' action."""
        self.quit()

    def do_activate(self):
        self.window = Gtk.ApplicationWindow(application=self)
        self.window.set_default_size(1200, 800)

        # Initialize Studio
        self.studio = SketchStudio(parent_window=self.window)
        self.window.set_child(self.studio)

        # --- Add Open/Save Buttons ---
        btn_open = Gtk.Button(label=_("Open..."))
        btn_open.set_tooltip_text(_("Open Sketch from File"))
        btn_open.connect("clicked", self.on_open_clicked)
        # Insert after the Cancel button
        self.studio.session_bar.insert_child_after(
            btn_open, self.studio.btn_cancel
        )

        btn_save = Gtk.Button(label=_("Save..."))
        btn_save.set_tooltip_text(_("Save Sketch to File"))
        btn_save.connect("clicked", self.on_save_clicked)
        # Insert after the Open button
        self.studio.session_bar.insert_child_after(btn_save, btn_open)
        # --- End of added buttons ---

        # Connect signals for testing
        self.studio.finished.connect(self.on_studio_finished)
        self.studio.cancelled.connect(self.on_studio_cancelled)

        # Setup initial Element via Studio
        self.add_initial_sketch()

        self.window.present()

        # Ensure the canvas has focus to receive key events immediately.
        if self.studio and self.studio.canvas:
            self.studio.canvas.grab_focus()

    def on_open_clicked(self, widget):
        """Handles opening a sketch from a file using Gtk.FileDialog."""
        dialog = Gtk.FileDialog.new()
        dialog.set_title(_("Open Sketch"))

        filter_rfs = Gtk.FileFilter.new()
        filter_rfs.set_name(_("RayForge Sketch Files"))
        filter_rfs.add_pattern("*.rfs")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(filter_rfs)

        dialog.set_filters(filters)
        dialog.set_default_filter(filter_rfs)

        dialog.open(self.window, None, self._on_open_dialog_finish)

    def _on_open_dialog_finish(self, dialog, result):
        try:
            file = dialog.open_finish(result)
            if file:
                path = file.get_path()
                if path and self.studio:
                    logger.info(f"Loading sketch from: {path}")
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        new_sketch = Sketch.from_dict(data)
                        self.studio.set_sketch(new_sketch)
                    except Exception as e:
                        logger.error(
                            f"Failed to load sketch file '{path}': {e}"
                        )
        except GLib.Error as e:
            # This catches user cancellation
            logger.debug(f"File open dialog cancelled: {e.message}")

    def on_save_clicked(self, widget):
        """Handles saving the current sketch to a file using Gtk.FileDialog."""
        if not self.studio or not self.studio.canvas.sketch_element:
            return

        dialog = Gtk.FileDialog.new()
        dialog.set_title(_("Save Sketch"))
        dialog.set_initial_name("sketch.rfs")

        filter_rfs = Gtk.FileFilter.new()
        filter_rfs.set_name(_("RayForge Sketch Files"))
        filter_rfs.add_pattern("*.rfs")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(filter_rfs)

        dialog.set_filters(filters)
        dialog.set_default_filter(filter_rfs)

        dialog.save(self.window, None, self._on_save_dialog_finish)

    def _on_save_dialog_finish(self, dialog, result):
        try:
            file = dialog.save_finish(result)
            if file:
                path = file.get_path()
                if path and self.studio and self.studio.canvas.sketch_element:
                    if not path.lower().endswith(".rfs"):
                        path += ".rfs"
                    logger.info(f"Saving sketch to: {path}")
                    try:
                        current_sketch = (
                            self.studio.canvas.sketch_element.sketch
                        )
                        data = current_sketch.to_dict()
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                    except Exception as e:
                        logger.error(f"Failed to save sketch to '{path}': {e}")
        except GLib.Error as e:
            logger.debug(f"File save dialog cancelled: {e.message}")

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

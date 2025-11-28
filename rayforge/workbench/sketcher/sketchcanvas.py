import logging
from typing import Optional, cast
from gi.repository import Gtk, Gdk, Adw
from ..canvas import WorldSurface
from .sketchelement import SketchElement
from .editor import SketchEditor
from .sketch_cmd import ModifyConstraintValueCommand
from rayforge.core.sketcher.constraints import Constraint

logger = logging.getLogger(__name__)


class SketchCanvas(WorldSurface):
    def __init__(self, parent_window: Gtk.Window, **kwargs):
        # A Sketcher doesn't have a fixed machine size. We initialize the
        # WorldSurface with a large default area to provide an "infinite" feel.
        super().__init__(width_mm=2000, height_mm=2000, **kwargs)
        self.parent_window = parent_window

        # This will hold a reference to the active dialog to prevent it from
        # being garbage-collected prematurely.
        self._active_dialog: Optional[Adw.MessageDialog] = None

        # The SketchCanvas owns a SketchEditor to manage the session.
        self.sketch_editor = SketchEditor(self.parent_window)

        # It creates a single, primary sketch element that is always active.
        self.sketch_element = SketchElement()
        self.sketch_element.constraint_edit_requested.connect(
            self._on_constraint_edit_requested
        )
        self.root.add(self.sketch_element)

        # Permanently enter edit mode on the primary sketch element.
        self.edit_context = self.sketch_element
        self.sketch_editor.activate(self.sketch_element)

    def reset_sketch(self) -> SketchElement:
        """
        Removes the current sketch element and replaces it with a new, empty
        one, ensuring all internal references are updated correctly.

        :return: The new SketchElement instance.
        """
        old_sketch = self.sketch_element

        # Deactivate the editor from the old element
        self.sketch_editor.deactivate()
        # Clear the edit context
        self.edit_context = None
        # Remove the old element from the scene graph
        if old_sketch:
            old_sketch.remove()

        # Create and configure the new element
        new_sketch = SketchElement()
        new_sketch.constraint_edit_requested.connect(
            self._on_constraint_edit_requested
        )
        self.root.add(new_sketch)

        # Update all internal references
        self.sketch_element = new_sketch
        self.edit_context = new_sketch
        self.sketch_editor.activate(new_sketch)

        return new_sketch

    def reset_view(self) -> None:
        """
        Overrides the base implementation to center the view on the geometric
        center of the sketch's contents.
        """
        logger.debug("Resetting SketchCanvas view to center sketch geometry.")
        if not self.sketch_element:
            super().reset_view()
            return

        sketch = self.sketch_element.sketch
        min_x, max_x, min_y, max_y = 0.0, 0.0, 0.0, 0.0
        has_bounds = False

        # 1. Calculate bounding box of all sketch geometry in Model
        # coordinates.
        geometry = sketch.to_geometry()
        if not geometry.is_empty():
            min_x, min_y, max_x, max_y = geometry.rect()
            has_bounds = True
        elif sketch.registry.points:
            # This case handles sketches with only points.
            xs = [p.x for p in sketch.registry.points]
            ys = [p.y for p in sketch.registry.points]
            if xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                has_bounds = True

        # If there is a bounding box, calculate its center.
        # Otherwise, the target is the model origin (0,0).
        if has_bounds:
            model_center_x = (min_x + max_x) / 2.0
            model_center_y = (min_y + max_y) / 2.0
        else:
            model_center_x = 0.0
            model_center_y = 0.0

        # 2. Transform the model center to world coordinates.
        model_to_world = (
            self.sketch_element.get_world_transform()
            @ self.sketch_element.content_transform
        )
        target_center_x, target_center_y = model_to_world.transform_point(
            (model_center_x, model_center_y)
        )

        # 3. Calculate pan to move this point to the view center.
        # The pan value is the coordinate of the world's top-left corner that
        # will be displayed in the view's top-left.
        pan_x = target_center_x - (self.width_mm / 2.0)
        pan_y = target_center_y - (self.height_mm / 2.0)

        self.set_pan(pan_x, pan_y)
        self.set_zoom(1.0)

    def _on_constraint_edit_requested(self, sender, constraint: Constraint):
        """
        Opens a dialog to edit the value of a constraint. This is triggered
        by a double-click on a constraint label.
        """
        # If a dialog is already open, do nothing.
        if self._active_dialog:
            return

        # Ensure the constraint has a 'value' attribute we can edit.
        if not hasattr(constraint, "value"):
            logger.warning(
                "Constraint edit requested for a constraint with no "
                f"'value' attribute: {type(constraint).__name__}"
            )
            return

        entry = Gtk.Entry()
        entry.set_text(str(float(getattr(constraint, "value", 0))))
        entry.set_activates_default(True)

        dialog = Adw.MessageDialog(
            transient_for=self.parent_window,
            modal=True,
            destroy_with_parent=True,
            heading="Edit Constraint",
            body=f"Enter the new value for the {type(constraint).__name__}.",
        )
        dialog.set_extra_child(entry)
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("ok", "OK")
        dialog.set_default_response("ok")
        dialog.set_close_response("cancel")

        def on_response(source, response_id):
            if response_id == "ok":
                try:
                    new_value = float(entry.get_text())
                    if new_value > 0:
                        cmd = ModifyConstraintValueCommand(
                            element=self.sketch_element,
                            constraint=constraint,
                            new_value=new_value,
                        )
                        self.sketch_editor.history_manager.execute(cmd)
                    else:
                        logger.warning("Constraint value must be positive.")
                except (ValueError, TypeError):
                    logger.warning("Invalid input for constraint value.")
            # Clear the reference to allow the dialog to be destroyed
            self._active_dialog = None

        # Store a reference to the dialog to prevent garbage collection
        self._active_dialog = dialog
        self._active_dialog.connect("response", on_response)
        self._active_dialog.present()

    def on_right_click_pressed(
        self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float
    ):
        """
        Overrides the base class to unconditionally delegate to the editor.
        """
        self.sketch_editor.handle_right_click(gesture, n_press, x, y)

    def on_key_pressed(
        self,
        controller: Gtk.EventControllerKey,
        keyval: int,
        keycode: int,
        state: Gdk.ModifierType,
    ) -> bool:
        """
        Overrides base to delegate sketcher-specific key presses to the
        editor before falling back to the WorldSurface's handlers.
        """
        # First, let the sketch editor handle its keys (Undo/Redo, Delete)
        if self.sketch_editor.handle_key_press(keyval, keycode, state):
            return True

        # Then, let the base class handle its keys (e.g., '1' for reset view)
        if super().on_key_pressed(controller, keyval, keycode, state):
            return True

        return False

    def update_sketch_cursor(self):
        """Forces an update of the cursor based on the editor's state."""
        if self.sketch_editor:
            cursor = self.sketch_editor.get_current_cursor()
            self.set_cursor(cursor)

    def on_motion(self, gesture: Gtk.Gesture, x: float, y: float):
        """
        Overrides the base canvas motion handler to implement sketcher-
        specific cursor logic, bypassing the default handle-based system.
        """
        # Store raw pixel coordinates for other uses (like scroll-to-zoom)
        self._mouse_pos = (x, y)

        world_x, world_y = self._get_world_coords(x, y)

        # Let the active tool update its hover state (e.g., for snapping)
        if self.sketch_element:
            self.sketch_element.on_hover_motion(world_x, world_y)

        # Set the cursor based on the complete state from the editor
        self.update_sketch_cursor()

    def on_motion_leave(self, controller: Gtk.EventControllerMotion):
        """Resets hover state and cursor when the mouse leaves the canvas."""
        super().on_motion_leave(controller)
        self.set_cursor(None)  # Reset to default cursor

    def on_button_press(
        self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float
    ):
        """
        Overrides the base Canvas handler to manage pie menu visibility,
        delegate to the active tool, and correctly handle gesture state for
        double-clicks.
        """
        # If the pie menu is visible, a left click should dismiss it.
        if (
            self.sketch_editor.pie_menu.is_visible()
            and gesture.get_current_button() != 3
        ):
            self.sketch_editor.pie_menu.popdown()
            gesture.set_state(Gtk.EventSequenceState.CLAIMED)
            return

        if gesture.get_current_button() == 3:
            return  # Already handled by the right-click gesture

        # Replicate the logic from the base Canvas.on_button_press but with
        # conditional gesture claiming.
        self.grab_focus()

        handled = False
        if self.edit_context:
            world_x, world_y = self._get_world_coords(x, y)
            # Delegate to the tool. The tool's return value determines if
            # the gesture sequence should be terminated.
            sketch_element = cast(SketchElement, self.edit_context)
            handled = sketch_element.handle_edit_press(
                world_x, world_y, n_press
            )

        # Only claim the gesture if the tool has fully handled the event
        # (e.g., a completed double-click). A single click should not
        # claim the gesture, allowing the second click to be detected.
        if handled:
            logger.debug(
                "Tool handled the press event, claiming gesture state."
            )
            gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        else:
            logger.debug("Tool did not handle press event, gesture continues.")

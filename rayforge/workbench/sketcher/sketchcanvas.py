import logging
from gi.repository import Gtk, Gdk
from ..worldsurface import WorldSurface
from .sketchelement import SketchElement
from .editor import SketchEditor

logger = logging.getLogger(__name__)


class SketchCanvas(WorldSurface):
    def __init__(self, parent_window: Gtk.Window, **kwargs):
        # A Sketcher doesn't have a fixed machine size. We initialize the
        # WorldSurface with a large default area to provide an "infinite" feel.
        super().__init__(width_mm=2000, height_mm=2000, **kwargs)
        self.parent_window = parent_window

        # The SketchCanvas owns a SketchEditor to manage the session.
        self.sketch_editor = SketchEditor(self.parent_window)

        # It creates a single, primary sketch element that is always active.
        self.sketch_element = SketchElement()
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

    def on_button_press(
        self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float
    ):
        """
        Overrides the base Canvas handler to manage pie menu visibility and
        delegate to the active tool.
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

        # Delegate to the standard Canvas implementation, which handles
        # focus and calls the edit_context's handlers.
        super().on_button_press(gesture, n_press, x, y)

import logging
from typing import Union, Optional, TYPE_CHECKING
from gi.repository import Gtk, Gdk
from ...core.sketcher.entities import Point, Entity
from ...core.sketcher.constraints import Constraint
from ...undo import HistoryManager
from .piemenu import SketchPieMenu

if TYPE_CHECKING:
    from .sketchelement import SketchElement


logger = logging.getLogger(__name__)


class SketchEditor:
    """
    The SketchEditor provides a controller for an interactive sketch editing
    session. It is not a widget, but rather a host that manages the UI
    (PieMenu), state (HistoryManager), and input delegation for a given
    SketchElement. It can be used by any canvas-like widget.
    """

    def __init__(self, parent_window: Gtk.Window):
        self.parent_window = parent_window
        self.sketch_element: Optional["SketchElement"] = None

        # The SketchEditor manages its own undo/redo history, separate from
        # the main document editor.
        self.history_manager = HistoryManager()

        # 1. Pie Menu Setup
        self.pie_menu = SketchPieMenu(self.parent_window)

        # Connect signals
        self.pie_menu.tool_selected.connect(self.on_tool_selected)
        self.pie_menu.constraint_selected.connect(self.on_constraint_selected)
        self.pie_menu.action_triggered.connect(self.on_action_triggered)
        self.pie_menu.right_clicked.connect(self.on_pie_menu_right_click)

    def activate(self, sketch_element: "SketchElement"):
        """Begins an editing session on the given SketchElement."""
        logger.debug(f"Activating SketchEditor for element {sketch_element}")
        self.sketch_element = sketch_element
        self.sketch_element.editor = self

    def deactivate(self):
        """Ends the current editing session."""
        logger.debug("Deactivating SketchEditor")
        if self.sketch_element:
            # Clean up any in-progress tool state
            self.sketch_element.current_tool.on_deactivate()
            self.sketch_element.editor = None
        self.sketch_element = None
        if self.pie_menu.is_visible():
            self.pie_menu.popdown()

    def on_pie_menu_right_click(self, sender, gesture, n_press, x, y):
        """
        Handles a right-click that happened on the PieMenu's drawing area.
        Translates coordinates and forwards to the main right-click handler.
        """
        if not self.sketch_element or not self.sketch_element.canvas:
            return

        child = self.pie_menu.get_child()
        if not child:
            return

        canvas_coords = child.translate_coordinates(
            self.sketch_element.canvas, x, y
        )
        if canvas_coords:
            canvas_x, canvas_y = canvas_coords
            self.handle_right_click(gesture, n_press, canvas_x, canvas_y)

    def handle_right_click(
        self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float
    ):
        """
        Opens the pie menu at the cursor location with resolved context.
        This is the primary entry point for right-click handling.
        """
        if not self.sketch_element or not self.sketch_element.canvas:
            return

        if self.pie_menu.is_visible():
            self.pie_menu.popdown()

        # Use the element's canvas to convert from widget to world coordinates
        world_x, world_y = self.sketch_element.canvas._get_world_coords(x, y)

        target: Optional[Union[Point, Entity, Constraint]] = None
        target_type: Optional[str] = None

        # Before showing the menu, we deactivate the current tool to clean
        # up any in-progress state.
        self.sketch_element.current_tool.on_deactivate()

        selection = self.sketch_element.selection
        selection_changed = False

        # 1. Hit Test
        hit_type, hit_obj = self.sketch_element.hittester.get_hit_data(
            world_x, world_y, self.sketch_element
        )
        target_type = hit_type

        # 2. Resolve Hit Object to Concrete Type AND Update Selection
        # If the clicked object is not already selected, select it.
        if hit_type == "point":
            assert isinstance(hit_obj, int)
            pid = hit_obj
            target = self.sketch_element.sketch.registry.get_point(pid)

            if pid not in selection.point_ids:
                selection.select_point(pid, is_multi=False)
                selection_changed = True

        elif hit_type == "junction":
            assert isinstance(hit_obj, int)
            pid = hit_obj
            target = self.sketch_element.sketch.registry.get_point(pid)

            if selection.junction_pid != pid:
                selection.select_junction(pid, is_multi=False)
                selection_changed = True

        elif hit_type == "entity":
            assert isinstance(hit_obj, Entity)
            entity = hit_obj
            target = entity

            if entity.id not in selection.entity_ids:
                selection.select_entity(entity, is_multi=False)
                selection_changed = True

        elif hit_type == "constraint":
            assert isinstance(hit_obj, int)
            idx = hit_obj
            if 0 <= idx < len(self.sketch_element.sketch.constraints):
                target = self.sketch_element.sketch.constraints[idx]

                if selection.constraint_idx != idx:
                    selection.select_constraint(idx, is_multi=False)
                    selection_changed = True

        elif hit_type is None:
            if (
                selection.point_ids
                or selection.entity_ids
                or selection.constraint_idx is not None
                or selection.junction_pid is not None
            ):
                selection.clear()
                selection_changed = True

        if selection_changed:
            self.sketch_element.mark_dirty()

        # 3. Pass Context (Sketch, Target, Type)
        self.pie_menu.set_context(self.sketch_element, target, target_type)

        win_coords = self.sketch_element.canvas.translate_coordinates(
            self.parent_window, x, y
        )
        if win_coords:
            win_x, win_y = win_coords
            logger.info(
                f"Opening Pie Menu at {win_x}, {win_y} (Type: {target_type})"
            )
            self.pie_menu.popup_at_location(win_x, win_y)

        gesture.set_state(Gtk.EventSequenceState.CLAIMED)

    def on_tool_selected(self, sender, tool: str):
        logger.info(f"Tool activated: {tool}")
        if self.sketch_element:
            self.sketch_element.set_tool(tool)

    def on_constraint_selected(self, sender, constraint_type: str):
        logger.info(f"Constraint activated: {constraint_type}")
        if not self.sketch_element:
            return

        ctx = self.sketch_element
        if constraint_type == "dist":
            ctx.add_distance_constraint()
        elif constraint_type == "horiz":
            ctx.add_horizontal_constraint()
        elif constraint_type == "vert":
            ctx.add_vertical_constraint()
        elif constraint_type == "radius":
            ctx.add_radius_constraint()
        elif constraint_type == "diameter":
            ctx.add_diameter_constraint()
        elif constraint_type == "perp":
            ctx.add_perpendicular()
        elif constraint_type == "tangent":
            ctx.add_tangent()
        elif constraint_type == "align":
            ctx.add_alignment_constraint()
        elif constraint_type == "equal":
            ctx.add_equal_constraint()

    def on_action_triggered(self, sender, action: str):
        logger.info(f"Action activated: {action}")
        if not self.sketch_element:
            return

        ctx = self.sketch_element
        if action == "construction":
            ctx.toggle_construction_on_selection()
        elif action == "delete":
            ctx.delete_selection()

    def handle_key_press(
        self, keyval: int, keycode: int, state: Gdk.ModifierType
    ) -> bool:
        """Handles key press events for the sketcher session."""
        if not self.sketch_element:
            return False

        is_ctrl = state & Gdk.ModifierType.CONTROL_MASK

        # Undo / Redo
        if is_ctrl:
            if keyval == Gdk.KEY_z:
                self.history_manager.undo()
                return True
            if keyval == Gdk.KEY_y:
                self.history_manager.redo()
                return True

        # Handle sketcher-specific keys
        if keyval == Gdk.KEY_Delete:
            self.sketch_element.delete_selection()
            return True
        elif keyval == Gdk.KEY_Escape:
            # If any elements are selected, unselect them.
            if self.sketch_element.get_selected_elements():
                self.sketch_element.unselect_all()
                return True

        return False

import logging
from typing import Union, Optional, TYPE_CHECKING
from gi.repository import Gtk, Gdk, GLib
from ...core.sketcher.entities import Point, Entity
from ...core.sketcher.constraints import Constraint
from ...core.sketcher.tools import SelectTool
from ...core.undo import HistoryManager
from ..canvas.cursor import get_tool_cursor
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

    KEY_SEQUENCE_TIMEOUT_MS = 1500  # 1.5 seconds

    def __init__(self, parent_window: Gtk.Window):
        self.parent_window = parent_window
        self.sketch_element: Optional["SketchElement"] = None

        # The SketchEditor manages its own undo/redo history, separate from
        # the main document editor.
        self.history_manager = HistoryManager()

        # 1. Key Press Handling State
        self.key_sequence = []
        self.key_sequence_timer_id: Optional[int] = None
        self._init_shortcuts()

        # 2. Pie Menu Setup
        # Pass the shortcuts dict to the pie menu for label generation
        self.pie_menu = SketchPieMenu(self.parent_window, self.shortcuts)

        # Connect signals
        self.pie_menu.tool_selected.connect(self.on_tool_selected)
        self.pie_menu.constraint_selected.connect(self.on_constraint_selected)
        self.pie_menu.action_triggered.connect(self.on_action_triggered)
        self.pie_menu.right_clicked.connect(self.on_pie_menu_right_click)

    def _init_shortcuts(self):
        """Initializes the keyboard shortcut mappings."""
        # This can only be called once we have a sketch_element, so we map
        # to function names and resolve them at runtime.
        self.shortcuts = {
            # Tools
            " ": "set_tool:select",
            "gl": "set_tool:line",
            "ga": "set_tool:arc",
            "gc": "set_tool:circle",
            "gr": "set_tool:rectangle",
            "go": "set_tool:rounded_rect",
            "gf": "set_tool:fill",
            "gn": "toggle_construction_on_selection",
            # Actions
            "ch": "add_chamfer_action",
            "cf": "add_fillet_action",
            # Constraints (Single Key)
            "h": "add_horizontal_constraint",
            "v": "add_vertical_constraint",
            "n": "add_perpendicular",
            "t": "add_tangent",
            "e": "add_equal_constraint",
            "o": "add_alignment_constraint",
            "c": "add_alignment_constraint",  # FreeCAD alias
            "s": "add_symmetry_constraint",
            # Constraints (K prefix)
            "kd": "add_distance_constraint",
            "kr": "add_radius_constraint",
            "ko": "add_diameter_constraint",
        }
        # A set of all prefixes for quick checking (e.g., "g", "k")
        self.shortcut_prefixes = {
            s[:i] for s in self.shortcuts for i in range(1, len(s))
        }

    def activate(self, sketch_element: "SketchElement"):
        """Begins an editing session on the given SketchElement."""
        logger.debug(f"Activating SketchEditor for element {sketch_element}")
        self.sketch_element = sketch_element
        self.sketch_element.editor = self

    def deactivate(self):
        """Ends the current editing session."""
        logger.debug("Deactivating SketchEditor")
        self._reset_key_sequence()
        if self.sketch_element:
            # Clean up any in-progress tool state
            self.sketch_element.current_tool.on_deactivate()
            if self.sketch_element.canvas:
                # Reset cursor to default
                self.sketch_element.canvas.set_cursor(None)
            self.sketch_element.editor = None
        self.sketch_element = None
        if self.pie_menu.is_visible():
            self.pie_menu.popdown()

    def get_current_cursor(self) -> Optional[Gdk.Cursor]:
        """
        Determines the appropriate cursor based on the current tool and
        context (e.g., hovering over a point).
        """
        if not self.sketch_element:
            return None

        # Priority 1: Check for specific hover states in the 'select' tool.
        select_tool = self.sketch_element.tools.get("select")
        if (
            self.sketch_element.active_tool_name == "select"
            and isinstance(select_tool, SelectTool)
            and select_tool.hovered_point_id is not None
        ):
            return Gdk.Cursor.new_from_name("move")

        # Priority 2: Return a tool-specific cursor.
        tool = self.sketch_element.active_tool_name
        if tool == "line":
            return get_tool_cursor("sketch-line-symbolic")
        if tool == "arc":
            return get_tool_cursor("sketch-arc-symbolic")
        if tool == "circle":
            return get_tool_cursor("sketch-circle-symbolic")
        if tool == "fill":
            return get_tool_cursor("sketch-fill-symbolic")
        if tool == "rectangle":
            return get_tool_cursor("sketch-rect-symbolic")
        if tool == "rounded_rect":
            return get_tool_cursor("sketch-rounded-rect-symbolic")

        # Default cursor for 'select' tool or any other case.
        return Gdk.Cursor.new_from_name("default")

    def on_pie_menu_right_click(self, sender, gesture, n_press, x, y):
        """
        Handles a right-click that happened on the PieMenu's drawing area.
        Translates coordinates and forwards to the main right-click handler.
        """
        sketch_element = self.sketch_element
        if not sketch_element or not sketch_element.canvas:
            return

        child = self.pie_menu.get_child()
        if not child:
            return

        canvas_coords = child.translate_coordinates(
            sketch_element.canvas, x, y
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
        sketch_element = self.sketch_element
        if not sketch_element or not sketch_element.canvas:
            return

        if self.pie_menu.is_visible():
            self.pie_menu.popdown()

        # Use the element's canvas to convert from widget to world coordinates
        world_x, world_y = sketch_element.canvas._get_world_coords(x, y)

        target: Optional[Union[Point, Entity, Constraint]] = None
        target_type: Optional[str] = None

        # Before showing the menu, we deactivate the current tool to clean
        # up any in-progress state.
        sketch_element.current_tool.on_deactivate()

        selection = sketch_element.selection
        selection_changed = False

        # 1. Hit Test
        hit_type, hit_obj = sketch_element.hittester.get_hit_data(
            world_x, world_y, sketch_element
        )
        target_type = hit_type

        # 2. Resolve Hit Object to Concrete Type AND Update Selection
        # If the clicked object is not already selected, select it.
        if hit_type == "point":
            assert isinstance(hit_obj, int)
            pid = hit_obj
            target = sketch_element.sketch.registry.get_point(pid)

            # Check if this point is a valid chamfer corner (2 lines). If so,
            # promote the selection type to "junction".
            if len(sketch_element.get_lines_at_point(pid)) == 2:
                target_type = "junction"
                if selection.junction_pid != pid:
                    selection.select_junction(pid, is_multi=False)
                    selection_changed = True
            else:
                if pid not in selection.point_ids:
                    selection.select_point(pid, is_multi=False)
                    selection_changed = True

        elif hit_type == "junction":
            assert isinstance(hit_obj, int)
            pid = hit_obj
            target = sketch_element.sketch.registry.get_point(pid)

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
            if 0 <= idx < len(sketch_element.sketch.constraints):
                target = sketch_element.sketch.constraints[idx]

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
            sketch_element.mark_dirty()

        # 3. Pass Context (Sketch, Target, Type)
        self.pie_menu.set_context(sketch_element, target, target_type)

        win_coords = sketch_element.canvas.translate_coordinates(
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
            if self.sketch_element.canvas:
                self.sketch_element.canvas.grab_focus()

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
        elif constraint_type == "symmetry":
            ctx.add_symmetry_constraint()

        if self.sketch_element.canvas:
            self.sketch_element.canvas.grab_focus()

    def on_action_triggered(self, sender, action: str):
        logger.info(f"Action activated: {action}")
        if not self.sketch_element:
            return

        ctx = self.sketch_element
        if action == "construction":
            ctx.toggle_construction_on_selection()
        elif action == "delete":
            ctx.delete_selection()
        elif action == "chamfer":
            ctx.add_chamfer_action()
        elif action == "fillet":
            ctx.add_fillet_action()

        if self.sketch_element.canvas:
            self.sketch_element.canvas.grab_focus()

    def _on_key_sequence_timeout(self) -> bool:
        """Callback to reset the key sequence after a delay."""
        logger.debug("Key sequence timed out.")
        self.key_sequence_timer_id = None
        self._reset_key_sequence()
        return GLib.SOURCE_REMOVE

    def _reset_key_sequence(self):
        """Clears the key sequence and cancels any pending timeout."""
        self.key_sequence.clear()
        if self.key_sequence_timer_id:
            GLib.source_remove(self.key_sequence_timer_id)
            self.key_sequence_timer_id = None

    def handle_key_press(
        self, keyval: int, keycode: int, state: Gdk.ModifierType
    ) -> bool:
        """Handles key press events for the sketcher session."""
        if not self.sketch_element:
            return False

        is_ctrl = state & Gdk.ModifierType.CONTROL_MASK

        # Priority 1: Immediate actions (Undo/Redo, Delete)
        if is_ctrl:
            if keyval == Gdk.KEY_z:
                self.history_manager.undo()
                self._reset_key_sequence()
                return True
            if keyval == Gdk.KEY_y:
                self.history_manager.redo()
                self._reset_key_sequence()
                return True

        if keyval == Gdk.KEY_Delete:
            self.sketch_element.delete_selection()
            self._reset_key_sequence()
            return True

        # Priority 2: Escape key logic
        if keyval == Gdk.KEY_Escape:
            self._reset_key_sequence()
            # If a tool is active, switch to select tool
            if self.sketch_element.active_tool_name != "select":
                self.sketch_element.set_tool("select")
                return True
            # If elements are selected, unselect them
            if self.sketch_element.get_selected_elements():
                self.sketch_element.unselect_all()
                return True
            return False  # Propagate up if nothing else to do

        # Priority 3: Shortcut sequence handling for normal keys
        key_unicode = Gdk.keyval_to_unicode(keyval)
        if key_unicode == 0:
            # Not a printable character, ignore for sequences.
            return False

        char = chr(key_unicode).lower()
        self.key_sequence.append(char)
        current_sequence = "".join(self.key_sequence)

        logger.debug(f"Key sequence: {current_sequence}")

        # Check for a complete shortcut match
        if current_sequence in self.shortcuts:
            action_str = self.shortcuts[current_sequence]
            logger.info(
                f"Shortcut '{current_sequence}' triggered: {action_str}"
            )

            # Special handling for methods with arguments (e.g., set_tool:line)
            if ":" in action_str:
                method_name, arg = action_str.split(":", 1)
                method = getattr(self.sketch_element, method_name, None)
                if method and callable(method):
                    method(arg)
            else:
                method = getattr(self.sketch_element, action_str, None)
                if method and callable(method):
                    method()

            self._reset_key_sequence()
            return True

        # If it's not a full match, check if it's a prefix of another shortcut
        if current_sequence in self.shortcut_prefixes:
            # It's a valid start, so reset the timeout timer and wait for the
            # next key.
            if self.key_sequence_timer_id:
                GLib.source_remove(self.key_sequence_timer_id)
            self.key_sequence_timer_id = GLib.timeout_add(
                self.KEY_SEQUENCE_TIMEOUT_MS, self._on_key_sequence_timeout
            )
            return True

        # If the sequence is not a match and not a prefix, it's invalid.
        self._reset_key_sequence()
        return False

import logging
from typing import cast, Union, Optional
from gi.repository import Gtk, Gdk
from ...core.sketcher.entities import Point, Entity
from ...core.sketcher.constraints import Constraint
from ..canvas import Canvas
from .piemenu import SketchPieMenu
from .sketchelement import SketchElement

logger = logging.getLogger(__name__)


class SketchCanvas(Canvas):
    def __init__(self, parent_window: Gtk.Window, **kwargs):
        super().__init__(**kwargs)
        self.parent_window = parent_window

        # 1. Edit Key Controller (Delete key)
        self._edit_key_ctrl = Gtk.EventControllerKey.new()
        self._edit_key_ctrl.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        self._edit_key_ctrl.connect("key-pressed", self.on_edit_key_pressed)
        self.add_controller(self._edit_key_ctrl)

        # 2. Pie Menu Setup
        self.pie_menu = SketchPieMenu(self.parent_window)

        # Connect signals
        self.pie_menu.tool_selected.connect(self.on_tool_selected)
        self.pie_menu.constraint_selected.connect(self.on_constraint_selected)
        self.pie_menu.action_triggered.connect(self.on_action_triggered)
        self.pie_menu.right_clicked.connect(self.on_pie_menu_right_click)

        # 3. Right Click Gesture for Pie Menu
        right_click = Gtk.GestureClick()
        right_click.set_button(3)  # Right Mouse Button
        right_click.connect("pressed", self.on_right_click)
        self.add_controller(right_click)

    def on_pie_menu_right_click(self, sender, gesture, n_press, x, y):
        """
        Handles a right-click that happened on the PieMenu's drawing area.
        Translates coordinates and forwards to the main right-click handler.
        """
        child = self.pie_menu.get_child()
        if not child:
            return

        canvas_coords = child.translate_coordinates(self, x, y)
        if canvas_coords:
            canvas_x, canvas_y = canvas_coords
            self.on_right_click(gesture, n_press, canvas_x, canvas_y)

    def on_right_click(self, gesture, n_press, x, y):
        """Open the pie menu at the cursor location with resolved context."""
        if self.pie_menu.is_visible():
            self.pie_menu.popdown()

        world_x, world_y = self._get_world_coords(x, y)

        target: Optional[Union[Point, Entity, Constraint]] = None
        target_type: Optional[str] = None

        # Determine context if we are editing a sketch
        if self.edit_context and isinstance(self.edit_context, SketchElement):
            sketch_elem = self.edit_context
            # Before showing the menu, we deactivate the current tool to clean
            # up any in-progress state.
            sketch_elem.current_tool.on_deactivate()

            selection = sketch_elem.selection
            selection_changed = False

            # 1. Hit Test
            hit_type, hit_obj = sketch_elem.hittester.get_hit_data(
                world_x, world_y, sketch_elem
            )
            target_type = hit_type

            # 2. Resolve Hit Object to Concrete Type AND Update Selection
            # If the clicked object is not already selected, select it.

            if hit_type == "point":
                assert isinstance(hit_obj, int)
                pid = hit_obj
                target = sketch_elem.sketch.registry.get_point(pid)

                if pid not in selection.point_ids:
                    selection.select_point(pid, is_multi=False)
                    selection_changed = True

            elif hit_type == "junction":
                # Junctions are essentially points in the registry
                assert isinstance(hit_obj, int)
                pid = hit_obj
                target = sketch_elem.sketch.registry.get_point(pid)

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
                # hit_obj is index in constraints list
                if 0 <= idx < len(sketch_elem.sketch.constraints):
                    target = sketch_elem.sketch.constraints[idx]

                    if selection.constraint_idx != idx:
                        selection.select_constraint(idx, is_multi=False)
                        selection_changed = True

            # If nothing was hit, clear selection
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
                sketch_elem.mark_dirty()

            # 3. Pass Context (Sketch, Target, Type)
            self.pie_menu.set_context(sketch_elem, target, target_type)

        # Translate coordinates from canvas-local to window-local
        # for the popover, which is parented to the window.
        win_coords = self.translate_coordinates(self.parent_window, x, y)
        if win_coords:
            win_x, win_y = win_coords
            logger.info(
                f"Opening Pie Menu at {win_x}, {win_y} (Type: {target_type})"
            )
            self.pie_menu.popup_at_location(win_x, win_y)

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
        if self.pie_menu.is_visible() and gesture.get_current_button() != 3:
            self.pie_menu.popdown()
            gesture.set_state(Gtk.EventSequenceState.CLAIMED)
            return

        if gesture.get_current_button() == 3:
            return

        if self.edit_context:
            self.grab_focus()
            self._was_dragging = False
            world_x, world_y = self._get_world_coords(x, y)

            if isinstance(self.edit_context, SketchElement):
                ctx = cast(SketchElement, self.edit_context)
                handled = ctx.handle_edit_press(world_x, world_y, n_press)
            else:
                handled = self.edit_context.handle_edit_press(world_x, world_y)

            if not handled and self._hovered_elem is None:
                self.leave_edit_mode()
            self.queue_draw()
            return

        super().on_button_press(gesture, n_press, x, y)

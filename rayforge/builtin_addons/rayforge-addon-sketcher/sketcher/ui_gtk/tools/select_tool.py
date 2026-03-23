import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
from gettext import gettext as _
import cairo

from rayforge.core.geo import Point as GeoPoint
from rayforge.core.matrix import Matrix
from rayforge.ui_gtk.shared.keyboard import PRIMARY_KEY_NAME
from ...core.commands import (
    CreateOrEditConstraintCommand,
    MoveControlPointCommand,
    MovePointCommand,
)
from ...core.constraints import (
    AngleConstraint,
    DiameterConstraint,
    DistanceConstraint,
    DragConstraint,
    RadiusConstraint,
)
from ...core.entities import (
    Arc,
    Bezier,
    Circle,
    Entity,
    Line,
    TextBoxEntity,
)
from ...core.types import EntityID
from .base import SketchTool
from .text_box_tool import TextBoxTool

if TYPE_CHECKING:
    from ...core.selection import SketchSelection

logger = logging.getLogger(__name__)


class SelectTool(SketchTool):
    """Handles selection and point dragging."""

    ICON = "sketch-select-symbolic"
    LABEL = _("Select")
    SHORTCUTS = [" "]

    def __init__(self, element):
        super().__init__(element)
        self.hovered_point_id: Optional[EntityID] = None
        self.hovered_constraint_idx: Optional[int] = None
        self.hovered_junction_pid: Optional[EntityID] = None
        self.hovered_entity_id: Optional[EntityID] = None

        # --- Box Selection State ---
        self.is_box_selecting: bool = False
        self.drag_start_world_pos: Optional[GeoPoint] = None
        self.drag_current_world_pos: Optional[GeoPoint] = None
        self.drag_initial_selection: Optional["SketchSelection"] = None

        # --- Drag State ---
        # For dragging a single point
        self.dragged_point_id: Optional[EntityID] = None
        self.drag_point_start_pos: Optional[GeoPoint] = None

        # For dragging entities (lines/arcs)
        self.dragged_entity: Optional[Entity] = None
        self.drag_start_model_pos: Optional[GeoPoint] = None

        # For dragging control points
        self.dragged_cp_bezier_id: Optional[EntityID] = None
        self.dragged_cp_index: Optional[int] = None
        self.drag_cp_start_offset: Optional[Tuple[float, float]] = None

        # State for stabilizing drag calculations and undo snapshots
        self.drag_start_wt_inv: Optional[Matrix] = None
        self.drag_start_ct_inv: Optional[Matrix] = None

        # Snapshots taken at start of drag
        self.drag_initial_positions: Dict[EntityID, GeoPoint] = {}
        self.drag_initial_entity_states: Dict[EntityID, Any] = {}

        self.drag_point_distances: Dict[EntityID, int] = {}

    def is_available(self, target, target_type) -> bool:
        return target is None

    def shortcut_is_active(self) -> bool:
        return True

    def _is_dragging(self) -> bool:
        """Returns True if currently dragging a point, entity, or CP."""
        return (
            self.dragged_point_id is not None
            or self.dragged_entity is not None
            or self.dragged_cp_bezier_id is not None
        )

    def _get_drag_start_world_pos(self) -> Tuple[float, float]:
        """Returns the world-space start position of the current drag."""
        if self.drag_point_start_pos is not None:
            return self.drag_point_start_pos
        if self.drag_start_model_pos is not None:
            return self.drag_start_model_pos
        return 0.0, 0.0

    @staticmethod
    def _calculate_snap_offset(target_pos: float, grid_size: float) -> float:
        """Calculates snap offset to align target to nearest grid line."""
        if grid_size <= 0:
            return 0.0
        snapped = round(target_pos / grid_size) * grid_size
        return snapped - target_pos

    def get_active_shortcuts(
        self,
    ) -> List[Tuple[Union[str, List[str]], str, Optional[Callable[[], bool]]]]:
        """Returns shortcuts available based on current tool state."""
        return [
            (PRIMARY_KEY_NAME, _("Snap to Grid"), lambda: self._is_dragging()),
            ("Shift", _("Constrain to Axis"), lambda: self._is_dragging()),
            (
                ["Shift", "Doubleclick"],
                _("Select Connected"),
                lambda: not self._is_dragging(),
            ),
        ]

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        hit_type, hit_obj = self.element.hittester.get_hit_data(
            world_x, world_y, self.element
        )
        logger.debug(
            f"SelectTool.on_press: n_press={n_press}, hit_type='{hit_type}'"
        )

        is_shift_pressed = False
        if self.element.canvas:
            is_shift_pressed = self.element.canvas._shift_pressed

        # Shift+double click on entity to select all connected geometry
        if n_press == 2 and hit_type == "entity" and is_shift_pressed:
            logger.debug("Shift+double-click on entity detected.")
            entity = cast(Entity, hit_obj)
            self.element.selection.select_connected_entities(
                entity.id, self.element.sketch.registry
            )
            self.element.mark_dirty()
            return True

        # Shift+double click on point to select all connected geometry
        if n_press == 2 and hit_type == "point" and is_shift_pressed:
            logger.debug("Shift+double-click on point detected.")
            pid = cast(EntityID, hit_obj)
            registry = self.element.sketch.registry
            entity_ids = self.element.selection.entity_ids
            entity_ids.clear()
            for entity in registry.entities:
                if pid in entity.get_point_ids():
                    connected = registry.get_connected_entity_ids(entity.id)
                    entity_ids.extend(connected)
            entity_ids[:] = list(set(entity_ids))
            self.element.selection.point_ids.clear()
            self.element.selection.constraint_idx = None
            self.element.selection.junction_pid = None
            self.element.selection.changed.send(self.element.selection)
            self.element.mark_dirty()
            return True

        # Double click on entity to add/edit constraints. This is a terminal
        # action, so returning True is correct.
        if n_press == 2 and hit_type == "entity":
            logger.debug("Double-click on entity detected.")
            entity = cast(Entity, hit_obj)
            if isinstance(entity, (Arc, Line, Circle)):
                cmd = CreateOrEditConstraintCommand(
                    self.element.sketch, entity
                )
                self.element.execute_command(cmd)
                if cmd.constraint is not None:
                    logger.debug(f"Constraint for editing: {cmd.constraint}")
                    self.element.constraint_edit_requested.send(
                        self.element, constraint=cmd.constraint
                    )
                return True
            elif isinstance(entity, TextBoxEntity):
                text_tool = self.element.tools.get("text_box")
                if isinstance(text_tool, TextBoxTool):
                    self.element.set_tool("text_box")
                    text_tool.start_editing(entity.id)
                    # Delegate the press to the new tool so it can
                    # initialize its own drag state, then return False to
                    # allow the gesture to continue into a drag.
                    text_tool.on_press(world_x, world_y, 1)
                    return False
                return True

        # Double click edits constraint value
        if n_press == 2 and hit_type == "constraint":
            logger.debug("Double-click on constraint detected.")
            idx = cast(int, hit_obj)
            constraints = self.element.sketch.constraints
            if constraints and idx < len(constraints):
                constr = constraints[idx]
                if isinstance(
                    constr,
                    (
                        AngleConstraint,
                        DiameterConstraint,
                        DistanceConstraint,
                        RadiusConstraint,
                    ),
                ):
                    logger.debug(
                        f"Emitting signal for constraint edit: {constr}"
                    )
                    self.element.constraint_edit_requested.send(
                        self.element, constraint=constr
                    )
                    return True

        # --- SINGLE CLICK LOGIC ---
        # For single-clicks (n_press == 1), we must return False to allow the
        # GTK gesture to continue listening for a potential second click.
        # Returning True here would terminate the gesture recognition.

        is_multi = False
        if self.element.canvas:
            is_multi = (
                self.element.canvas._shift_pressed
                or self.element.canvas._ctrl_pressed
            )

        if hit_type == "constraint":
            idx = cast(int, hit_obj)
            self.element.selection.select_constraint(idx, is_multi)

            # Also prepare for a drag if the constraint is point-like.
            constraints = self.element.sketch.constraints
            if constraints and idx < len(constraints):
                constr = constraints[idx]
                pid_to_drag = constr.get_draggable_point()

                if pid_to_drag is not None:
                    self._prepare_point_drag(pid_to_drag)

            self.element.mark_dirty()
            return False

        if hit_type == "junction":
            pid = cast(EntityID, hit_obj)
            self.element.selection.select_junction(pid, is_multi)
            self._prepare_point_drag(pid)
            self.element.mark_dirty()
            return False

        if hit_type in ("control_point_in", "control_point_out"):
            point_id, bezier_id, cp_index = hit_obj
            self._prepare_control_point_drag(bezier_id, cp_index)
            self.element.mark_dirty()
            return False

        if hit_type == "point":
            pid = cast(EntityID, hit_obj)
            self.element.selection.select_point(pid, is_multi)
            self._prepare_point_drag(pid)
            self.element.mark_dirty()
            return False

        elif hit_type == "entity":
            entity = cast(Entity, hit_obj)
            self.element.selection.select_entity(entity, is_multi)
            mx, my = self.element.hittester.screen_to_model(
                world_x, world_y, self.element
            )
            self._prepare_entity_drag(entity, mx, my)
            self.element.mark_dirty()
            return False

        else:
            # Click on empty space: Prepare for Box Selection
            if not is_multi:
                self.element.selection.clear()
                self.drag_initial_selection = None
            else:
                # Store the selection state BEFORE the drag
                self.drag_initial_selection = self.element.selection.copy()

            self.is_box_selecting = True
            self.drag_start_world_pos = (world_x, world_y)
            self.drag_current_world_pos = (world_x, world_y)
            self.element.mark_dirty()
            return False

    def on_drag(self, world_dx: float, world_dy: float):
        if self._is_dragging() and self.element.canvas:
            canvas = self.element.canvas
            if canvas._shift_pressed:
                if abs(world_dx) > abs(world_dy):
                    world_dy = 0.0
                else:
                    world_dx = 0.0
            if canvas._ctrl_pressed and canvas.grid_size > 0:
                start_x, start_y = self._get_drag_start_world_pos()
                target_x = start_x + world_dx
                target_y = start_y + world_dy
                snap_offset_x = self._calculate_snap_offset(
                    target_x, canvas.grid_size
                )
                snap_offset_y = self._calculate_snap_offset(
                    target_y, canvas.grid_size
                )
                world_dx += snap_offset_x
                world_dy += snap_offset_y

        if self.dragged_cp_bezier_id is not None:
            self._handle_control_point_drag(world_dx, world_dy)
        elif self.dragged_point_id is not None:
            self._handle_point_drag(world_dx, world_dy)
        elif self.dragged_entity is not None:
            self._handle_entity_drag(world_dx, world_dy)
        elif self.is_box_selecting and self.drag_start_world_pos:
            # Update current drag position and perform live selection
            start_x, start_y = self.drag_start_world_pos
            self.drag_current_world_pos = (
                start_x + world_dx,
                start_y + world_dy,
            )
            self._update_live_box_selection()
            self.element.mark_dirty()

    def on_release(self, world_x: float, world_y: float):
        # Handle the end of a Box Selection
        if self.is_box_selecting:
            # Selection is already live. Just clean up the drag state.
            self.is_box_selecting = False
            self.drag_start_world_pos = None
            self.drag_current_world_pos = None
            self.drag_initial_selection = None
            self.element.mark_dirty()
            return

        # Handle the end of a Control Point drag
        if (
            self.dragged_cp_bezier_id is not None
            and self.dragged_cp_index is not None
            and self.drag_cp_start_offset is not None
        ):
            bezier_id = self.dragged_cp_bezier_id
            cp_index = self.dragged_cp_index
            bezier = self._safe_get_entity(bezier_id)
            if isinstance(bezier, Bezier):
                start_offset = self.drag_cp_start_offset
                end_offset = bezier.cp1 if cp_index == 1 else bezier.cp2
                if start_offset != end_offset:
                    cmd = MoveControlPointCommand(
                        self.element.sketch,
                        bezier_id,
                        cp_index,
                        start_offset,
                        end_offset,
                    )
                    self.element.execute_command(cmd)

            self.dragged_cp_bezier_id = None
            self.dragged_cp_index = None
            self.drag_cp_start_offset = None
            self.element.mark_dirty()
            return

        # If a point was dragged, create an undoable command
        if self.dragged_point_id is not None and self.drag_point_start_pos:
            p = self._safe_get_point(self.dragged_point_id)
            if p:
                start_x, start_y = self.drag_point_start_pos
                end_x, end_y = p.x, p.y

                # Only create a command if the point actually moved
                if abs(start_x - end_x) > 1e-6 or abs(start_y - end_y) > 1e-6:
                    # Pass the full snapshot (points + entities)
                    # We must copy because self.drag_initial_* are cleared
                    # below
                    snapshot = (
                        self.drag_initial_positions.copy(),
                        self.drag_initial_entity_states.copy(),
                    )
                    cmd = MovePointCommand(
                        self.element.sketch,
                        self.dragged_point_id,
                        (start_x, start_y),
                        (end_x, end_y),
                        snapshot=snapshot,
                    )
                    self.element.execute_command(cmd)

        # Clear all drag-related state
        self.dragged_point_id = None
        self.drag_point_start_pos = None
        self.dragged_entity = None
        self.drag_start_model_pos = None
        self.drag_initial_positions.clear()
        self.drag_initial_entity_states.clear()
        self.drag_point_distances.clear()
        self.drag_start_wt_inv = None
        self.drag_start_ct_inv = None

        # Final solve, now allowing constraint status to be updated.
        # Note: The command execution will have already triggered a solve.
        # This solve is for the final state after releasing the mouse.
        self.element.sketch.solve()
        # Perform a final, guaranteed update to settle the bounds.
        self.element.update_bounds_from_sketch()
        self.element.mark_dirty()

    def on_hover_motion(self, world_x: float, world_y: float):
        if self.is_box_selecting:
            return

        hit_type, hit_obj = self.element.hittester.get_hit_data(
            world_x, world_y, self.element
        )

        new_hover_pid = None
        new_hover_constraint_idx = None
        new_hover_junction_pid = None
        new_hover_entity_id = None

        if hit_type == "point":
            new_hover_pid = hit_obj
        elif hit_type == "constraint":
            new_hover_constraint_idx = hit_obj
        elif hit_type == "junction":
            new_hover_junction_pid = hit_obj
        elif hit_type == "entity":
            new_hover_entity_id = hit_obj.id

        if (
            self.hovered_point_id != new_hover_pid
            or self.hovered_constraint_idx != new_hover_constraint_idx
            or self.hovered_junction_pid != new_hover_junction_pid
            or self.hovered_entity_id != new_hover_entity_id
        ):
            self.hovered_point_id = new_hover_pid
            self.hovered_constraint_idx = new_hover_constraint_idx
            self.hovered_junction_pid = new_hover_junction_pid
            self.hovered_entity_id = new_hover_entity_id
            self.element.mark_dirty()

    def draw_overlay(self, ctx: cairo.Context):
        """Draws the selection box."""
        if (
            not self.is_box_selecting
            or not self.drag_start_world_pos
            or not self.drag_current_world_pos
        ):
            return

        if not self.element.canvas:
            return

        # Transform World Coordinates to Screen Coordinates
        view_transform = self.element.canvas.view_transform
        start_px = view_transform.transform_point(self.drag_start_world_pos)
        curr_px = view_transform.transform_point(self.drag_current_world_pos)

        x, y = start_px
        w = curr_px[0] - x
        h = curr_px[1] - y

        ctx.save()

        # Draw a blue selection box.
        ctx.set_source_rgba(0.2, 0.6, 1.0, 0.2)  # Selection Fill: Blue
        ctx.set_dash([4, 2])
        ctx.rectangle(x, y, w, h)
        ctx.fill_preserve()

        # Stroke border
        ctx.set_source_rgba(0.2, 0.6, 1.0, 0.7)  # Selection Border: Blue
        ctx.set_line_width(1.0)
        ctx.stroke()

        ctx.restore()

    # --- Drag Logic Handlers ---

    def _update_live_box_selection(self):
        """
        Calculates and updates the selection based on the current drag box.
        """
        if not self.drag_start_world_pos or not self.drag_current_world_pos:
            return

        # Calculate box in world coords
        start_wx, start_wy = self.drag_start_world_pos
        end_wx, end_wy = self.drag_current_world_pos

        # Convert world box to Model Space for query
        mx1, my1 = self.element.hittester.screen_to_model(
            start_wx, start_wy, self.element
        )
        mx2, my2 = self.element.hittester.screen_to_model(
            end_wx, end_wy, self.element
        )
        model_min_x = min(mx1, mx2)
        model_max_x = max(mx1, mx2)
        model_min_y = min(my1, my2)
        model_max_y = max(my1, my2)

        # Perform a "crossing" selection.
        points_hit, entities_hit = self.element.hittester.get_objects_in_rect(
            model_min_x,
            model_min_y,
            model_max_x,
            model_max_y,
            self.element,
            strict_containment=False,
        )

        is_additive = self.drag_initial_selection is not None

        if is_additive:
            # Restore the pre-drag state
            initial_state = self.drag_initial_selection
            if initial_state:
                self.element.selection.point_ids = initial_state.point_ids[:]
                self.element.selection.entity_ids = initial_state.entity_ids[:]

            # Add newly found items
            for pid in points_hit:
                if pid not in self.element.selection.point_ids:
                    self.element.selection.point_ids.append(pid)

            for eid in entities_hit:
                if eid not in self.element.selection.entity_ids:
                    self.element.selection.entity_ids.append(eid)
        else:
            # Not additive, so the selection is exactly what's in the box
            self.element.selection.point_ids = points_hit
            self.element.selection.entity_ids = entities_hit
            # Clear other selection types
            self.element.selection.constraint_idx = None
            self.element.selection.junction_pid = None

    def _get_model_delta(self, world_dx: float, world_dy: float) -> GeoPoint:
        """Safely converts a world-space delta to a model-space delta."""
        if self.drag_start_wt_inv is None or self.drag_start_ct_inv is None:
            return 0.0, 0.0

        wt_vec = self.drag_start_wt_inv.without_translation()
        ct_vec = self.drag_start_ct_inv.without_translation()

        ldx, ldy = wt_vec.transform_vector((world_dx, world_dy))
        mdx, mdy = ct_vec.transform_vector((ldx, ldy))
        return mdx, mdy

    def _handle_point_drag(self, world_dx: float, world_dy: float):
        """Logic for dragging a single point."""
        if self.dragged_point_id is None or self.drag_point_start_pos is None:
            return

        mdx, mdy = self._get_model_delta(world_dx, world_dy)
        start_x, start_y = self.drag_point_start_pos
        target_x = start_x + mdx
        target_y = start_y + mdy

        drag_constraints = []

        # Ask the sketch model for the group of points that must move together.
        coincident_group = self.element.sketch.get_coincident_points(
            self.dragged_point_id
        )

        # Apply a strong drag constraint to ALL points in the group.
        for pid in coincident_group:
            drag_constraints.append(
                DragConstraint(pid, target_x, target_y, weight=0.1)
            )

        base_hold_weight = 0.01
        max_hops = max(
            (d for d in self.drag_point_distances.values() if d > 0), default=1
        )
        for pid, pos in self.drag_initial_positions.items():
            # Skip any point that is part of the actively dragged group.
            if pid in coincident_group:
                continue

            p = self._safe_get_point(pid)
            if not p or p.fixed:
                continue
            hops = self.drag_point_distances.get(pid, -1)
            weight = 0
            if hops == -1:
                weight = base_hold_weight
            elif hops > 0:
                weight = base_hold_weight * (hops / max_hops)
            if weight > 0:
                drag_constraints.append(
                    DragConstraint(pid, pos[0], pos[1], weight=weight)
                )

        self.element.sketch.solve(
            extra_constraints=drag_constraints, update_constraint_status=False
        )
        self.element.mark_dirty()

    def _handle_entity_drag(self, world_dx: float, world_dy: float):
        """Logic for dragging one or more selected entities."""
        mdx, mdy = self._get_model_delta(world_dx, world_dy)
        if mdx == 0 and mdy == 0:
            return

        # 1. Identify all unique points from all selected entities
        points_to_drag = set()
        for eid in self.element.selection.entity_ids:
            entity = self.element.sketch.registry.get_entity(eid)
            if entity:
                points_to_drag.update(entity.get_point_ids())

        drag_constraints = []
        strong_drag_weight = 1.0
        for pid in points_to_drag:
            p = self._safe_get_point(pid)
            if not p or p.fixed:
                continue

            initial_pos = self.drag_initial_positions.get(pid)
            if initial_pos:
                target_x = initial_pos[0] + mdx
                target_y = initial_pos[1] + mdy
                drag_constraints.append(
                    DragConstraint(
                        pid, target_x, target_y, weight=strong_drag_weight
                    )
                )

        # 3. Add weak "holding" constraints for all other points
        hold_weight = 0.01
        for pid, pos in self.drag_initial_positions.items():
            if pid in points_to_drag:
                continue
            p = self._safe_get_point(pid)
            if not p or p.fixed:
                continue
            drag_constraints.append(
                DragConstraint(pid, pos[0], pos[1], weight=hold_weight)
            )

        # 4. Solve and update
        self.element.sketch.solve(
            extra_constraints=drag_constraints, update_constraint_status=False
        )
        self.element.mark_dirty()

    # --- Drag Preparation ---

    def _prepare_point_drag(self, pid: EntityID):
        """Sets up state for dragging a single point."""
        self.dragged_point_id = pid
        self.dragged_entity = None  # Mutually exclusive
        p = self._safe_get_point(pid)
        if not p:
            return

        self.drag_point_start_pos = (p.x, p.y)
        self._cache_drag_start_state()
        self._calculate_geometric_hops(pid)

    def _prepare_entity_drag(
        self, entity: Entity, model_x: float, model_y: float
    ):
        """Sets up state for dragging an entity (or group of entities)."""
        self.dragged_entity = entity
        self.dragged_point_id = None  # Mutually exclusive
        self.drag_start_model_pos = (model_x, model_y)
        self._cache_drag_start_state()

    def _prepare_control_point_drag(self, bezier_id: EntityID, cp_index: int):
        """Sets up state for dragging a control point.

        Does not clear selection.
        """
        self.dragged_cp_bezier_id = bezier_id
        self.dragged_cp_index = cp_index
        self.dragged_point_id = None
        self.dragged_entity = None
        bezier = self._safe_get_entity(bezier_id)
        if not isinstance(bezier, Bezier):
            return
        if cp_index == 1:
            self.drag_cp_start_offset = bezier.cp1
        else:
            self.drag_cp_start_offset = bezier.cp2
        self._cache_drag_start_state()

    def _handle_control_point_drag(self, world_dx: float, world_dy: float):
        """Logic for dragging a control point offset."""
        if self.dragged_cp_bezier_id is None or self.dragged_cp_index is None:
            return
        bezier = self._safe_get_entity(self.dragged_cp_bezier_id)
        if not isinstance(bezier, Bezier):
            return
        mdx, mdy = self._get_model_delta(world_dx, world_dy)
        if self.drag_cp_start_offset is None:
            base_x, base_y = 0.0, 0.0
        else:
            base_x, base_y = self.drag_cp_start_offset
        new_offset = (base_x + mdx, base_y + mdy)

        if self.dragged_cp_index == 1:
            bezier.cp1 = new_offset
            point_id = bezier.start_idx
        else:
            bezier.cp2 = new_offset
            point_id = bezier.end_idx

        p = self._safe_get_point(point_id)
        if p is not None:
            registry = self.element.sketch.registry
            p.apply_constraint(
                registry, bezier, self.dragged_cp_index, self.element.sketch
            )

        self.element.mark_dirty()

    def _cache_drag_start_state(self):
        """
        Caches transforms and ALL state (points + entities) at start of drag.
        """
        self.drag_start_wt_inv = self.element.get_world_transform().invert()
        self.drag_start_ct_inv = self.element.content_transform.invert()

        # Capture Points
        self.drag_initial_positions = {
            pt.id: (pt.x, pt.y) for pt in self.element.sketch.registry.points
        }

        # Capture Entity States
        self.drag_initial_entity_states = {}
        for e in self.element.sketch.registry.entities:
            state = e.get_state()
            if state is not None:
                self.drag_initial_entity_states[e.id] = state

    def _safe_get_point(self, pid: EntityID):
        try:
            return self.element.sketch.registry.get_point(pid)
        except IndexError:
            return None

    def _safe_get_entity(self, eid: EntityID):
        try:
            return self.element.sketch.registry.get_entity(eid)
        except IndexError:
            return None

    def _calculate_geometric_hops(self, start_pid: EntityID):
        """
        Calculates distance (in entity hops) from a start point to all others
        using BFS. Results are stored in self.drag_point_distances.
        """
        registry = self.element.sketch.registry
        if not registry.points:
            self.drag_point_distances = {}
            return

        adj: Dict[EntityID, List[EntityID]] = {
            p.id: [] for p in registry.points
        }
        for entity in registry.entities:
            if isinstance(entity, Line):
                p1, p2 = entity.p1_idx, entity.p2_idx
                if p1 in adj and p2 in adj:
                    adj[p1].append(p2)
                    adj[p2].append(p1)
            elif isinstance(entity, Arc):
                s, e, c = entity.start_idx, entity.end_idx, entity.center_idx
                if s in adj and e in adj and c in adj:
                    adj[s].extend([e, c])
                    adj[e].extend([s, c])
                    adj[c].extend([s, e])
            elif isinstance(entity, Circle):
                c, r = entity.center_idx, entity.radius_pt_idx
                if c in adj and r in adj:
                    adj[c].append(r)
                    adj[r].append(c)

        q = [(start_pid, 0)]
        distances = {p.id: -1 for p in registry.points}
        if start_pid in distances:
            distances[start_pid] = 0

        head = 0
        while head < len(q):
            curr_pid, dist = q[head]
            head += 1

            for neighbor_pid in adj.get(curr_pid, []):
                if distances.get(neighbor_pid) == -1:  # Unvisited
                    distances[neighbor_pid] = dist + 1
                    q.append((neighbor_pid, dist + 1))

        self.drag_point_distances = distances

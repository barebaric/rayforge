import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
from gettext import gettext as _
import cairo

from rayforge.core.geo import Point as GeoPoint
from rayforge.core.matrix import Matrix
from ...core.commands import (
    CreateOrEditConstraintCommand,
    MoveControlPointCommand,
    MovePointCommand,
)
from ...core.constraints import (
    AngleConstraint,
    CoincidentConstraint,
    DiameterConstraint,
    DistanceConstraint,
    DragConstraint,
    PointOnLineConstraint,
    RadiusConstraint,
    SymmetryConstraint,
)
from ...core.constraints.symmetry import draw_symmetry_arrows
from ...core.entities import (
    Arc,
    Bezier,
    Circle,
    Entity,
    Line,
    Point,
    TextBoxEntity,
)
from ...core.snap import (
    SnapResult,
    DragContext,
    SnapLineType,
    SNAP_LINE_STYLES,
    SnapLineStyle,
)
from ...core.types import EntityID
from .base import SketchTool, SketcherKey
from .text_box_tool import TextBoxTool

if TYPE_CHECKING:
    from ...core.selection import SketchSelection


logger = logging.getLogger(__name__)

DEBUG_SNAPPING = os.environ.get("DEBUG_SNAPPING", "").lower() in (
    "1",
    "true",
    "yes",
)
if DEBUG_SNAPPING:
    print("DEBUG_SNAPPING mode enabled")


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

        # --- Magnetic Snap State ---
        self.current_snap_result: Optional[SnapResult] = None
        self.magnetic_snap_enabled: bool = True

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

    def get_active_shortcuts(
        self,
    ) -> List[Tuple[Union[str, List[str]], str, Optional[Callable[[], bool]]]]:
        """Returns shortcuts available based on current tool state."""
        return [
            ("Shift", _("Constrain to Axis"), lambda: self._is_dragging()),
            ("Tab", _("Toggle Magnetic Snap"), lambda: self._is_dragging()),
            (
                ["Shift", "Doubleclick"],
                _("Select Connected"),
                lambda: not self._is_dragging(),
            ),
        ]

    def handle_key_event(
        self, key: SketcherKey, shift: bool = False, ctrl: bool = False
    ) -> bool:
        """Handle key events for toggling magnetic snap."""
        if key == SketcherKey.TAB and self._is_dragging():
            self.magnetic_snap_enabled = not self.magnetic_snap_enabled
            state = "enabled" if self.magnetic_snap_enabled else "disabled"
            logger.debug(f"Magnetic snap {state}")
            return True
        return False

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
                    snap_constraints = self._build_snap_constraints()
                    cmd = MovePointCommand(
                        self.element.sketch,
                        self.dragged_point_id,
                        (start_x, start_y),
                        (end_x, end_y),
                        snapshot=snapshot,
                        snap_constraints=snap_constraints,
                    )
                    self.element.execute_command(cmd)

                self.current_snap_result = None

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
        """Draws the selection box and snap lines."""
        if self.is_box_selecting:
            self._draw_selection_box(ctx)

        if self._is_dragging() and self.current_snap_result is not None:
            self._draw_snap_lines(ctx)

        if DEBUG_SNAPPING and not self._is_dragging():
            self._draw_debug_snap_lines(ctx)

    def _draw_selection_box(self, ctx: cairo.Context):
        if not self.drag_start_world_pos or not self.drag_current_world_pos:
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

        if self.magnetic_snap_enabled and self.element.canvas:
            snap_result = self._query_magnetic_snap((target_x, target_y))
            if snap_result.snapped:
                target_x, target_y = snap_result.position
                self.current_snap_result = snap_result
            else:
                self.current_snap_result = None
        else:
            self.current_snap_result = None

        drag_constraints = []

        # Ask the sketch model for the group of points that must move together.
        coincident_group = self.element.sketch.get_coincident_points(
            self.dragged_point_id
        )

        # Check if any point in the coincident group is fixed - if so,
        # the dragged point can't move and we shouldn't apply rigid drag.
        any_fixed = False
        for pid in coincident_group:
            p = self._safe_get_point(pid)
            if p and p.fixed:
                any_fixed = True
                break

        # Apply a strong drag constraint to ALL points in the coincident
        # group (they all move to the same target position).
        for pid in coincident_group:
            drag_constraints.append(
                DragConstraint(pid, target_x, target_y, weight=0.1)
            )

        # Also include rigidly connected points (e.g., ellipse center drag
        # should move all ellipse points together). These move by the same
        # delta from their initial positions, not to the same target.
        # Skip this if the dragged point is fixed (coincident with fixed).
        dragged_group = coincident_group
        if not any_fixed:
            rigid_points = (
                self.element.sketch.registry.get_rigidly_connected_points(
                    self.dragged_point_id
                )
            )
            for pid in rigid_points:
                if pid in coincident_group:
                    continue
                initial_pos = self.drag_initial_positions.get(pid)
                if initial_pos:
                    p = self._safe_get_point(pid)
                    if p and not p.fixed:
                        rigid_target_x = initial_pos[0] + mdx
                        rigid_target_y = initial_pos[1] + mdy
                        drag_constraints.append(
                            DragConstraint(
                                pid, rigid_target_x, rigid_target_y, weight=0.1
                            )
                        )
            dragged_group = coincident_group | set(rigid_points)

        base_hold_weight = 0.01
        max_hops = max(
            (d for d in self.drag_point_distances.values() if d > 0), default=1
        )
        for pid, pos in self.drag_initial_positions.items():
            # Skip any point that is part of the actively dragged group.
            if pid in dragged_group:
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
        first_entity_point = None
        for eid in self.element.selection.entity_ids:
            entity = self.element.sketch.registry.get_entity(eid)
            if entity:
                entity_points = entity.get_point_ids()
                points_to_drag.update(entity_points)
                if first_entity_point is None and entity_points:
                    first_entity_point = entity_points[0]

        # 2. Magnetic snap: use first entity point as reference
        if self.magnetic_snap_enabled and self.element.canvas:
            if first_entity_point is not None:
                initial_pos = self.drag_initial_positions.get(
                    first_entity_point
                )
                if initial_pos:
                    ref_target_x = initial_pos[0] + mdx
                    ref_target_y = initial_pos[1] + mdy
                    snap_result = self._query_magnetic_snap(
                        (ref_target_x, ref_target_y)
                    )
                    if snap_result.snapped:
                        snapped_x, snapped_y = snap_result.position
                        mdx = snapped_x - initial_pos[0]
                        mdy = snapped_y - initial_pos[1]
                        self.current_snap_result = snap_result
                    else:
                        self.current_snap_result = None
            else:
                self.current_snap_result = None
        else:
            self.current_snap_result = None

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

    def _get_magnetic_snap_context(self) -> DragContext:
        dragged_point_ids: Set[EntityID] = set()
        dragged_entity_ids: Set[EntityID] = set()

        if self.dragged_point_id is not None:
            dragged_point_ids.add(self.dragged_point_id)
            dragged_point_ids.update(
                self.element.sketch.get_coincident_points(
                    self.dragged_point_id
                )
            )
        elif self.dragged_entity is not None:
            dragged_entity_ids.add(self.dragged_entity.id)
            for eid in self.element.selection.entity_ids:
                dragged_entity_ids.add(eid)
                entity = self.element.sketch.registry.get_entity(eid)
                if entity:
                    dragged_point_ids.update(entity.get_point_ids())

        return DragContext(
            dragged_point_ids=dragged_point_ids,
            dragged_entity_ids=dragged_entity_ids,
            initial_positions=self.drag_initial_positions,
        )

    def _query_magnetic_snap(self, position: GeoPoint) -> SnapResult:
        context = self._get_magnetic_snap_context()
        if self.element.canvas:
            scale_x, _ = self.element.canvas.view_transform.get_scale()
            if scale_x > 0:
                self.element.snap_engine.threshold = 5.0 / scale_x
        return self.element.snap_engine.query(
            self.element.sketch.registry, position, context
        )

    def _build_snap_constraints(self) -> List[Any]:
        constraints: List[Any] = []

        if (
            not self.current_snap_result
            or not self.current_snap_result.primary_snap_point
            or self.dragged_point_id is None
        ):
            return constraints

        sp = self.current_snap_result.primary_snap_point

        if sp.line_type == SnapLineType.MIDPOINT:
            if isinstance(sp.source, Line):
                line = sp.source
                if self.dragged_point_id not in (line.p1_idx, line.p2_idx):
                    constraints.append(
                        SymmetryConstraint(
                            p1=line.p1_idx,
                            p2=line.p2_idx,
                            center=self.dragged_point_id,
                        )
                    )

        elif sp.line_type == SnapLineType.ENTITY_POINT:
            if isinstance(sp.source, Point):
                target_point = sp.source
                if target_point.id != self.dragged_point_id:
                    constraints.append(
                        CoincidentConstraint(
                            p1=self.dragged_point_id,
                            p2=target_point.id,
                        )
                    )

        elif sp.line_type == SnapLineType.ON_ENTITY:
            if isinstance(sp.source, (Line, Arc, Circle)):
                entity = sp.source
                constraints.append(
                    PointOnLineConstraint(
                        point_id=self.dragged_point_id,
                        shape_id=entity.id,
                    )
                )

        return constraints

    def _draw_snap_lines(self, ctx: cairo.Context) -> None:
        if not self.current_snap_result or not self.element.canvas:
            return

        to_screen = self.element.hittester.get_model_to_screen_transform(
            self.element
        )

        canvas_width = self.element.canvas.get_width()
        canvas_height = self.element.canvas.get_height()

        ctx.save()

        sp = self.current_snap_result.primary_snap_point
        skip_snap_lines = sp is not None and sp.line_type in (
            SnapLineType.ENTITY_POINT,
            SnapLineType.MIDPOINT,
            SnapLineType.ON_ENTITY,
        )

        if not skip_snap_lines:
            for snap_line in self.current_snap_result.snap_lines:
                style = snap_line.style
                ctx.set_source_rgba(*style.color)
                if style.dash:
                    ctx.set_dash(style.dash)
                else:
                    ctx.set_dash([])
                ctx.set_line_width(style.line_width)

                if snap_line.is_horizontal:
                    _, screen_y = to_screen.transform_point(
                        (0, snap_line.coordinate)
                    )
                    ctx.move_to(0, screen_y)
                    ctx.line_to(canvas_width, screen_y)
                else:
                    screen_x, _ = to_screen.transform_point(
                        (snap_line.coordinate, 0)
                    )
                    ctx.move_to(screen_x, 0)
                    ctx.line_to(screen_x, canvas_height)
                ctx.stroke()

        if self.current_snap_result.primary_snap_point:
            sp = self.current_snap_result.primary_snap_point
            sx, sy = to_screen.transform_point((sp.x, sp.y))
            if sp.line_type == SnapLineType.EQUIDISTANT and sp.spacing:
                style = SNAP_LINE_STYLES.get(sp.line_type, SnapLineStyle())
                ctx.set_source_rgba(*style.color)
                ctx.set_dash([])
                ctx.set_line_width(2.0)
                scale_x, _ = to_screen.get_scale()
                head_len = min(sp.spacing * scale_x * 0.15, 8)
                head_width = 4
                tick_len = 6

                coords = (
                    sp.pattern_coords
                    if sp.pattern_coords
                    else (sp.y if sp.is_horizontal else sp.x,)
                )

                def draw_double_arrow(
                    ctx, x1, y1, x2, y2, head_len, head_width
                ):
                    dx = x2 - x1
                    dy = y2 - y1
                    length = (dx * dx + dy * dy) ** 0.5
                    if length < 1e-6:
                        return
                    ux, uy = dx / length, dy / length
                    ctx.move_to(x1, y1)
                    ctx.line_to(x2, y2)
                    for px, py, direction in [(x1, y1, -1), (x2, y2, 1)]:
                        hx = px - direction * ux * head_len
                        hy = py - direction * uy * head_len
                        ctx.move_to(hx - uy * head_width, hy + ux * head_width)
                        ctx.line_to(px, py)
                        ctx.line_to(hx + uy * head_width, hy - ux * head_width)

                if sp.is_horizontal:
                    axis_x = (
                        sp.axis_coord if sp.axis_coord is not None else sp.x
                    )
                    for i in range(len(coords) - 1):
                        y1, y2 = coords[i], coords[i + 1]
                        if abs(y2 - y1 - sp.spacing) > 0.5:
                            continue
                        sy1 = to_screen.transform_point((axis_x, y1))[1]
                        sy2 = to_screen.transform_point((axis_x, y2))[1]
                        sx = to_screen.transform_point((axis_x, sp.y))[0]
                        draw_double_arrow(
                            ctx, sx, sy1, sx, sy2, head_len, head_width
                        )
                        ctx.move_to(sx - tick_len, sy1)
                        ctx.line_to(sx + tick_len, sy1)
                        ctx.move_to(sx - tick_len, sy2)
                        ctx.line_to(sx + tick_len, sy2)
                else:
                    axis_y = (
                        sp.axis_coord if sp.axis_coord is not None else sp.y
                    )
                    for i in range(len(coords) - 1):
                        x1, x2 = coords[i], coords[i + 1]
                        if abs(x2 - x1 - sp.spacing) > 0.5:
                            continue
                        sx1 = to_screen.transform_point((x1, axis_y))[0]
                        sx2 = to_screen.transform_point((x2, axis_y))[0]
                        sy = to_screen.transform_point((sp.x, axis_y))[1]
                        draw_double_arrow(
                            ctx, sx1, sy, sx2, sy, head_len, head_width
                        )
                        ctx.move_to(sx1, sy - tick_len)
                        ctx.line_to(sx1, sy + tick_len)
                        ctx.move_to(sx2, sy - tick_len)
                        ctx.line_to(sx2, sy + tick_len)
                ctx.stroke()
            elif sp.line_type == SnapLineType.MIDPOINT:
                if isinstance(sp.source, Line):
                    line = sp.source
                    p1 = self._safe_get_point(line.p1_idx)
                    p2 = self._safe_get_point(line.p2_idx)
                    if p1 and p2:
                        s1 = to_screen.transform_point((p1.x, p1.y))
                        s2 = to_screen.transform_point((p2.x, p2.y))
                        style = SNAP_LINE_STYLES.get(
                            sp.line_type, SnapLineStyle()
                        )
                        ctx.set_source_rgba(*style.color)
                        ctx.set_dash([])
                        ctx.set_line_width(1.5)
                        draw_symmetry_arrows(ctx, s1, s2)
                        ctx.stroke()
            elif sp.line_type == SnapLineType.ENTITY_POINT:
                style = SNAP_LINE_STYLES.get(sp.line_type, SnapLineStyle())
                ctx.set_source_rgba(*style.color)
                ctx.set_dash([])
                ctx.set_line_width(2.0)
                ctx.arc(sx, sy, 8, 0, 2 * 3.14159)
                ctx.stroke()
            elif sp.line_type == SnapLineType.ON_ENTITY:
                if sp.source is not None:
                    style = SNAP_LINE_STYLES.get(
                        SnapLineType.ON_ENTITY, SnapLineStyle()
                    )
                    ctx.save()
                    ctx.transform(cairo.Matrix(*to_screen.for_cairo()))
                    scale_x, _ = to_screen.get_scale()
                    scale = scale_x if scale_x > 1e-9 else 1.0
                    self.element.renderer.draw_entity_highlight(
                        ctx, sp.source, style.color, line_width=3.0 / scale
                    )
                    ctx.restore()
            else:
                ctx.set_source_rgba(1.0, 0.0, 1.0, 0.8)
                ctx.arc(sx, sy, 5, 0, 2 * 3.14159)
                ctx.fill()

        ctx.restore()

    def _draw_debug_snap_lines(self, ctx: cairo.Context) -> None:
        if not self.element.canvas:
            return

        to_screen = self.element.hittester.get_model_to_screen_transform(
            self.element
        )

        canvas_width = self.element.canvas.get_width()
        canvas_height = self.element.canvas.get_height()

        center_x = canvas_width / 2
        center_y = canvas_height / 2
        view_transform = self.element.canvas.view_transform
        model_center = view_transform.invert().transform_point(
            (center_x, center_y)
        )

        old_threshold = self.element.snap_engine.threshold
        self.element.snap_engine.threshold = 1e9
        snap_lines = self.element.snap_engine.get_visible_snap_lines(
            self.element.sketch.registry,
            model_center,
            DragContext(),
        )
        self.element.snap_engine.threshold = old_threshold

        print(f"DEBUG_SNAPPING: found {len(snap_lines)} snap lines")

        ctx.save()
        for snap_line in snap_lines:
            style = snap_line.style
            ctx.set_source_rgba(*style.color)
            if style.dash:
                ctx.set_dash(style.dash)
            else:
                ctx.set_dash([])
            ctx.set_line_width(style.line_width)

            if snap_line.is_horizontal:
                _, screen_y = to_screen.transform_point(
                    (0, snap_line.coordinate)
                )
                ctx.move_to(0, screen_y)
                ctx.line_to(canvas_width, screen_y)
            else:
                screen_x, _ = to_screen.transform_point(
                    (snap_line.coordinate, 0)
                )
                ctx.move_to(screen_x, 0)
                ctx.line_to(screen_x, canvas_height)
            ctx.stroke()

        ctx.restore()

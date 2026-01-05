import logging
import math
import cairo
from blinker import Signal
from typing import Tuple, List, Optional, TYPE_CHECKING, cast
from ...core.matrix import Matrix
from ...core.sketcher import Sketch
from ...core.sketcher.commands import (
    AddItemsCommand,
    RemoveItemsCommand,
    ToggleConstructionCommand,
    UnstickJunctionCommand,
    ChamferCommand,
    FilletCommand,
)
from ...core.sketcher.constraints import (
    PerpendicularConstraint,
    TangentConstraint,
    RadiusConstraint,
    DiameterConstraint,
    DistanceConstraint,
    HorizontalConstraint,
    VerticalConstraint,
    EqualDistanceConstraint,
    CoincidentConstraint,
    PointOnLineConstraint,
    EqualLengthConstraint,
    SymmetryConstraint,
    Constraint,
    CollinearConstraint,
)
from ...core.sketcher.entities import Line, Arc, Circle, Entity, Point
from ...core.sketcher.selection import SketchSelection
from ...core.sketcher.tools import (
    SelectTool,
    LineTool,
    ArcTool,
    CircleTool,
    FillTool,
    RoundedRectTool,
    RectangleTool,
)
from ..canvas import CanvasElement
from .hittest import SketchHitTester
from .renderer import SketchRenderer

if TYPE_CHECKING:
    from ...core.undo.command import Command
    from .sketchcanvas import SketchCanvas
    from .editor import SketchEditor

logger = logging.getLogger(__name__)


class SketchElement(CanvasElement):
    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        width: float = 1.0,
        height: float = 1.0,
        sketch: Optional[Sketch] = None,
        **kwargs,
    ):
        # Pass the required positional arguments to the parent class.
        super().__init__(
            x=x,
            y=y,
            width=width,
            height=height,
            is_editable=True,
            clip=False,
            **kwargs,
        )

        # Signals
        self.constraint_edit_requested = Signal()

        # Model
        self._sketch: Sketch

        # State Managers
        self.selection = SketchSelection()
        self.hittester = SketchHitTester()
        self.renderer = SketchRenderer(self)
        self.editor: Optional["SketchEditor"]

        # This must be set after self.selection is initialized
        self.sketch = sketch if sketch is not None else Sketch()

        # Tools
        self.tools = {
            "select": SelectTool(self),
            "line": LineTool(self),
            "arc": ArcTool(self),
            "circle": CircleTool(self),
            "fill": FillTool(self),
            "rounded_rect": RoundedRectTool(self),
            "rectangle": RectangleTool(self),
        }
        self.active_tool_name = "select"

        # Config
        self.point_radius = 5.0
        self.line_width = 2.0

    @property
    def sketch(self) -> Sketch:
        return self._sketch

    @sketch.setter
    def sketch(self, new_sketch: Sketch):
        logger.debug(f"Called for sketch '{new_sketch.name}'")
        # Disconnect from old sketch's signals if it exists
        if hasattr(self, "_sketch"):
            self._disconnect_signals()

        self._sketch = new_sketch

        # Connect to new sketch's signals
        self._connect_signals()

    def _connect_signals(self):
        """Connects to signals that indicate the model has changed."""
        self.sketch.updated.connect(self._on_model_changed)
        if self.sketch and self.sketch.input_parameters is not None:
            logger.debug(
                f"Connecting to VarSet signals on "
                f"{type(self.sketch.input_parameters).__name__} "
                f"(id: {id(self.sketch.input_parameters)})"
            )
            self.sketch.input_parameters.var_added.connect(
                self._on_model_changed
            )
            self.sketch.input_parameters.var_removed.connect(
                self._on_model_changed
            )
            self.sketch.input_parameters.var_value_changed.connect(
                self._on_model_changed
            )
            self.sketch.input_parameters.var_definition_changed.connect(
                self._on_model_changed
            )
            self.sketch.input_parameters.cleared.connect(
                self._on_model_changed
            )

    def _disconnect_signals(self):
        """Disconnects signals to prevent leaks."""
        self.sketch.updated.disconnect(self._on_model_changed)
        if self.sketch and self.sketch.input_parameters is not None:
            logger.debug(
                "Disconnecting from VarSet signals on "
                f"{type(self.sketch.input_parameters).__name__} "
                f"(id: {id(self.sketch.input_parameters)})"
            )
            try:
                self.sketch.input_parameters.var_added.disconnect(
                    self._on_model_changed
                )
                self.sketch.input_parameters.var_removed.disconnect(
                    self._on_model_changed
                )
                self.sketch.input_parameters.var_value_changed.disconnect(
                    self._on_model_changed
                )
                self.sketch.input_parameters.var_definition_changed.disconnect(
                    self._on_model_changed
                )
                self.sketch.input_parameters.cleared.disconnect(
                    self._on_model_changed
                )
            except Exception as e:
                logger.warning(
                    f"Error during signal disconnection (safe to ignore): {e}"
                )

    def _on_model_changed(self, sender, **kwargs):
        """
        Central handler for all model changes. Triggers a solve and redraw.
        """
        logger.debug(
            f"Triggered by {type(sender).__name__} (id: {id(sender)}) "
            f"with kwargs: {kwargs}. Solving and redrawing."
        )
        self.sketch.solve()
        self.update_bounds_from_sketch()
        self.mark_dirty()

    def remove(self):
        """Overrides remove to cleanup signal connections."""
        self._disconnect_signals()
        super().remove()

    @property
    def current_tool(self):
        return self.tools.get(self.active_tool_name, self.tools["select"])

    def execute_command(self, command: "Command"):
        """Executes a command via the history manager if available."""
        if self.editor:
            self.editor.history_manager.execute(command)

    def get_selected_elements(self) -> bool:
        """
        Helper method to check if any internal items (points, entities, etc.)
        are selected. Returns a boolean, not a list of elements.
        """
        sel = self.selection
        return bool(
            sel.point_ids
            or sel.entity_ids
            or sel.constraint_idx is not None
            or sel.junction_pid is not None
        )

    def unselect_all(self):
        """Clears the internal sketch selection."""
        self.selection.clear()
        self.mark_dirty()

    def update_bounds_from_sketch(self):
        """
        Calculates the bounding box of the sketch geometry and updates the
        element's size and transform. For empty sketches, it creates a
        minimum-sized box and centers the origin. For non-empty sketches,
        it shrinks to fit the geometry exactly.
        """
        # A sketch is considered "empty" for bounding purposes if it has no
        # entities and at most one point (which would be the origin).
        is_truly_empty = (
            len(self.sketch.registry.entities) == 0
            and len(self.sketch.registry.points) <= 1
        )

        new_width: float
        new_height: float
        new_offset_x: float
        new_offset_y: float

        if is_truly_empty:
            # Apply a minimum dimension for selectability and center the
            # origin.
            min_dim = 50.0
            new_width = min_dim
            new_height = min_dim
            new_offset_x = min_dim / 2.0
            new_offset_y = min_dim / 2.0
        else:
            # Calculate the precise bounding box of all geometry.
            geometry = self.sketch.to_geometry()
            if geometry.is_empty():
                # This case handles sketches with only points.
                if not self.sketch.registry.points:
                    min_x, max_x, min_y, max_y = 0, 0, 0, 0
                else:
                    xs = [p.x for p in self.sketch.registry.points]
                    ys = [p.y for p in self.sketch.registry.points]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
            else:
                min_x, min_y, max_x, max_y = geometry.rect()

            # The element size is exactly the geometry size. No padding.
            new_width = max_x - min_x
            new_height = max_y - min_y
            # The offset moves the geometry's top-left to the element's origin.
            new_offset_x = -min_x
            new_offset_y = -min_y

        # Calculate the change in offset needed to keep the content visually
        # stationary on the canvas during the bounds update.
        current_offset_x, current_offset_y = (
            self.content_transform.get_translation()
        )
        delta_x = new_offset_x - current_offset_x
        delta_y = new_offset_y - current_offset_y

        # Apply all the calculated updates.
        self.content_transform = Matrix.translation(new_offset_x, new_offset_y)
        self.width = new_width
        self.height = new_height

        # Update the element's main transform to counteract the content shift.
        self.set_transform(
            self.transform @ Matrix.translation(-delta_x, -delta_y)
        )

        self.mark_dirty()

    def on_edit_mode_leave(self):
        """Called when this element is no longer the Canvas's edit_context."""
        self.update_bounds_from_sketch()

    # =========================================================================
    # Rendering
    # =========================================================================

    def draw(self, ctx: cairo.Context):
        """Main draw entry point."""
        self.renderer.draw(ctx)

    def draw_edit_overlay(self, ctx: cairo.Context):
        """Draws constraints, points, and handles on top of the canvas."""
        self.renderer.draw_edit_overlay(ctx)
        # Allow the active tool to draw its own overlay (e.g. selection box)
        self.current_tool.draw_overlay(ctx)

    # =========================================================================
    # Input Handling (Routed to Tools)
    # =========================================================================

    def handle_edit_press(
        self, world_x: float, world_y: float, n_press: int = 1
    ) -> bool:
        return self.current_tool.on_press(world_x, world_y, n_press)

    def handle_edit_drag(self, world_dx: float, world_dy: float):
        self.current_tool.on_drag(world_dx, world_dy)

    def handle_edit_release(self, world_x: float, world_y: float):
        self.current_tool.on_release(world_x, world_y)

    def on_hover_motion(self, world_x: float, world_y: float):
        """Dispatches hover events to the currently active tool."""
        self.current_tool.on_hover_motion(world_x, world_y)

    # =========================================================================
    # Capabilities Querying
    # =========================================================================

    def get_lines_at_point(self, pid: int) -> List[Line]:
        """Returns a list of all Line entities connected to a point."""
        return [
            e
            for e in self.sketch.registry.entities
            if isinstance(e, Line) and pid in (e.p1_idx, e.p2_idx)
        ]

    def is_action_supported(self, action: str) -> bool:
        """
        Determines if a generic action is valid for the current selection.
        """
        has_points = len(self.selection.point_ids) > 0
        has_entities = len(self.selection.entity_ids) > 0
        has_constraints = self.selection.constraint_idx is not None
        has_junctions = self.selection.junction_pid is not None

        if action == "construction":
            # Can toggle construction on entities
            return has_entities

        if action == "delete":
            # Can delete almost anything selected
            return (
                has_points or has_entities or has_constraints or has_junctions
            )

        if action in ("chamfer", "fillet"):
            # Valid if exactly one junction is selected and it connects
            # exactly two lines.
            if self.selection.junction_pid is not None:
                lines_at_junction = self.get_lines_at_point(
                    self.selection.junction_pid
                )
                return len(lines_at_junction) == 2
            return False

        return False

    def is_constraint_supported(self, constraint_type: str) -> bool:
        """
        Determines if a specific constraint type can be applied to the
        current selection. Delegates to the backend Sketch model.
        """
        return self.sketch.supports_constraint(
            constraint_type,
            self.selection.point_ids,
            self.selection.entity_ids,
        )

    # =========================================================================
    # Command Actions
    # =========================================================================

    def set_tool(self, tool_name: str):
        if tool_name in self.tools and self.active_tool_name != tool_name:
            # Deactivate the old tool before switching to the new one.
            self.current_tool.on_deactivate()
            self.active_tool_name = tool_name
            self.mark_dirty()
            if self.canvas:
                canvas = cast("SketchCanvas", self.canvas)
                canvas.update_sketch_cursor()

    def toggle_construction_on_selection(self):
        """
        Toggles the construction flag for currently selected entities.
        If any selected entity is non-construction, all become construction.
        Otherwise, all become normal geometry.
        """
        if not self.selection.entity_ids or not self.editor:
            return

        cmd = ToggleConstructionCommand(
            self.sketch, _("Toggle Construction"), self.selection.entity_ids
        )
        self.execute_command(cmd)

    def _get_items_to_delete(
        self,
    ) -> Tuple[List[Point], List[Entity], List[Constraint]]:
        """
        Calculates the full set of items to be deleted based on the current
        selection, including dependent items.
        """
        to_delete_constraints: List[Constraint] = []
        to_delete_entity_ids = set(self.selection.entity_ids)
        to_delete_point_ids = set(self.selection.point_ids)

        # 1. Selected Constraints
        if self.selection.constraint_idx is not None:
            if self.sketch.constraints and (
                0
                <= self.selection.constraint_idx
                < len(self.sketch.constraints)
            ):
                to_delete_constraints.append(
                    self.sketch.constraints[self.selection.constraint_idx]
                )

        registry_entities = self.sketch.registry.entities or []
        entity_map = {e.id: e for e in registry_entities}

        # 2. Cascading: If points are deleted, find entities that use them
        for e in registry_entities:
            if e.id in to_delete_entity_ids:
                continue

            p_ids: List[int] = []
            if isinstance(e, Line):
                p_ids = [e.p1_idx, e.p2_idx]
            elif isinstance(e, Arc):
                p_ids = [e.start_idx, e.end_idx, e.center_idx]
            elif isinstance(e, Circle):
                p_ids = [e.center_idx, e.radius_pt_idx]

            # If any control point is marked for deletion, the entity must go
            if any(pid in to_delete_point_ids for pid in p_ids):
                to_delete_entity_ids.add(e.id)

        # 2.5. Cleanup Implicit Constraints for Deleted Entities (Arc geometry)
        # Arcs rely on EqualDistanceConstraint(c, s, c, e) not linked by ID.
        current_constraints = self.sketch.constraints or []
        for eid in to_delete_entity_ids:
            e = entity_map.get(eid)
            if isinstance(e, Arc):
                c, s, end = e.center_idx, e.start_idx, e.end_idx

                # Find constraints matching this geometry
                for constr in current_constraints:
                    if isinstance(constr, EqualDistanceConstraint):
                        # Check if constraint matches dist(c,s) == dist(c,end)
                        set1 = {constr.p1, constr.p2}
                        set2 = {constr.p3, constr.p4}

                        target1 = {c, s}
                        target2 = {c, end}

                        # Match either order
                        if (set1 == target1 and set2 == target2) or (
                            set1 == target2 and set2 == target1
                        ):
                            if constr not in to_delete_constraints:
                                to_delete_constraints.append(constr)

        # 3. Orphan Points: Find points in deleted entities not used by
        # remaining entities
        if to_delete_entity_ids:
            used_points_by_remaining = set()
            points_of_deleted_entities = set()

            for e in registry_entities:
                p_ids = []
                if isinstance(e, Line):
                    p_ids = [e.p1_idx, e.p2_idx]
                elif isinstance(e, Arc):
                    p_ids = [e.start_idx, e.end_idx, e.center_idx]
                elif isinstance(e, Circle):
                    p_ids = [e.center_idx, e.radius_pt_idx]

                if e.id in to_delete_entity_ids:
                    points_of_deleted_entities.update(p_ids)
                else:
                    used_points_by_remaining.update(p_ids)

            orphans = points_of_deleted_entities - used_points_by_remaining
            to_delete_point_ids.update(orphans)

        # 4. Cleanup Constraints (Dependencies)
        for constr in current_constraints:
            if constr in to_delete_constraints:
                continue

            should_remove = False

            # Check Point Dependencies
            points_in_constraint = []
            if isinstance(
                constr,
                (
                    DistanceConstraint,
                    HorizontalConstraint,
                    VerticalConstraint,
                    CoincidentConstraint,
                ),
            ):
                points_in_constraint = [constr.p1, constr.p2]
            elif isinstance(constr, EqualDistanceConstraint):
                points_in_constraint = [
                    constr.p1,
                    constr.p2,
                    constr.p3,
                    constr.p4,
                ]
            elif isinstance(constr, PointOnLineConstraint):
                points_in_constraint = [constr.point_id]
            elif isinstance(constr, SymmetryConstraint):
                points_in_constraint = [constr.p1, constr.p2]
                if constr.center is not None:
                    points_in_constraint.append(constr.center)
            elif isinstance(constr, CollinearConstraint):
                points_in_constraint = [constr.p1, constr.p2, constr.p3]

            if any(p in to_delete_point_ids for p in points_in_constraint):
                should_remove = True

            # Check Entity Dependencies
            if not should_remove:
                entities_in_constraint = []
                if isinstance(constr, PerpendicularConstraint):
                    entities_in_constraint = [constr.e1_id, constr.e2_id]
                elif isinstance(constr, TangentConstraint):
                    entities_in_constraint = [constr.line_id, constr.shape_id]
                elif isinstance(constr, RadiusConstraint):
                    entities_in_constraint = [constr.entity_id]
                elif isinstance(constr, DiameterConstraint):
                    entities_in_constraint = [constr.circle_id]
                elif isinstance(constr, PointOnLineConstraint):
                    entities_in_constraint = [constr.shape_id]
                elif isinstance(constr, EqualLengthConstraint):
                    entities_in_constraint = constr.entity_ids
                elif isinstance(constr, SymmetryConstraint):
                    if constr.axis is not None:
                        entities_in_constraint = [constr.axis]

                if any(
                    e in to_delete_entity_ids for e in entities_in_constraint
                ):
                    should_remove = True

            if should_remove and constr not in to_delete_constraints:
                to_delete_constraints.append(constr)

        # 5. Get actual objects from IDs
        final_points = [
            p
            for p in self.sketch.registry.points
            if p.id in to_delete_point_ids and not p.fixed
        ]
        final_entities = [
            e for e in registry_entities if e.id in to_delete_entity_ids
        ]

        return final_points, final_entities, to_delete_constraints

    def delete_selection(self) -> bool:
        """
        Robust deletion logic.
        """
        if not self.editor:
            return False

        # Handle "un-sticking" a junction point
        if self.selection.junction_pid is not None:
            cmd = UnstickJunctionCommand(
                self.sketch, self.selection.junction_pid
            )
            self.execute_command(cmd)
            self.selection.clear()
            return True

        (
            points_to_del,
            entities_to_del,
            constraints_to_del,
        ) = self._get_items_to_delete()

        did_work = bool(points_to_del or entities_to_del or constraints_to_del)
        if did_work:
            cmd = RemoveItemsCommand(
                self.sketch,
                _("Delete Selection"),
                points=points_to_del,
                entities=entities_to_del,
                constraints=constraints_to_del,
            )
            self.execute_command(cmd)
            self.selection.clear()

        return did_work

    def _get_two_points_from_selection(self) -> Optional[Tuple[Point, Point]]:
        """Helper to resolve 2 points from point list or line selection."""
        # Case A: 2 Points selected
        if len(self.selection.point_ids) == 2:
            p1 = self._get_point(self.selection.point_ids[0])
            p2 = self._get_point(self.selection.point_ids[1])
            if p1 and p2:
                return p1, p2

        # Case B: 1 Line selected
        if len(self.selection.entity_ids) == 1:
            eid = self.selection.entity_ids[0]
            e = self._get_entity_by_id(eid)
            if isinstance(e, Line):
                p1 = self._get_point(e.p1_idx)
                p2 = self._get_point(e.p2_idx)
                if p1 and p2:
                    return p1, p2

        return None

    def add_horizontal_constraint(self):
        if not self.is_constraint_supported("horiz"):
            logger.warning("Horizontal constraint not valid for selection.")
            return
        if not self.editor:
            return

        constraints_to_add = []

        # Case 1: Two points selected
        if (
            len(self.selection.point_ids) == 2
            and not self.selection.entity_ids
        ):
            p1_id, p2_id = self.selection.point_ids
            constraints_to_add.append(HorizontalConstraint(p1_id, p2_id))

        # Case 2: One or more lines selected
        elif (
            len(self.selection.entity_ids) > 0 and not self.selection.point_ids
        ):
            for eid in self.selection.entity_ids:
                e = self._get_entity_by_id(eid)
                if isinstance(e, Line):
                    constraints_to_add.append(
                        HorizontalConstraint(e.p1_idx, e.p2_idx)
                    )

        if constraints_to_add:
            cmd = AddItemsCommand(
                self.sketch,
                _("Add Horizontal Constraint"),
                constraints=constraints_to_add,
            )
            self.execute_command(cmd)

    def add_vertical_constraint(self):
        if not self.is_constraint_supported("vert"):
            logger.warning("Vertical constraint not valid for selection.")
            return
        if not self.editor:
            return

        constraints_to_add = []

        # Case 1: Two points selected
        if (
            len(self.selection.point_ids) == 2
            and not self.selection.entity_ids
        ):
            p1_id, p2_id = self.selection.point_ids
            constraints_to_add.append(VerticalConstraint(p1_id, p2_id))

        # Case 2: One or more lines selected
        elif (
            len(self.selection.entity_ids) > 0 and not self.selection.point_ids
        ):
            for eid in self.selection.entity_ids:
                e = self._get_entity_by_id(eid)
                if isinstance(e, Line):
                    constraints_to_add.append(
                        VerticalConstraint(e.p1_idx, e.p2_idx)
                    )

        if constraints_to_add:
            cmd = AddItemsCommand(
                self.sketch,
                _("Add Vertical Constraint"),
                constraints=constraints_to_add,
            )
            self.execute_command(cmd)

    def add_distance_constraint(self):
        if not self.is_constraint_supported("dist"):
            logger.warning("Distance constraint not valid for selection.")
            return

        points = self._get_two_points_from_selection()
        if points and self.editor:
            p1, p2 = points
            dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
            constr = DistanceConstraint(p1.id, p2.id, dist)
            cmd = AddItemsCommand(
                self.sketch, _("Add Distance Constraint"), constraints=[constr]
            )
            self.execute_command(cmd)
        else:
            logger.warning("Select 2 Points or 1 Line for Distance.")

    def add_radius_constraint(self):
        """Adds a radius constraint to a selected Arc or Circle."""
        if not self.is_constraint_supported("radius"):
            logger.warning("Radius constraint requires exactly 1 Arc/Circle.")
            return

        eid = self.selection.entity_ids[0]
        e = self._get_entity_by_id(eid)

        radius = 0.0
        if isinstance(e, Arc):
            s = self.sketch.registry.get_point(e.start_idx)
            c = self.sketch.registry.get_point(e.center_idx)
            if s and c:
                radius = math.hypot(s.x - c.x, s.y - c.y)
        elif isinstance(e, Circle):
            r_pt = self.sketch.registry.get_point(e.radius_pt_idx)
            c = self.sketch.registry.get_point(e.center_idx)
            if r_pt and c:
                radius = math.hypot(r_pt.x - c.x, r_pt.y - c.y)

        if radius > 0 and e and self.editor:
            constr = RadiusConstraint(e.id, radius)
            cmd = AddItemsCommand(
                self.sketch, _("Add Radius Constraint"), constraints=[constr]
            )
            self.execute_command(cmd)
        else:
            logger.warning("Could not add radius constraint.")

    def add_diameter_constraint(self):
        """Adds a diameter constraint to a selected Circle."""
        if not self.is_constraint_supported("diameter"):
            logger.warning("Diameter constraint requires exactly 1 Circle.")
            return

        eid = self.selection.entity_ids[0]
        e = self._get_entity_by_id(eid)

        if isinstance(e, Circle) and self.editor:
            c = self.sketch.registry.get_point(e.center_idx)
            r_pt = self.sketch.registry.get_point(e.radius_pt_idx)
            if c and r_pt:
                radius = math.hypot(r_pt.x - c.x, r_pt.y - c.y)
                constr = DiameterConstraint(e.id, radius * 2.0)
                cmd = AddItemsCommand(
                    self.sketch,
                    _("Add Diameter Constraint"),
                    constraints=[constr],
                )
                self.execute_command(cmd)
        else:
            logger.warning("Selected entity is not a Circle.")

    def add_alignment_constraint(self):
        """
        Adds a Coincident (Point-Point) or PointOnLine constraint based on
        the current selection.
        """
        if not self.editor:
            return

        if self.is_constraint_supported("coincident"):
            p1_id, p2_id = self.selection.point_ids
            constr = CoincidentConstraint(p1_id, p2_id)
            cmd = AddItemsCommand(
                self.sketch,
                _("Add Coincident Constraint"),
                constraints=[constr],
            )
            self.execute_command(cmd)
            return

        if self.is_constraint_supported("point_on_line"):
            # The support check guarantees we have 1 entity and 1 valid point
            sel_entity_id = self.selection.entity_ids[0]
            target_pid = self.selection.point_ids[0]

            constr = PointOnLineConstraint(target_pid, sel_entity_id)
            cmd = AddItemsCommand(
                self.sketch, _("Add Point On Shape"), constraints=[constr]
            )
            self.execute_command(cmd)
            return

        logger.warning(
            "For Align: Select 2 points, OR 1 line/arc/circle and 1 "
            "distinct point."
        )

    def add_perpendicular(self):
        if not self.is_constraint_supported("perp"):
            logger.warning(
                "Perpendicular constraint requires 2 compatible entities "
                "(lines, arcs, or circles)."
            )
            return

        if not self.editor:
            return

        e1_id = self.selection.entity_ids[0]
        e2_id = self.selection.entity_ids[1]

        constr = PerpendicularConstraint(e1_id, e2_id)
        cmd = AddItemsCommand(
            self.sketch,
            _("Add Perpendicular Constraint"),
            constraints=[constr],
        )
        self.execute_command(cmd)

    def add_tangent(self):
        if not self.is_constraint_supported("tangent"):
            logger.warning("Tangent: Select 1 Line and 1 Arc/Circle.")
            return

        sel_line = None
        sel_shape = None

        for eid in self.selection.entity_ids:
            e = self._get_entity_by_id(eid)
            if isinstance(e, Line):
                sel_line = e
            elif isinstance(e, (Arc, Circle)):
                sel_shape = e

        if sel_line and sel_shape and self.editor:
            constr = TangentConstraint(sel_line.id, sel_shape.id)
            cmd = AddItemsCommand(
                self.sketch, _("Add Tangent Constraint"), constraints=[constr]
            )
            self.execute_command(cmd)
        else:
            logger.warning("Select 1 Line and 1 Arc/Circle for Tangent.")

    def add_equal_constraint(self):
        """
        Adds or merges an equal length/radius constraint for the selected
        entities.
        """
        if not self.is_constraint_supported("equal") or not self.editor:
            logger.warning("Equal constraint requires 2+ Lines/Arcs/Circles.")
            return

        selected_ids = set(self.selection.entity_ids)
        existing_constraints_to_merge = []
        final_ids = set(selected_ids)

        # Find any existing equality constraints involving the selected
        # entities
        for constr in self.sketch.constraints:
            if isinstance(constr, EqualLengthConstraint):
                # If there's any overlap, this constraint needs to be merged
                if not selected_ids.isdisjoint(constr.entity_ids):
                    existing_constraints_to_merge.append(constr)
                    final_ids.update(constr.entity_ids)

        # Create commands to remove old constraints and add the new one
        remove_cmd = RemoveItemsCommand(
            self.sketch, "", constraints=existing_constraints_to_merge
        )
        new_constr = EqualLengthConstraint(list(final_ids))
        add_cmd = AddItemsCommand(
            self.sketch, _("Add Equal Constraint"), constraints=[new_constr]
        )

        # Execute as a single undoable action. This is a simplified composite.
        remove_cmd._do_execute()

        # Manually link the undo operations to make them atomic
        original_add_undo = add_cmd._do_undo

        def composite_undo():
            original_add_undo()
            remove_cmd._do_undo()

        add_cmd._do_undo = composite_undo
        self.execute_command(add_cmd)

    def add_symmetry_constraint(self):
        """
        Adds a symmetry constraint.
        Supported modes:
        1. 3 Points: The first point in selection is considered the Center.
        2. 2 Points + 1 Line: The line is the Axis.
        """
        if not self.is_constraint_supported("symmetry") or not self.editor:
            logger.warning(
                "Symmetry: Select 3 Points (1st is center) OR 2 Points + 1 "
                "Line."
            )
            return

        point_ids = self.selection.point_ids
        entity_ids = self.selection.entity_ids
        constr = None

        if len(point_ids) == 3 and not entity_ids:
            center = point_ids[0]
            p1 = point_ids[1]
            p2 = point_ids[2]
            constr = SymmetryConstraint(p1, p2, center=center)
        elif len(point_ids) == 2 and len(entity_ids) == 1:
            p1 = point_ids[0]
            p2 = point_ids[1]
            axis = entity_ids[0]
            constr = SymmetryConstraint(p1, p2, axis=axis)

        if constr:
            cmd = AddItemsCommand(
                self.sketch, _("Add Symmetry Constraint"), constraints=[constr]
            )
            self.execute_command(cmd)

    def add_chamfer_action(self):
        """Creates a chamfer at the selected corner junction."""
        DEFAULT_DISTANCE = 10.0
        if not self.is_action_supported("chamfer") or not self.editor:
            logger.warning("Chamfer requires a selected corner of two lines.")
            return

        corner_pid = self.selection.junction_pid
        if corner_pid is None:
            return

        lines_at_junction = self.get_lines_at_point(corner_pid)

        if len(lines_at_junction) != 2:
            return
        line1, line2 = lines_at_junction

        # Determine a sensible default chamfer size
        corner_pt = self._get_point(corner_pid)
        if not corner_pt:
            return

        other1_pid = (
            line1.p2_idx if line1.p1_idx == corner_pid else line1.p1_idx
        )
        other1_pt = self._get_point(other1_pid)
        if not other1_pt:
            return

        len1 = math.hypot(other1_pt.x - corner_pt.x, other1_pt.y - corner_pt.y)

        other2_pid = (
            line2.p2_idx if line2.p1_idx == corner_pid else line2.p1_idx
        )
        other2_pt = self._get_point(other2_pid)
        if not other2_pt:
            return

        len2 = math.hypot(other2_pt.x - corner_pt.x, other2_pt.y - corner_pt.y)

        chamfer_dist = min(DEFAULT_DISTANCE, len1 / 2.0, len2 / 2.0)
        if chamfer_dist < 1e-6:
            logger.warning("Lines are too short to create a chamfer.")
            return

        cmd = ChamferCommand(
            self.sketch, corner_pid, line1.id, line2.id, chamfer_dist
        )
        self.execute_command(cmd)

    def add_fillet_action(self):
        """Creates a fillet at the selected corner junction."""
        DEFAULT_RADIUS = 10.0
        if not self.is_action_supported("fillet") or not self.editor:
            logger.warning("Fillet requires a selected corner of two lines.")
            return

        corner_pid = self.selection.junction_pid
        if corner_pid is None:
            return

        lines_at_junction = self.get_lines_at_point(corner_pid)

        if len(lines_at_junction) != 2:
            return
        line1, line2 = lines_at_junction

        # Determine a sensible default fillet radius
        corner_pt = self._get_point(corner_pid)
        if not corner_pt:
            return

        other1_pid = (
            line1.p2_idx if line1.p1_idx == corner_pid else line1.p1_idx
        )
        other1_pt = self._get_point(other1_pid)
        if not other1_pt:
            return

        len1 = math.hypot(other1_pt.x - corner_pt.x, other1_pt.y - corner_pt.y)

        other2_pid = (
            line2.p2_idx if line2.p1_idx == corner_pid else line2.p1_idx
        )
        other2_pt = self._get_point(other2_pid)
        if not other2_pt:
            return

        len2 = math.hypot(other2_pt.x - corner_pt.x, other2_pt.y - corner_pt.y)

        # Calculate angle between lines to determine max radius
        v1 = (other1_pt.x - corner_pt.x, other1_pt.y - corner_pt.y)
        v2 = (other2_pt.x - corner_pt.x, other2_pt.y - corner_pt.y)
        len_v1 = math.hypot(v1[0], v1[1])
        len_v2 = math.hypot(v2[0], v2[1])
        u1 = (v1[0] / len_v1, v1[1] / len_v1)
        u2 = (v2[0] / len_v2, v2[1] / len_v2)
        dot = u1[0] * u2[0] + u1[1] * u2[1]
        angle = math.acos(max(-1.0, min(1.0, dot)))
        half_angle = angle / 2.0

        # Max radius is limited by the shorter line and the angle
        max_radius = min(len1, len2) * math.sin(half_angle)
        fillet_radius = min(DEFAULT_RADIUS, max_radius)

        if fillet_radius < 1e-6:
            logger.warning(
                "Lines are too short or angle too acute for fillet."
            )
            return

        cmd = FilletCommand(
            self.sketch, corner_pid, line1.id, line2.id, fillet_radius
        )
        self.execute_command(cmd)

    def _get_entity_by_id(self, eid: int) -> Optional[Entity]:
        return self.sketch.registry.get_entity(eid)

    def _get_point(self, pid: int) -> Optional[Point]:
        try:
            return self.sketch.registry.get_point(pid)
        except IndexError:
            return None

    def remove_point_if_unused(self, pid: Optional[int]) -> bool:
        """
        Removes a point from the registry if it's not part of any entity.

        Args:
            pid: The point ID to remove.

        Returns:
            True if the point was removed, False otherwise.
        """
        if pid is None:
            return False
        removed = self.sketch.remove_point_if_unused(pid)
        if removed:
            self.mark_dirty()
        return removed

    def mark_dirty(self, ancestors=False, recursive=False):
        super().mark_dirty(ancestors=ancestors, recursive=recursive)
        if self.canvas:
            self.canvas.queue_draw()

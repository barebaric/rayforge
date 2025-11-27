import logging
import math
import cairo

from blinker import Signal
from rayforge.workbench.canvas import CanvasElement
from rayforge.core.matrix import Matrix

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.entities import Line, Arc, Circle
from rayforge.core.sketcher.constraints import (
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
)

from .selection import SketchSelection
from .hittest import SketchHitTester
from .renderer import SketchRenderer
from .tools import SelectTool, LineTool, ArcTool, CircleTool

logger = logging.getLogger(__name__)


class SketchElement(CanvasElement):
    constraint_edit_requested = Signal()

    def __init__(self, x=0, y=0, width=1.0, height=1.0, **kwargs):
        kwargs["is_editable"] = True
        # Pass the required positional arguments to the parent class.
        super().__init__(x=x, y=y, width=width, height=height, **kwargs)

        # Model
        self.sketch = Sketch()

        # State Managers
        self.selection = SketchSelection()
        self.hittester = SketchHitTester()
        self.renderer = SketchRenderer(self)

        # Tools
        self.tools = {
            "select": SelectTool(self),
            "line": LineTool(self),
            "arc": ArcTool(self),
            "circle": CircleTool(self),
        }
        self.active_tool_name = "select"

        # Config
        self.point_radius = 5.0
        self.line_width = 2.0
        self.auto_bounds_padding = 40.0

    @property
    def current_tool(self):
        return self.tools.get(self.active_tool_name, self.tools["select"])

    def update_bounds_from_sketch(self):
        """
        Calculates the bounding box of the sketch content and updates the
        element's size and transform to encompass it without shifting the
        visual location of the content.
        """
        points = self.sketch.registry.points
        if not points:
            return

        # 1. Calculate Model Bounding Box
        pad = self.auto_bounds_padding
        xs = [p.x for p in points]
        ys = [p.y for p in points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 2. Determine view offset to keep content inside [0, w], [0, h]
        target_offset_x = pad - min_x
        target_offset_y = pad - min_y

        # 3. Calculate new dimensions
        new_w = (max_x - min_x) + 2 * pad
        new_h = (max_y - min_y) + 2 * pad

        # 4. Calculate change in offset to adjust parent transform
        current_offset_x, current_offset_y = (
            self.content_transform.get_translation()
        )
        delta_x = target_offset_x - current_offset_x
        delta_y = target_offset_y - current_offset_y

        # 5. Apply updates
        self.content_transform = Matrix.translation(
            target_offset_x, target_offset_y
        )
        self.width = max(new_w, 50)
        self.height = max(new_h, 50)

        # Update parent transform to counteract the content shift.
        self.set_transform(
            self.transform @ Matrix.translation(-delta_x, -delta_y)
        )

        self.mark_dirty()

    # =========================================================================
    # Rendering
    # =========================================================================

    def draw(self, ctx: cairo.Context):
        """Main draw entry point."""
        self.renderer.draw(ctx)

    def draw_edit_overlay(self, ctx: cairo.Context):
        """Draws constraints, points, and handles on top of the canvas."""
        self.renderer.draw_edit_overlay(ctx)

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
        if hasattr(self.current_tool, "on_hover_motion"):
            self.current_tool.on_hover_motion(world_x, world_y)

    # =========================================================================
    # Capabilities Querying
    # =========================================================================

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
        if tool_name in self.tools:
            self.active_tool_name = tool_name
            self.mark_dirty()

    def toggle_construction_on_selection(self):
        """
        Toggles the construction flag for currently selected entities.
        If any selected entity is non-construction, all become construction.
        Otherwise, all become normal geometry.
        """
        entities_to_modify = []
        for eid in self.selection.entity_ids:
            ent = self.sketch.registry.get_entity(eid)
            if ent:
                entities_to_modify.append(ent)

        if not entities_to_modify:
            return

        # Logic: If any selected entity is NOT construction, set all to
        # construction.
        # Otherwise (all are construction), set all to normal.
        has_normal = any(not e.construction for e in entities_to_modify)
        new_state = has_normal

        for e in entities_to_modify:
            e.construction = new_state

        self.mark_dirty()

    def _unstick_junction(self, pid: int) -> bool:
        """Separates entities at a shared point."""
        try:
            junction_pt = self.sketch.registry.get_point(pid)
        except IndexError:
            return False

        entities_at_junction = []
        for e in self.sketch.registry.entities:
            p_ids = []
            if isinstance(e, Line):
                p_ids = [e.p1_idx, e.p2_idx]
            elif isinstance(e, Arc):
                p_ids = [e.start_idx, e.end_idx, e.center_idx]
            elif isinstance(e, Circle):
                p_ids = [e.center_idx, e.radius_pt_idx]
            if pid in p_ids:
                entities_at_junction.append(e)

        if len(entities_at_junction) < 2:
            return False

        # Keep the first entity, modify the rest
        is_first = True
        for e in entities_at_junction:
            if is_first:
                is_first = False
                continue

            # Create a new point at the same location
            new_pid = self.sketch.add_point(junction_pt.x, junction_pt.y)

            # Re-assign the point in the entity
            if isinstance(e, Line):
                if e.p1_idx == pid:
                    e.p1_idx = new_pid
                if e.p2_idx == pid:
                    e.p2_idx = new_pid
            elif isinstance(e, Arc):
                if e.start_idx == pid:
                    e.start_idx = new_pid
                if e.end_idx == pid:
                    e.end_idx = new_pid
                if e.center_idx == pid:
                    e.center_idx = new_pid
            elif isinstance(e, Circle):
                if e.center_idx == pid:
                    e.center_idx = new_pid
                if e.radius_pt_idx == pid:
                    e.radius_pt_idx = new_pid
        return True

    def delete_selection(self) -> bool:
        """
        Robust deletion logic.
        """
        # Handle "un-sticking" a junction point
        if self.selection.junction_pid is not None:
            did_work = self._unstick_junction(self.selection.junction_pid)
            self.selection.clear()
            if did_work:
                self.sketch.solve()
                self.mark_dirty()
            return did_work

        to_delete_constraints = []
        to_delete_entities = set(self.selection.entity_ids)
        to_delete_points = set(self.selection.point_ids)

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
            if e.id in to_delete_entities:
                continue

            p_ids = []
            if isinstance(e, Line):
                p_ids = [e.p1_idx, e.p2_idx]
            elif isinstance(e, Arc):
                p_ids = [e.start_idx, e.end_idx, e.center_idx]
            elif isinstance(e, Circle):
                p_ids = [e.center_idx, e.radius_pt_idx]

            # If any control point is marked for deletion, the entity must go
            if any(pid in to_delete_points for pid in p_ids):
                to_delete_entities.add(e.id)

        # 2.5. Cleanup Implicit Constraints for Deleted Entities (Arc geometry)
        # Arcs rely on EqualDistanceConstraint(c, s, c, e) not linked by ID.
        current_constraints = self.sketch.constraints or []
        for eid in to_delete_entities:
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
        if to_delete_entities:
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

                if e.id in to_delete_entities:
                    points_of_deleted_entities.update(p_ids)
                else:
                    used_points_by_remaining.update(p_ids)

            orphans = points_of_deleted_entities - used_points_by_remaining
            to_delete_points.update(orphans)

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

            if any(p in to_delete_points for p in points_in_constraint):
                should_remove = True

            # Check Entity Dependencies
            if not should_remove:
                entities_in_constraint = []
                if isinstance(constr, PerpendicularConstraint):
                    entities_in_constraint = [constr.l1_id, constr.l2_id]
                elif isinstance(constr, TangentConstraint):
                    entities_in_constraint = [constr.line_id, constr.shape_id]
                elif isinstance(constr, RadiusConstraint):
                    entities_in_constraint = [constr.entity_id]
                elif isinstance(constr, DiameterConstraint):
                    entities_in_constraint = [constr.circle_id]
                elif isinstance(constr, PointOnLineConstraint):
                    entities_in_constraint = [constr.line_id]

                if any(
                    e in to_delete_entities for e in entities_in_constraint
                ):
                    should_remove = True

            if should_remove:
                to_delete_constraints.append(constr)

        # 5. Execute Deletion
        did_work = False

        # Remove Constraints
        for c in to_delete_constraints:
            if c in self.sketch.constraints:
                self.sketch.constraints.remove(c)
                did_work = True

        # Remove Entities
        if to_delete_entities:
            self.sketch.registry.entities = [
                e
                for e in self.sketch.registry.entities
                if e.id not in to_delete_entities
            ]
            self.sketch.registry._entity_map = {
                e.id: e for e in self.sketch.registry.entities
            }
            did_work = True

        # Remove Points
        if to_delete_points:
            self.sketch.registry.points = [
                p
                for p in self.sketch.registry.points
                if p.fixed or (p.id not in to_delete_points)
            ]
            did_work = True

        self.selection.clear()

        if did_work:
            try:
                self.sketch.solve()
            except Exception:
                pass
            self.mark_dirty()

        return did_work

    def _get_two_points_from_selection(self):
        """Helper to resolve 2 points from point list or line selection."""
        # Case A: 2 Points selected
        if len(self.selection.point_ids) == 2:
            p1 = self.sketch.registry.get_point(self.selection.point_ids[0])
            p2 = self.sketch.registry.get_point(self.selection.point_ids[1])
            return p1, p2

        # Case B: 1 Line selected
        if len(self.selection.entity_ids) == 1:
            eid = self.selection.entity_ids[0]
            e = self._get_entity_by_id(eid)
            if isinstance(e, Line):
                p1 = self.sketch.registry.get_point(e.p1_idx)
                p2 = self.sketch.registry.get_point(e.p2_idx)
                return p1, p2

        return None, None

    def add_horizontal_constraint(self):
        if not self.is_constraint_supported("horiz"):
            logger.warning("Horizontal constraint not valid for selection.")
            return

        p1, p2 = self._get_two_points_from_selection()
        if p1 and p2:
            self.sketch.constrain_horizontal(p1.id, p2.id)
            self.sketch.solve()
            self.mark_dirty()

    def add_vertical_constraint(self):
        if not self.is_constraint_supported("vert"):
            logger.warning("Vertical constraint not valid for selection.")
            return

        p1, p2 = self._get_two_points_from_selection()
        if p1 and p2:
            self.sketch.constrain_vertical(p1.id, p2.id)
            self.sketch.solve()
            self.mark_dirty()

    def add_distance_constraint(self):
        if not self.is_constraint_supported("dist"):
            logger.warning("Distance constraint not valid for selection.")
            return

        p1, p2 = self._get_two_points_from_selection()
        if p1 and p2:
            dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
            self.sketch.constrain_distance(p1.id, p2.id, dist)
            self.sketch.solve()
            self.mark_dirty()
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

        if radius > 0 and e:
            self.sketch.constrain_radius(e.id, radius)
            self.sketch.solve()
            self.mark_dirty()
        else:
            logger.warning("Could not add radius constraint.")

    def add_diameter_constraint(self):
        """Adds a diameter constraint to a selected Circle."""
        if not self.is_constraint_supported("diameter"):
            logger.warning("Diameter constraint requires exactly 1 Circle.")
            return

        eid = self.selection.entity_ids[0]
        e = self._get_entity_by_id(eid)

        if isinstance(e, Circle):
            c = self.sketch.registry.get_point(e.center_idx)
            r_pt = self.sketch.registry.get_point(e.radius_pt_idx)
            if c and r_pt:
                radius = math.hypot(r_pt.x - c.x, r_pt.y - c.y)
                self.sketch.constrain_diameter(e.id, radius * 2.0)
                self.sketch.solve()
                self.mark_dirty()
        else:
            logger.warning("Selected entity is not a Circle.")

    def add_alignment_constraint(self):
        """
        Adds a Coincident (Point-Point) or PointOnLine constraint based on
        the current selection.
        """
        if self.is_constraint_supported("coincident"):
            p1_id, p2_id = self.selection.point_ids
            self.sketch.constrain_coincident(p1_id, p2_id)
            self.sketch.solve()
            self.mark_dirty()
            return

        if self.is_constraint_supported("point_on_line"):
            # We need to find which is line and which is point
            selected_lines = [
                ent
                for eid in self.selection.entity_ids
                if isinstance((ent := self._get_entity_by_id(eid)), Line)
            ]
            sel_line = selected_lines[0]
            target_pid = self.selection.point_ids[0]

            # Ensure point is not one of the line's endpoints
            if target_pid not in {sel_line.p1_idx, sel_line.p2_idx}:
                self.sketch.constrain_point_on_line(target_pid, sel_line.id)
                self.sketch.solve()
                self.mark_dirty()
                return
            else:
                logger.warning(
                    "Point is already an endpoint of the selected line."
                )

        logger.warning(
            "For Align: Select 2 points, OR 1 line and 1 distinct point."
        )

    def add_perpendicular(self):
        if not self.is_constraint_supported("perp"):
            logger.warning("Perpendicular constraint requires 2 Lines.")
            return

        e1 = self._get_entity_by_id(self.selection.entity_ids[0])
        e2 = self._get_entity_by_id(self.selection.entity_ids[1])

        if e1 and e2:
            self.sketch.constraints.append(
                PerpendicularConstraint(e1.id, e2.id)
            )
            self.sketch.solve()
            self.mark_dirty()
        else:
            logger.warning("Perpendicular constraint requires 2 Lines.")

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

        if sel_line and sel_shape:
            self.sketch.constrain_tangent(sel_line.id, sel_shape.id)
            self.sketch.solve()
            self.mark_dirty()
        else:
            logger.warning("Select 1 Line and 1 Arc/Circle for Tangent.")

    def _get_entity_by_id(self, eid):
        return self.sketch.registry.get_entity(eid)

    def mark_dirty(self, ancestors=False, recursive=False):
        super().mark_dirty(ancestors=ancestors, recursive=recursive)
        if self.canvas:
            self.canvas.queue_draw()

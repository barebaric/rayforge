import cairo
import math
from collections import defaultdict
from ...core.geo.primitives import find_closest_point_on_line
from ...core.sketcher.entities import Line, Arc, Circle
from ...core.sketcher.constraints import (
    DistanceConstraint,
    RadiusConstraint,
    DiameterConstraint,
    HorizontalConstraint,
    VerticalConstraint,
    PerpendicularConstraint,
    TangentConstraint,
    CoincidentConstraint,
    PointOnLineConstraint,
    EqualLengthConstraint,
    SymmetryConstraint,
)
from ...core.sketcher.constraints.base import ConstraintStatus


class SketchRenderer:
    """Handles rendering of the sketch to a Cairo context."""

    def __init__(self, element):
        self.element = element

    def draw(self, ctx: cairo.Context):
        """Main draw entry point for sketch entities."""
        ctx.save()

        # Apply the Content Transform (Model -> Local)
        content_matrix = cairo.Matrix(
            *self.element.content_transform.for_cairo()
        )
        ctx.transform(content_matrix)

        # Calculate the inverse scale to maintain constant line width on
        # screen.
        scale = 1.0
        if self.element.canvas and hasattr(
            self.element.canvas, "get_view_scale"
        ):
            scale_x, _ = self.element.canvas.get_view_scale()
            scale = scale_x if scale_x > 1e-9 else 1.0

        scaled_line_width = self.element.line_width / scale

        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_width(scaled_line_width)

        # Check if the element is the active edit context on the canvas.
        is_editing = (
            self.element.canvas
            and self.element.canvas.edit_context is self.element
        )

        # Draw the Origin Icon (Underneath geometry) only when in edit mode.
        if is_editing:
            self._draw_origin(ctx)

        self._draw_fills(ctx)
        self._draw_entities(ctx, is_editing, scaled_line_width)
        ctx.restore()

    def draw_edit_overlay(self, ctx: cairo.Context):
        """Draws constraints, points, and handles on top of the canvas."""
        if not self.element.canvas:
            return

        to_screen = self.element.hittester.get_model_to_screen_transform(
            self.element
        )
        ctx.set_font_size(12)

        # Draw points first, so that constraint overlays are drawn on top.
        self._draw_points(ctx, to_screen)
        self._draw_overlays(ctx, to_screen)

    def _draw_origin(self, ctx: cairo.Context):
        """Draws a fixed symbol at (0,0)."""
        # The Origin is physically at 0,0 in Model Space
        scale = 1.0
        # Check if the host canvas supports get_view_scale
        if self.element.canvas and hasattr(
            self.element.canvas, "get_view_scale"
        ):
            scale_x, _ = self.element.canvas.get_view_scale()
            scale = scale_x if scale_x > 1e-9 else 1.0

        ctx.save()
        ctx.set_source_rgb(0.8, 0.2, 0.2)  # Reddish
        # Scale line width so it stays constant on screen
        ctx.set_line_width(2.0 / scale)

        len_ = 10.0 / scale
        ctx.move_to(-len_, 0)
        ctx.line_to(len_, 0)
        ctx.move_to(0, -len_)
        ctx.line_to(0, len_)
        ctx.stroke()

        # Circle
        ctx.arc(0, 0, 4.0 / scale, 0, 2 * math.pi)
        ctx.stroke()
        ctx.restore()

    def _draw_fills(self, ctx: cairo.Context):
        """Draws the filled regions of the sketch."""
        fill_geometries = self.element.sketch.get_fill_geometries()

        for geo in fill_geometries:
            ctx.new_path()
            geo.to_cairo(ctx)
            ctx.close_path()

            ctx.save()
            ctx.set_source_rgba(0.85, 0.85, 0.85, 0.7)
            ctx.fill()
            ctx.restore()

    # --- Entities ---

    def _draw_entities(
        self, ctx: cairo.Context, is_editing: bool, base_line_width: float
    ):
        is_sketch_fully_constrained = self.element.sketch.is_fully_constrained
        entities = self.element.sketch.registry.entities or []

        for entity in entities:
            # If not in edit mode, skip drawing construction geometry.
            if not is_editing and entity.construction:
                continue

            is_sel = entity.id in self.element.selection.entity_ids
            ctx.save()

            # 1. Define the Path
            has_path = False
            if isinstance(entity, Line):
                has_path = self._define_line_path(ctx, entity)
            elif isinstance(entity, Arc):
                has_path = self._define_arc_path(ctx, entity)
            elif isinstance(entity, Circle):
                has_path = self._define_circle_path(ctx, entity)

            if not has_path:
                ctx.restore()
                continue

            # 2. Draw Selection Underlay (Blurry Glow)
            if is_sel:
                ctx.save()
                # Remove dash pattern for the glow so it's solid
                ctx.set_dash([])
                # Semi-transparent blue for selection
                ctx.set_source_rgba(0.2, 0.6, 1.0, 0.4)
                # Thicker line for the glow effect
                ctx.set_line_width(base_line_width * 3.0)
                ctx.stroke_preserve()
                ctx.restore()

            # 3. Draw Actual Entity
            if entity.construction:
                # Calculate scale to convert screen pixels to world coords
                scale = self.element.line_width / base_line_width
                # Set dash pattern in screen pixels (5 on, 5 off)
                ctx.set_dash([5.0 / scale, 5.0 / scale])
                # Reduce width for construction lines
                ctx.set_line_width(base_line_width * 0.8)
                if entity.constrained:
                    ctx.set_source_rgb(0.2, 0.3, 0.6)  # Dark Blue
                else:
                    ctx.set_source_rgb(0.3, 0.5, 0.8)  # Light Blue
            else:
                self._set_standard_color(
                    ctx,
                    False,  # Selection handled by underlay
                    entity.constrained,
                    is_sketch_fully_constrained,
                )

            ctx.stroke()
            ctx.restore()

    def _set_standard_color(
        self,
        ctx: cairo.Context,
        is_selected: bool,
        is_constrained: bool,
        is_sketch_fully_constrained: bool,
    ):
        if is_selected:
            ctx.set_source_rgb(0.2, 0.6, 1.0)  # Blue
        elif is_constrained:
            if is_sketch_fully_constrained:
                ctx.set_source_rgb(0.0, 0.6, 0.0)  # Darker Green
            else:
                ctx.set_source_rgb(0.2, 0.8, 0.2)  # Light Green
        else:
            fg_rgba = self.element.canvas.get_color()
            ctx.set_source_rgb(fg_rgba.red, fg_rgba.green, fg_rgba.blue)

    def _safe_get_point(self, pid):
        try:
            return self.element.sketch.registry.get_point(pid)
        except IndexError:
            return None

    def _define_line_path(self, ctx: cairo.Context, line: Line) -> bool:
        """Defines the path for a line without stroking."""
        p1 = self._safe_get_point(line.p1_idx)
        p2 = self._safe_get_point(line.p2_idx)
        if p1 and p2:
            ctx.move_to(p1.x, p1.y)
            ctx.line_to(p2.x, p2.y)
            return True
        return False

    def _define_arc_path(self, ctx: cairo.Context, arc: Arc) -> bool:
        """Defines the path for an arc without stroking."""
        start = self._safe_get_point(arc.start_idx)
        end = self._safe_get_point(arc.end_idx)
        center = self._safe_get_point(arc.center_idx)
        if not (start and end and center):
            return False

        radius = math.hypot(start.x - center.x, start.y - center.y)
        start_a = math.atan2(start.y - center.y, start.x - center.x)
        end_a = math.atan2(end.y - center.y, end.x - center.x)

        ctx.new_sub_path()
        if arc.clockwise:
            ctx.arc_negative(center.x, center.y, radius, start_a, end_a)
        else:
            ctx.arc(center.x, center.y, radius, start_a, end_a)
        return True

    def _define_circle_path(self, ctx: cairo.Context, circle: Circle) -> bool:
        """Defines the path for a circle without stroking."""
        center = self._safe_get_point(circle.center_idx)
        radius_pt = self._safe_get_point(circle.radius_pt_idx)
        if not (center and radius_pt):
            return False

        radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
        ctx.new_sub_path()
        ctx.arc(center.x, center.y, radius, 0, 2 * math.pi)
        return True

    # --- Overlays (Constraints & Junctions) ---

    def _set_constraint_color(self, ctx, constraint, is_hovered):
        """
        Sets the standard drawing color for constraints based on hover and
        status.
        """
        if is_hovered:
            ctx.set_source_rgb(1.0, 0.8, 0.0)  # Yellow for hover
        elif constraint.status == ConstraintStatus.ERROR:
            ctx.set_source_rgb(1.0, 0.2, 0.2)  # Red for error
        elif constraint.status == ConstraintStatus.EXPRESSION_BASED:
            ctx.set_source_rgb(1.0, 0.6, 0.0)  # Orange for expression
        else:  # VALID
            ctx.set_source_rgb(0.0, 0.6, 0.0)  # Green for valid

    def _draw_selection_underlay(self, ctx, width_scale=3.0):
        """Draws a semi-transparent blue underlay for the current path."""
        ctx.save()
        ctx.set_source_rgba(0.2, 0.6, 1.0, 0.4)
        ctx.set_line_width(ctx.get_line_width() * width_scale)
        ctx.stroke_preserve()
        ctx.restore()

    def _format_constraint_value(self, constr):
        """Helper to format the value string for constraints."""
        # Always show the evaluated numeric value, which is kept up-to-date
        # by the solver, even for expressions.
        return f"{float(constr.value):.1f}"

    def _draw_overlays(self, ctx: cairo.Context, to_screen):
        # --- Stage 0: Get Hover State ---
        select_tool = self.element.tools.get("select")
        hovered_constraint_idx = (
            select_tool.hovered_constraint_idx if select_tool else None
        )

        # --- Stage 1: Collect Grouped Constraints (like Equality) ---
        equality_groups = {}  # Map entity_id -> group_id
        constraints = self.element.sketch.constraints or []
        for idx, constr in enumerate(constraints):
            if isinstance(constr, EqualLengthConstraint):
                for eid in constr.entity_ids:
                    equality_groups[eid] = idx

        # --- Stage 2: Draw Individual Constraints ---
        for idx, constr in enumerate(constraints):
            # Skip drawing EqualLengthConstraint here, it's handled below.
            if isinstance(constr, EqualLengthConstraint):
                continue

            is_sel = idx == self.element.selection.constraint_idx
            is_hovered = idx == hovered_constraint_idx

            if isinstance(constr, DistanceConstraint):
                self._draw_distance_constraint(
                    ctx, constr, is_sel, is_hovered, to_screen
                )
            elif isinstance(constr, (RadiusConstraint, DiameterConstraint)):
                self._draw_circular_constraint(
                    ctx, constr, is_sel, is_hovered, to_screen
                )
            elif isinstance(
                constr, (HorizontalConstraint, VerticalConstraint)
            ):
                self._draw_hv_constraint(
                    ctx, constr, is_sel, is_hovered, to_screen
                )
            elif isinstance(constr, PerpendicularConstraint):
                self._draw_perp_constraint(
                    ctx, constr, is_sel, is_hovered, to_screen
                )
            elif isinstance(constr, TangentConstraint):
                self._draw_tangent_constraint(
                    ctx, constr, is_sel, is_hovered, to_screen
                )
            elif isinstance(constr, CoincidentConstraint):
                origin_id = getattr(self.element.sketch, "origin_id", -1)
                pid_to_draw = constr.p1
                if constr.p1 == origin_id and origin_id != -1:
                    pid_to_draw = constr.p2
                self._draw_point_constraint(
                    ctx, constr, pid_to_draw, to_screen, is_sel, is_hovered
                )
            elif isinstance(constr, PointOnLineConstraint):
                self._draw_point_constraint(
                    ctx,
                    constr,
                    constr.point_id,
                    to_screen,
                    is_sel,
                    is_hovered,
                )
            elif isinstance(constr, SymmetryConstraint):
                self._draw_symmetry_constraint(
                    ctx, constr, is_sel, is_hovered, to_screen
                )

        # --- Stage 3: Draw Symbols on Entities from Collected Groups ---
        if equality_groups:
            for entity_id, group_id in equality_groups.items():
                entity = self.element.sketch.registry.get_entity(entity_id)
                if not entity:
                    continue

                is_sel = group_id == self.element.selection.constraint_idx
                is_hovered = group_id == hovered_constraint_idx
                self._draw_equality_symbol(
                    ctx, entity, group_id, is_sel, is_hovered, to_screen
                )

        # Draw implicit junction constraints
        self._draw_junctions(ctx, to_screen)

    def _draw_equality_symbol(
        self, ctx, entity, constraint_idx, is_selected, is_hovered, to_screen
    ):
        """
        Draws a larger, always-horizontal '=' symbol offset from the entity.
        """
        if not entity:
            return

        try:
            constr = self.element.sketch.constraints[constraint_idx]
        except (IndexError, TypeError):
            return

        # Use the constraint's own logic to find the anchor point
        pos = constr._get_symbol_pos(
            entity,
            self.element.sketch.registry,
            to_screen.transform_point,
            self.element,
        )
        if not pos:
            return

        sx, sy = pos
        ctx.save()

        # If selected, draw a blue background circle underlay
        if is_selected:
            ctx.set_source_rgba(0.2, 0.6, 1.0, 0.4)
            ctx.arc(sx, sy, 10, 0, 2 * math.pi)
            ctx.fill()

        self._set_constraint_color(ctx, constr, is_hovered)
        ctx.set_font_size(16)
        ext = ctx.text_extents("=")
        # Center the text on the calculated screen point
        ctx.move_to(sx - ext.width / 2, sy + ext.height / 2)
        ctx.show_text("=")
        ctx.restore()
        ctx.new_path()

    def _draw_junctions(self, ctx, to_screen):
        registry = self.element.sketch.registry
        select_tool = self.element.tools.get("select")
        hovered_junction_pid = (
            select_tool.hovered_junction_pid if select_tool else None
        )

        point_counts = defaultdict(int)
        for entity in registry.entities:
            if isinstance(entity, Line):
                point_counts[entity.p1_idx] += 1
                point_counts[entity.p2_idx] += 1
            elif isinstance(entity, Arc):
                point_counts[entity.start_idx] += 1
                point_counts[entity.end_idx] += 1
                point_counts[entity.center_idx] += 1
            elif isinstance(entity, Circle):
                point_counts[entity.center_idx] += 1
                point_counts[entity.radius_pt_idx] += 1

        for pid, count in point_counts.items():
            if count > 1:
                is_sel = pid == self.element.selection.junction_pid
                is_hovered = pid == hovered_junction_pid
                p = self._safe_get_point(pid)
                if p:
                    sx, sy = to_screen.transform_point((p.x, p.y))
                    ctx.save()
                    ctx.set_line_width(1.5)

                    radius = self.element.point_radius + 4
                    ctx.new_sub_path()
                    ctx.arc(sx, sy, radius, 0, 2 * math.pi)

                    if is_sel:
                        self._draw_selection_underlay(ctx)

                    # Junctions are always implicit, so we use slightly
                    # different colors
                    if is_hovered:
                        ctx.set_source_rgba(1.0, 0.6, 0.0, 0.9)
                    else:
                        ctx.set_source_rgba(0.0, 0.6, 0.0, 0.8)

                    ctx.stroke()
                    ctx.restore()

    def _draw_point_constraint(
        self, ctx, constr, pid, to_screen, is_selected, is_hovered
    ):
        p = self._safe_get_point(pid)
        if not p:
            return
        sx, sy = to_screen.transform_point((p.x, p.y))
        ctx.save()
        ctx.set_line_width(1.5)

        radius = self.element.point_radius + 4
        ctx.new_sub_path()
        ctx.arc(sx, sy, radius, 0, 2 * math.pi)

        if is_selected:
            self._draw_selection_underlay(ctx)

        self._set_constraint_color(ctx, constr, is_hovered)
        ctx.stroke()
        ctx.restore()

    def _draw_circular_constraint(
        self, ctx, constr, is_selected, is_hovered, to_screen
    ):
        pos_data = constr.get_label_pos(
            self.element.sketch.registry,
            to_screen.transform_point,
            self.element,
        )
        if not pos_data:
            return
        sx, sy, arc_mid_sx, arc_mid_sy = pos_data

        val_str = self._format_constraint_value(constr)
        if isinstance(constr, RadiusConstraint):
            label = f"R{val_str}"
        elif isinstance(constr, DiameterConstraint):
            label = f"Ø{val_str}"
        else:
            return

        ext = ctx.text_extents(label)

        ctx.save()
        # Set background color based on selection, hover, and status
        if is_selected:
            ctx.set_source_rgba(0.2, 0.6, 1.0, 0.4)  # Blue selection
        elif is_hovered:
            ctx.set_source_rgba(1.0, 0.95, 0.85, 0.9)  # Light yellow hover
        elif constr.status == ConstraintStatus.ERROR:
            ctx.set_source_rgba(1.0, 0.8, 0.8, 0.9)  # Light red background
        elif constr.status == ConstraintStatus.EXPRESSION_BASED:
            ctx.set_source_rgba(1.0, 0.9, 0.7, 0.9)  # Light orange background
        else:  # VALID
            ctx.set_source_rgba(1, 1, 1, 0.8)  # Default white background

        bg_x = sx - ext.width / 2 - 4
        bg_y = sy - ext.height / 2 - 4
        ctx.rectangle(bg_x, bg_y, ext.width + 8, ext.height + 8)
        ctx.fill()
        ctx.new_path()

        # Set text color based on status
        if constr.status == ConstraintStatus.ERROR:
            ctx.set_source_rgb(0.8, 0.0, 0.0)  # Red text for error
        else:
            ctx.set_source_rgb(0, 0, 0.5)  # Dark blue otherwise

        ctx.move_to(sx - ext.width / 2, sy + ext.height / 2 - 2)
        ctx.show_text(label)

        self._set_constraint_color(ctx, constr, is_hovered)
        ctx.set_line_width(1)
        ctx.set_dash([4, 4])
        ctx.move_to(sx, sy)
        ctx.line_to(arc_mid_sx, arc_mid_sy)
        ctx.stroke()
        ctx.restore()

    def _draw_distance_constraint(
        self, ctx, constr, is_selected, is_hovered, to_screen
    ):
        p1 = self._safe_get_point(constr.p1)
        p2 = self._safe_get_point(constr.p2)
        if not (p1 and p2):
            return

        s1 = to_screen.transform_point((p1.x, p1.y))
        s2 = to_screen.transform_point((p2.x, p2.y))
        mx, my = (s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2

        label = self._format_constraint_value(constr)
        ext = ctx.text_extents(label)

        ctx.save()
        # Set background color based on selection, hover, and status
        if is_selected:
            ctx.set_source_rgba(0.2, 0.6, 1.0, 0.4)  # Blue selection
        elif is_hovered:
            ctx.set_source_rgba(1.0, 0.95, 0.85, 0.9)  # Light yellow hover
        elif constr.status == ConstraintStatus.ERROR:
            ctx.set_source_rgba(1.0, 0.8, 0.8, 0.9)  # Light red background
        elif constr.status == ConstraintStatus.EXPRESSION_BASED:
            ctx.set_source_rgba(1.0, 0.9, 0.7, 0.9)  # Light orange background
        else:  # VALID
            ctx.set_source_rgba(1, 1, 1, 0.8)  # Default white background

        # Draw label background
        bg_x = mx - ext.width / 2 - 4
        bg_y = my - ext.height / 2 - 4
        ctx.rectangle(bg_x, bg_y, ext.width + 8, ext.height + 8)
        ctx.fill()
        ctx.new_path()

        # Set text color based on status
        if constr.status == ConstraintStatus.ERROR:
            ctx.set_source_rgb(0.8, 0.0, 0.0)  # Red text for error
        else:
            ctx.set_source_rgb(0, 0, 0.5)  # Dark blue otherwise

        ctx.move_to(mx - ext.width / 2, my + ext.height / 2 - 2)
        ctx.show_text(label)
        ctx.new_path()

        # Draw Dash Line - only if no solid line entity connects these points
        has_geometry = False
        entities = self.element.sketch.registry.entities or []
        for entity in entities:
            if isinstance(entity, Line):
                if {entity.p1_idx, entity.p2_idx} == {constr.p1, constr.p2}:
                    has_geometry = True
                    break

        if not has_geometry:
            self._set_constraint_color(ctx, constr, is_hovered)
            ctx.set_line_width(1)
            ctx.set_dash([4, 4])
            ctx.move_to(s1[0], s1[1])
            ctx.line_to(s2[0], s2[1])
            ctx.stroke()

        ctx.restore()

    def _draw_hv_constraint(
        self, ctx, constr, is_selected, is_hovered, to_screen
    ):
        p1 = self._safe_get_point(constr.p1)
        p2 = self._safe_get_point(constr.p2)
        if not (p1 and p2):
            return

        s1 = to_screen.transform_point((p1.x, p1.y))
        s2 = to_screen.transform_point((p2.x, p2.y))

        t_marker = 0.2
        mx = s1[0] + (s2[0] - s1[0]) * t_marker
        my = s1[1] + (s2[1] - s1[1]) * t_marker

        size = 8
        ctx.save()
        ctx.set_line_width(2)
        if isinstance(constr, HorizontalConstraint):
            ctx.move_to(mx - size, my - 10)
            ctx.line_to(mx + size, my - 10)
        else:
            ctx.move_to(mx + 10, my - size)
            ctx.line_to(mx + 10, my + size)

        if is_selected:
            self._draw_selection_underlay(ctx)

        self._set_constraint_color(ctx, constr, is_hovered)
        ctx.stroke()
        ctx.restore()

    def _draw_perp_constraint(
        self, ctx, constr, is_selected, is_hovered, to_screen
    ):
        data = constr.get_visuals(
            self.element.sketch.registry, to_screen.transform_point
        )
        if not data:
            return

        sx, sy, ang1, ang2 = data

        ctx.save()
        ctx.set_line_width(1.5)

        # If we have angles, it's the classic line-line perpendicular case
        if ang1 is not None and ang2 is not None:
            radius = 16.0
            diff = ang2 - ang1
            while diff <= -math.pi:
                diff += 2 * math.pi
            while diff > math.pi:
                diff -= 2 * math.pi

            ctx.new_sub_path()
            if diff > 0:
                ctx.arc(sx, sy, radius, ang1, ang2)
            else:
                ctx.arc_negative(sx, sy, radius, ang1, ang2)

            if is_selected:
                self._draw_selection_underlay(ctx)

            self._set_constraint_color(ctx, constr, is_hovered)
            ctx.stroke()

            # Dot
            mid = ang1 + diff / 2
            dx = sx + math.cos(mid) * radius * 0.6
            dy = sy + math.sin(mid) * radius * 0.6
            ctx.new_sub_path()
            ctx.arc(dx, dy, 2.0, 0, 2 * math.pi)
            ctx.fill()
        else:
            # For all other cases (line-arc, arc-arc), draw a box at anchor
            sz = 8.0
            ctx.new_sub_path()
            ctx.rectangle(sx - sz, sy - sz, sz * 2, sz * 2)

            if is_selected:
                self._draw_selection_underlay(ctx)

            self._set_constraint_color(ctx, constr, is_hovered)
            ctx.stroke()

        ctx.restore()

    def _draw_tangent_constraint(
        self, ctx, constr, is_selected, is_hovered, to_screen
    ):
        line = self.element.sketch.registry.get_entity(constr.line_id)
        shape = self.element.sketch.registry.get_entity(constr.shape_id)

        if not (line and shape):
            return
        p1 = self._safe_get_point(line.p1_idx)
        p2 = self._safe_get_point(line.p2_idx)
        center = self._safe_get_point(shape.center_idx)

        if not (p1 and p2 and center):
            return

        # Find closest point on infinite line from center (in model space)
        tangent_mx, tangent_my = find_closest_point_on_line(
            (p1.x, p1.y), (p2.x, p2.y), center.x, center.y
        )

        # Convert everything to screen space BEFORE calculating offset
        sx_tangent, sy_tangent = to_screen.transform_point(
            (tangent_mx, tangent_my)
        )
        sx_center, sy_center = to_screen.transform_point((center.x, center.y))

        # Calculate angle in screen space
        angle = math.atan2(sy_tangent - sy_center, sx_tangent - sx_center)

        # Apply fixed pixel offset in screen space
        offset = 15.0
        sx = sx_tangent + offset * math.cos(angle)
        sy = sy_tangent + offset * math.sin(angle)

        # Draw the symbol
        ctx.save()
        ctx.set_line_width(1.5)

        ctx.translate(sx, sy)
        # Rotate so the 'line' part is parallel to the tangent line.
        # The angle calculated is normal to the tangent line.
        ctx.rotate(angle + math.pi / 2.0)

        radius = 6.0

        # Draw arc part (a 120 degree arc segment)
        # In our rotated space, center is "below" (negative y).
        ctx.new_sub_path()
        ctx.arc(
            0,
            -radius,
            radius,
            math.pi / 2 - math.pi / 3,
            math.pi / 2 + math.pi / 3,
        )

        # Draw line part (a horizontal line at y=0)
        ctx.move_to(-radius * 1.2, 0)
        ctx.line_to(radius * 1.2, 0)

        if is_selected:
            self._draw_selection_underlay(ctx)

        self._set_constraint_color(ctx, constr, is_hovered)
        ctx.stroke()
        ctx.restore()

    def _draw_symmetry_constraint(
        self, ctx, constr, is_selected, is_hovered, to_screen
    ):
        p1 = self._safe_get_point(constr.p1)
        p2 = self._safe_get_point(constr.p2)
        if not (p1 and p2):
            return

        s1 = to_screen.transform_point((p1.x, p1.y))
        s2 = to_screen.transform_point((p2.x, p2.y))

        mx = (s1[0] + s2[0]) / 2.0
        my = (s1[1] + s2[1]) / 2.0

        angle = math.atan2(s2[1] - s1[1], s2[0] - s1[0])

        # Determine offset for the icons from the center
        offset = 12.0

        ctx.save()
        ctx.set_line_width(1.5)

        # Combine both markers into one path for efficient stroking
        ctx.new_sub_path()

        # Left marker ">"
        lx = mx - offset * math.cos(angle)
        ly = my - offset * math.sin(angle)

        ctx.save()
        ctx.translate(lx, ly)
        ctx.rotate(angle)
        # Draw ">" shape centered at 0,0
        # coords: (-3, -4) -> (3, 0) -> (-3, 4)
        ctx.move_to(-3, -4)
        ctx.line_to(3, 0)
        ctx.line_to(-3, 4)
        ctx.restore()

        # Draw right marker "<"
        # Position: move forward from midpoint along the line
        rx = mx + offset * math.cos(angle)
        ry = my + offset * math.sin(angle)

        ctx.save()
        ctx.translate(rx, ry)
        ctx.rotate(angle)
        # Draw "<" shape centered at 0,0
        # coords: (3, -4) -> (-3, 0) -> (3, 4)
        ctx.move_to(3, -4)
        ctx.line_to(-3, 0)
        ctx.line_to(3, 4)
        ctx.restore()

        if is_selected:
            self._draw_selection_underlay(ctx)

        self._set_constraint_color(ctx, constr, is_hovered)
        ctx.stroke()
        ctx.restore()

    # --- Points ---

    def _draw_points(self, ctx, to_screen):
        """Draws all sketch points, including selection highlights."""
        is_sketch_fully_constrained = self.element.sketch.is_fully_constrained
        points = self.element.sketch.registry.points or []
        origin_id = getattr(self.element.sketch, "origin_id", -1)
        hover_pid = self.element.tools["select"].hovered_point_id

        # Determine points that should be highlighted due to entity selection
        entity_points = set()
        for eid in self.element.selection.entity_ids:
            ent = self._get_entity_by_id(eid)
            if isinstance(ent, Line):
                entity_points.add(ent.p1_idx)
                entity_points.add(ent.p2_idx)
            elif isinstance(ent, Arc):
                entity_points.add(ent.start_idx)
                entity_points.add(ent.end_idx)
                entity_points.add(ent.center_idx)
            elif isinstance(ent, Circle):
                entity_points.add(ent.center_idx)
                entity_points.add(ent.radius_pt_idx)

        for p in points:
            sx, sy = to_screen.transform_point((p.x, p.y))

            is_hovered = p.id == hover_pid
            is_explicit_sel = p.id in self.element.selection.point_ids
            is_implicit_sel = p.id in entity_points

            # Handle origin point separately for selection/hover feedback
            if p.id == origin_id:
                if is_hovered or is_explicit_sel:
                    ctx.save()
                    if is_hovered:
                        ctx.set_source_rgba(1.0, 0.2, 0.2, 1.0)
                    else:  # Selected
                        ctx.set_source_rgba(0.2, 0.6, 1.0, 1.0)  # Blue
                    ctx.set_line_width(2.0)
                    ctx.arc(
                        sx, sy, self.element.point_radius * 1.5, 0, 2 * math.pi
                    )
                    ctx.stroke()
                    ctx.restore()
                continue  # Always skip drawing solid dot for origin

            r = self.element.point_radius

            # 1. Selection Glow Underlay
            if is_explicit_sel or is_implicit_sel:
                ctx.save()
                ctx.set_source_rgba(
                    0.2, 0.6, 1.0, 0.4
                )  # Semi-transparent blue
                ctx.arc(sx, sy, r + 4, 0, 2 * math.pi)
                ctx.fill()
                ctx.restore()

            # 2. Main Point (Hover or Standard Color)
            if is_hovered:
                ctx.set_source_rgba(1.0, 0.2, 0.2, 1.0)
            elif p.constrained:
                if is_sketch_fully_constrained:
                    ctx.set_source_rgba(0.0, 0.6, 0.0, 1.0)  # Darker Green
                else:
                    ctx.set_source_rgba(0.2, 0.8, 0.2, 1.0)  # Light Green
            else:
                ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)  # Black

            ctx.arc(sx, sy, r, 0, 2 * math.pi)
            ctx.fill()

    def _get_entity_by_id(self, eid):
        return self.element.sketch.registry.get_entity(eid)

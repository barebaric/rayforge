import cairo
import math
from collections import defaultdict
from rayforge.core.sketcher.entities import Line, Arc, Circle
from rayforge.core.sketcher.constraints import (
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
)


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

        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_width(self.element.line_width)

        # Check if the element is the active edit context on the canvas.
        is_editing = (
            self.element.canvas
            and self.element.canvas.edit_context is self.element
        )

        # Draw the Origin Icon (Underneath geometry) only when in edit mode.
        if is_editing:
            self._draw_origin(ctx)

        self._draw_entities(ctx, is_editing)
        ctx.restore()

    def draw_edit_overlay(self, ctx: cairo.Context):
        """Draws constraints, points, and handles on top of the canvas."""
        if not self.element.canvas:
            return

        to_screen = self.element.hittester.get_model_to_screen_transform(
            self.element
        )
        ctx.set_font_size(12)

        self._draw_overlays(ctx, to_screen)
        self._draw_points(ctx, to_screen)

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

    # --- Entities ---

    def _draw_entities(self, ctx: cairo.Context, is_editing: bool):
        entities = self.element.sketch.registry.entities or []
        for entity in entities:
            # If not in edit mode, skip drawing construction geometry.
            if not is_editing and entity.construction:
                continue

            is_sel = entity.id in self.element.selection.entity_ids
            ctx.save()

            if entity.construction:
                ctx.set_dash([5, 5])
                ctx.set_line_width(self.element.line_width * 0.8)
                if is_sel:
                    ctx.set_source_rgb(1.0, 0.6, 0.0)  # Orange
                elif entity.constrained:
                    ctx.set_source_rgb(0.2, 0.3, 0.6)  # Dark Blue
                else:
                    ctx.set_source_rgb(0.3, 0.5, 0.8)  # Light Blue
            else:
                self._set_standard_color(ctx, is_sel, entity.constrained)

            if isinstance(entity, Line):
                self._draw_line_entity(ctx, entity)
            elif isinstance(entity, Arc):
                self._draw_arc_entity(ctx, entity)
            elif isinstance(entity, Circle):
                self._draw_circle_entity(ctx, entity)

            ctx.restore()

    def _set_standard_color(
        self, ctx: cairo.Context, is_selected: bool, is_constrained: bool
    ):
        if is_selected:
            ctx.set_source_rgb(1.0, 0.6, 0.0)  # Orange
        elif is_constrained:
            ctx.set_source_rgb(0.2, 0.8, 0.2)  # Light Green
        else:
            ctx.set_source_rgb(0.0, 0.0, 0.0)  # Black

    def _safe_get_point(self, pid):
        try:
            return self.element.sketch.registry.get_point(pid)
        except IndexError:
            return None

    def _draw_line_entity(self, ctx: cairo.Context, line: Line):
        p1 = self._safe_get_point(line.p1_idx)
        p2 = self._safe_get_point(line.p2_idx)
        if p1 and p2:
            ctx.move_to(p1.x, p1.y)
            ctx.line_to(p2.x, p2.y)
            ctx.stroke()

    def _draw_arc_entity(self, ctx: cairo.Context, arc: Arc):
        start = self._safe_get_point(arc.start_idx)
        end = self._safe_get_point(arc.end_idx)
        center = self._safe_get_point(arc.center_idx)
        if not (start and end and center):
            return

        radius = math.hypot(start.x - center.x, start.y - center.y)
        start_a = math.atan2(start.y - center.y, start.x - center.x)
        end_a = math.atan2(end.y - center.y, end.x - center.x)

        ctx.new_sub_path()
        if arc.clockwise:
            ctx.arc_negative(center.x, center.y, radius, start_a, end_a)
        else:
            ctx.arc(center.x, center.y, radius, start_a, end_a)
        ctx.stroke()

    # --- Overlays (Constraints & Junctions) ---
    def _draw_circle_entity(self, ctx: cairo.Context, circle: Circle):
        center = self._safe_get_point(circle.center_idx)
        radius_pt = self._safe_get_point(circle.radius_pt_idx)
        if not (center and radius_pt):
            return

        radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
        ctx.new_sub_path()
        ctx.arc(center.x, center.y, radius, 0, 2 * math.pi)
        ctx.stroke()

    def _draw_overlays(self, ctx: cairo.Context, to_screen):
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

            if is_sel:
                ctx.set_source_rgb(1.0, 0.2, 0.2)
            else:
                ctx.set_source_rgb(0.0, 0.6, 0.0)

            if isinstance(constr, DistanceConstraint):
                self._draw_distance_constraint(ctx, constr, is_sel, to_screen)
            elif isinstance(constr, (RadiusConstraint, DiameterConstraint)):
                self._draw_circular_constraint(ctx, constr, is_sel, to_screen)
            elif isinstance(
                constr, (HorizontalConstraint, VerticalConstraint)
            ):
                self._draw_hv_constraint(ctx, constr, to_screen)
            elif isinstance(constr, PerpendicularConstraint):
                self._draw_perp_constraint(ctx, constr, is_sel, to_screen)
            elif isinstance(constr, TangentConstraint):
                self._draw_tangent_constraint(ctx, constr, to_screen)
            elif isinstance(constr, CoincidentConstraint):
                origin_id = getattr(self.element.sketch, "origin_id", -1)
                pid_to_draw = constr.p1
                if constr.p1 == origin_id and origin_id != -1:
                    pid_to_draw = constr.p2
                self._draw_point_constraint(
                    ctx, pid_to_draw, to_screen, is_sel
                )
            elif isinstance(constr, PointOnLineConstraint):
                self._draw_point_constraint(
                    ctx, constr.point_id, to_screen, is_sel
                )

        # --- Stage 3: Draw Symbols on Entities from Collected Groups ---
        if equality_groups:
            for entity_id, group_id in equality_groups.items():
                entity = self.element.sketch.registry.get_entity(entity_id)
                if not entity:
                    continue

                is_sel = group_id == self.element.selection.constraint_idx
                if is_sel:
                    ctx.set_source_rgb(1.0, 0.2, 0.2)
                else:
                    ctx.set_source_rgb(0.0, 0.6, 0.0)
                self._draw_equality_symbol(ctx, entity, to_screen)

        # Draw implicit junction constraints
        self._draw_junctions(ctx, to_screen)

    def _draw_equality_symbol(self, ctx, entity, to_screen):
        """Draws an '=' symbol on a single entity."""
        if not entity:
            return
        mid_x, mid_y, angle = 0.0, 0.0, 0.0

        if isinstance(entity, Line):
            p1 = self._safe_get_point(entity.p1_idx)
            p2 = self._safe_get_point(entity.p2_idx)
            if not (p1 and p2):
                return
            mid_x, mid_y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
            angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        elif isinstance(entity, Arc):
            center = self._safe_get_point(entity.center_idx)
            start = self._safe_get_point(entity.start_idx)
            end = self._safe_get_point(entity.end_idx)
            if not (center and start and end):
                return
            start_a = math.atan2(start.y - center.y, start.x - center.x)
            end_a = math.atan2(end.y - center.y, end.x - center.x)
            angle_range = end_a - start_a
            if entity.clockwise:
                if angle_range > 0:
                    angle_range -= 2 * math.pi
            else:
                if angle_range < 0:
                    angle_range += 2 * math.pi
            mid_angle = start_a + angle_range / 2.0
            radius = math.hypot(start.x - center.x, start.y - center.y)
            mid_x = center.x + radius * math.cos(mid_angle)
            mid_y = center.y + radius * math.sin(mid_angle)
            angle = mid_angle + math.pi / 2
        elif isinstance(entity, Circle):
            center = self._safe_get_point(entity.center_idx)
            radius_pt = self._safe_get_point(entity.radius_pt_idx)
            if not (center and radius_pt):
                return
            radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
            angle = math.atan2(radius_pt.y - center.y, radius_pt.x - center.x)
            mid_x = center.x + radius * math.cos(angle)
            mid_y = center.y + radius * math.sin(angle)
            angle += math.pi / 2

        sx, sy = to_screen.transform_point((mid_x, mid_y))
        ctx.save()
        ctx.translate(sx, sy)
        ctx.rotate(angle)
        ctx.set_font_size(14)
        ext = ctx.text_extents("=")
        ctx.move_to(-ext.width / 2, ext.height / 2)
        ctx.show_text("=")
        ctx.restore()
        ctx.new_path()

    def _draw_junctions(self, ctx, to_screen):
        registry = self.element.sketch.registry
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
                p = self._safe_get_point(pid)
                if p:
                    sx, sy = to_screen.transform_point((p.x, p.y))
                    ctx.save()
                    ctx.set_line_width(1.5)
                    if is_sel:
                        ctx.set_source_rgba(1.0, 0.2, 0.2, 0.9)
                    else:
                        ctx.set_source_rgba(0.0, 0.6, 0.0, 0.8)
                    radius = self.element.point_radius + 4
                    ctx.arc(sx, sy, radius, 0, 2 * math.pi)
                    ctx.stroke()
                    ctx.restore()

    def _draw_point_constraint(self, ctx, pid, to_screen, is_selected):
        p = self._safe_get_point(pid)
        if not p:
            return
        sx, sy = to_screen.transform_point((p.x, p.y))
        ctx.save()
        ctx.set_line_width(1.5)
        if is_selected:
            ctx.set_source_rgba(1.0, 0.2, 0.2, 0.9)
        else:
            ctx.set_source_rgba(0.0, 0.6, 0.0, 0.8)

        radius = self.element.point_radius + 4
        ctx.arc(sx, sy, radius, 0, 2 * math.pi)
        ctx.stroke()
        ctx.restore()

    def _draw_circular_constraint(self, ctx, constr, is_selected, to_screen):
        pos_data = self.element.hittester.get_circular_label_pos(
            constr, to_screen, self.element
        )
        if not pos_data:
            return
        sx, sy, arc_mid_sx, arc_mid_sy = pos_data

        if isinstance(constr, RadiusConstraint):
            label = f"R{float(constr.value):.1f}"
        elif isinstance(constr, DiameterConstraint):
            label = f"Ø{float(constr.value):.1f}"
        else:
            return

        ext = ctx.text_extents(label)

        ctx.save()
        if is_selected:
            ctx.set_source_rgba(1, 0.9, 0.9, 0.9)
        else:
            ctx.set_source_rgba(1, 1, 1, 0.8)

        bg_x = sx - ext.width / 2 - 4
        bg_y = sy - ext.height / 2 - 4
        ctx.rectangle(bg_x, bg_y, ext.width + 8, ext.height + 8)
        ctx.fill()
        ctx.new_path()

        ctx.set_source_rgb(0, 0, 0.5)
        ctx.move_to(sx - ext.width / 2, sy + ext.height / 2 - 2)
        ctx.show_text(label)

        ctx.set_source_rgba(0.5, 0.5, 0.5, 0.5)
        ctx.set_line_width(1)
        ctx.set_dash([4, 4])
        ctx.move_to(sx, sy)
        ctx.line_to(arc_mid_sx, arc_mid_sy)
        ctx.stroke()
        ctx.restore()

    def _draw_distance_constraint(self, ctx, constr, is_selected, to_screen):
        p1 = self._safe_get_point(constr.p1)
        p2 = self._safe_get_point(constr.p2)
        if not (p1 and p2):
            return

        s1 = to_screen.transform_point((p1.x, p1.y))
        s2 = to_screen.transform_point((p2.x, p2.y))
        mx, my = (s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2

        label = f"{float(constr.value):.1f}"
        ext = ctx.text_extents(label)

        ctx.save()
        if is_selected:
            ctx.set_source_rgba(1, 0.9, 0.9, 0.9)
        else:
            ctx.set_source_rgba(1, 1, 1, 0.8)

        # Draw label background
        bg_x = mx - ext.width / 2 - 4
        bg_y = my - ext.height / 2 - 4
        ctx.rectangle(bg_x, bg_y, ext.width + 8, ext.height + 8)
        ctx.fill()
        ctx.new_path()

        # Draw Text
        ctx.set_source_rgb(0, 0, 0.5)
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
            ctx.set_source_rgba(0.5, 0.5, 0.5, 0.5)
            ctx.set_line_width(1)
            ctx.set_dash([4, 4])
            ctx.move_to(s1[0], s1[1])
            ctx.line_to(s2[0], s2[1])
            ctx.stroke()

        ctx.restore()

    def _draw_hv_constraint(self, ctx, constr, to_screen):
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
        ctx.stroke()
        ctx.restore()

    def _draw_perp_constraint(self, ctx, constr, is_selected, to_screen):
        data = self.element.hittester.get_perp_intersection_screen(
            constr, to_screen, self.element
        )
        if not data:
            return
        sx, sy, ang1, ang2 = data

        ctx.save()
        if is_selected:
            ctx.set_source_rgb(1.0, 0.2, 0.2)
        else:
            ctx.set_source_rgb(0.0, 0.6, 0.0)
        ctx.set_line_width(1.5)

        radius = 16.0
        diff = ang2 - ang1
        while diff <= -math.pi:
            diff += 2 * math.pi
        while diff > math.pi:
            diff -= 2 * math.pi

        if diff > 0:
            ctx.arc(sx, sy, radius, ang1, ang2)
        else:
            ctx.arc_negative(sx, sy, radius, ang1, ang2)
        ctx.stroke()

        # Dot
        mid = ang1 + diff / 2
        dx = sx + math.cos(mid) * radius * 0.6
        dy = sy + math.sin(mid) * radius * 0.6
        ctx.new_sub_path()
        ctx.arc(dx, dy, 2.0, 0, 2 * math.pi)
        ctx.fill()
        ctx.restore()

    def _draw_tangent_constraint(self, ctx, constr, to_screen):
        line = self.element.sketch.registry.get_entity(constr.line_id)
        shape = self.element.sketch.registry.get_entity(constr.shape_id)
        if not (isinstance(line, Line) and isinstance(shape, (Arc, Circle))):
            return
        p = self._safe_get_point(line.p1_idx)
        if p:
            sx, sy = to_screen.transform_point((p.x, p.y))
            ctx.move_to(sx + 10, sy + 10)
            ctx.show_text("⦸")

    # --- Points ---

    def _draw_points(self, ctx, to_screen):
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
                        ctx.set_source_rgba(1.0, 0.6, 0.0, 1.0)
                    ctx.set_line_width(2.0)
                    ctx.arc(
                        sx, sy, self.element.point_radius * 1.5, 0, 2 * math.pi
                    )
                    ctx.stroke()
                    ctx.restore()
                continue  # Always skip drawing solid dot for origin

            if is_hovered:
                ctx.set_source_rgba(1.0, 0.2, 0.2, 1.0)
                r = self.element.point_radius * 1.5
            elif is_explicit_sel or is_implicit_sel:
                ctx.set_source_rgba(1.0, 0.6, 0.0, 1.0)  # Orange
                r = self.element.point_radius * 1.2
            elif p.constrained:
                ctx.set_source_rgba(0.2, 0.8, 0.2, 1.0)  # Light Green
                r = self.element.point_radius
            else:
                ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)  # Black
                r = self.element.point_radius

            ctx.arc(sx, sy, r, 0, 2 * math.pi)
            ctx.fill()

    def _get_entity_by_id(self, eid):
        return self.element.sketch.registry.get_entity(eid)

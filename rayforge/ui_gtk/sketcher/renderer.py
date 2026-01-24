import cairo
import math
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, Set, Tuple

from ...core.geo import Geometry
from ...core.matrix import Matrix
from ...core.sketcher.constraints import (
    CoincidentConstraint,
    PointOnLineConstraint,
)
from ...core.sketcher.entities import (
    Arc,
    Circle,
    Entity,
    Line,
    Point,
    TextBoxEntity,
)
from ...core.sketcher.tools import TextBoxTool
from ..canvas import WorldSurface

if TYPE_CHECKING:
    from .sketchelement import SketchElement


class SketchRenderer:
    """Handles rendering of the sketch to a Cairo context."""

    def __init__(self, element: "SketchElement") -> None:
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
        if isinstance(self.element.canvas, WorldSurface):
            scale_x, _ = self.element.canvas.get_view_scale()
            scale = scale_x if scale_x > 1e-9 else 1.0

        scaled_line_width = self.element.line_width / scale

        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_width(scaled_line_width)

        # Check if the element is the active edit context on the canvas.
        is_editing = bool(
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

        ctx.set_font_size(12)

        # Draw points first, so that constraint overlays are drawn on top.
        to_screen = self.element.hittester.get_model_to_screen_transform(
            self.element
        )
        self._draw_points(ctx, to_screen)
        self._draw_overlays(ctx)

    def _draw_origin(self, ctx: cairo.Context):
        """Draws a fixed symbol at (0,0)."""
        # The Origin is physically at 0,0 in Model Space
        scale = 1.0
        # Check if the host canvas supports get_view_scale
        if self.element.canvas:
            get_view_scale = getattr(
                self.element.canvas, "get_view_scale", None
            )
            if get_view_scale:
                scale_x, _ = get_view_scale()
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
        # Identify the text box being edited to exclude its fill
        exclude_ids = set()
        text_tool = self.element.tools.get("text_box")
        if (
            self.element.active_tool_name == "text_box"
            and isinstance(text_tool, TextBoxTool)
            and text_tool.editing_entity_id is not None
        ):
            exclude_ids.add(text_tool.editing_entity_id)

        fill_geometries = self.element.sketch.get_fill_geometries(
            exclude_ids=exclude_ids
        )

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
        text_tool = self.element.tools.get("text_box")

        for entity in entities:
            # If a text box is being actively edited, its tool overlay will
            # draw it, so we skip the main render pass to avoid flicker.
            is_being_edited = (
                isinstance(entity, TextBoxEntity)
                and self.element.active_tool_name == "text_box"
                and isinstance(text_tool, TextBoxTool)
                and text_tool.editing_entity_id == entity.id
            )
            if is_being_edited:
                continue

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
            elif isinstance(entity, TextBoxEntity):
                has_path = self._define_text_box_path(ctx, entity)

            if not has_path:
                ctx.restore()
                continue

            # 2. Draw Selection Underlay (Blurry Glow)
            if is_sel:
                ctx.save()
                ctx.set_dash([])
                ctx.set_source_rgba(0.2, 0.6, 1.0, 0.4)
                if isinstance(entity, TextBoxEntity):
                    ctx.set_line_width(base_line_width * 2.0)
                else:
                    ctx.set_line_width(base_line_width * 3.0)
                ctx.stroke_preserve()
                ctx.restore()

            # 3. Draw Actual Entity
            if isinstance(entity, TextBoxEntity):
                # The selection glow is a stroke. To make the text readable,
                # we fill it with its standard color, not the selection color.
                self._set_standard_color(
                    ctx,
                    False,  # Selection handled by the glow underlay
                    entity.constrained,
                    is_sketch_fully_constrained,
                )
                ctx.fill()
            elif entity.construction:
                scale = self.element.line_width / base_line_width
                ctx.set_dash([5.0 / scale, 5.0 / scale])
                ctx.set_line_width(base_line_width * 0.8)
                if entity.constrained:
                    ctx.set_source_rgb(0.2, 0.3, 0.6)  # Dark Blue
                else:
                    ctx.set_source_rgb(0.3, 0.5, 0.8)  # Light Blue
                ctx.stroke()
            else:
                self._set_standard_color(
                    ctx,
                    is_sel,
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
            if self.element.canvas:
                fg_rgba = self.element.canvas.get_color()
                ctx.set_source_rgb(fg_rgba.red, fg_rgba.green, fg_rgba.blue)
            else:
                ctx.set_source_rgb(0.0, 0.0, 0.0)

    def _safe_get_point(self, pid: int) -> Optional[Point]:
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

    def _define_text_box_path(
        self, ctx: cairo.Context, entity: TextBoxEntity
    ) -> bool:
        if not entity.content:
            return False

        p_origin = self._safe_get_point(entity.origin_id)
        p_width = self._safe_get_point(entity.width_id)
        p_height = self._safe_get_point(entity.height_id)

        if not (p_origin and p_width and p_height):
            return False

        natural_geo = Geometry.from_text(entity.content, entity.font_config)
        natural_geo.flip_y()

        _, descent, font_height = entity.get_font_metrics()

        transformed_geo = natural_geo.map_to_frame(
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p_height.x, p_height.y),
            anchor_y=-descent,
            stable_src_height=font_height,
        )

        transformed_geo.to_cairo(ctx)
        return True

    # --- Overlays (Constraints & Junctions) ---

    def _draw_overlays(self, ctx: cairo.Context):
        # --- Stage 0: Get Hover State ---
        select_tool = self.element.tools.get("select")
        hovered_constraint_idx = (
            select_tool.hovered_constraint_idx if select_tool else None
        )

        # Collect all points associated with text boxes to hide their overlays
        text_box_point_ids = set()
        for entity in self.element.sketch.registry.entities:
            if isinstance(entity, TextBoxEntity):
                text_box_point_ids.update(
                    entity.get_all_frame_point_ids(
                        self.element.sketch.registry
                    )
                )

        to_screen_transform = (
            self.element.hittester.get_model_to_screen_transform(self.element)
        )
        to_screen_func = to_screen_transform.transform_point

        # --- Stage 1: Draw Individual Constraints ---
        constraints = self.element.sketch.constraints or []
        for idx, constr in enumerate(constraints):
            if not constr.user_visible:
                continue

            # Filter specific constraints on text box points to reduce clutter
            if isinstance(
                constr, (CoincidentConstraint, PointOnLineConstraint)
            ):
                if constr.depends_on_points(text_box_point_ids):
                    continue

            is_sel = idx == self.element.selection.constraint_idx
            is_hovered = idx == hovered_constraint_idx

            constr.draw(
                ctx,
                self.element.sketch.registry,
                to_screen_func,
                is_sel,
                is_hovered,
                point_radius=self.element.point_radius,
            )

        # Draw implicit junction constraints
        self._draw_junctions(ctx, to_screen_func, text_box_point_ids)

    def _draw_junctions(
        self,
        ctx: cairo.Context,
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        text_box_point_ids: Set[int],
    ) -> None:
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
                # Hide junction visuals for text box points
                if pid in text_box_point_ids:
                    continue

                is_sel = pid == self.element.selection.junction_pid
                is_hovered = pid == hovered_junction_pid
                p = self._safe_get_point(pid)
                if p:
                    sx, sy = to_screen((p.x, p.y))
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

    def _draw_selection_underlay(
        self, ctx: cairo.Context, width_scale: float = 3.0
    ) -> None:
        """Draws a semi-transparent blue underlay for the current path."""
        ctx.save()
        ctx.set_source_rgba(0.2, 0.6, 1.0, 0.4)
        ctx.set_line_width(ctx.get_line_width() * width_scale)
        ctx.stroke_preserve()
        ctx.restore()

    def _get_entity_by_id(self, eid: int) -> Optional[Entity]:
        return self.element.sketch.registry.get_entity(eid)

    # --- Points ---

    def _draw_points(
        self, ctx: cairo.Context, to_screen_matrix: Matrix
    ) -> None:
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
            elif isinstance(ent, TextBoxEntity):
                # Highlight the points if the text box is selected
                entity_points.update(
                    ent.get_all_frame_point_ids(self.element.sketch.registry)
                )

        to_screen = to_screen_matrix.transform_point

        for p in points:
            sx, sy = to_screen((p.x, p.y))

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

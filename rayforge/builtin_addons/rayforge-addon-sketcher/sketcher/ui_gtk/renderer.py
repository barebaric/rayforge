import cairo
import logging
import math
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, Set

from rayforge.core.geo import Geometry, Point as GeoPoint
from rayforge.core.matrix import Matrix
from rayforge.ui_gtk.canvas import WorldSurface
from ..core.commands import BezierPreviewState
from ..core.commands.dimension import DimensionData
from ..core.constraints import (
    CoincidentConstraint,
    PointOnLineConstraint,
)
from ..core.entities import (
    Arc,
    Bezier,
    Circle,
    Ellipse,
    Entity,
    Line,
    Point,
    TextBoxEntity,
)
from ..core.types import EntityID
from ..core.sketch import FillStyle
from .tools import PathTool, TextBoxTool

if TYPE_CHECKING:
    from .sketchelement import SketchElement

logger = logging.getLogger(__name__)


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

        to_screen = self.element.hittester.get_model_to_screen_transform(
            self.element
        )
        self._draw_points(ctx, to_screen)
        self._draw_overlays(ctx)
        self._draw_preview_dimensions(ctx)
        self._draw_bezier_control_handles(ctx)

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
        exclude_ids = set()
        text_tool = self.element.tools.get("text_box")
        if (
            self.element.active_tool_name == "text_box"
            and isinstance(text_tool, TextBoxTool)
            and text_tool.editing_entity_id is not None
        ):
            exclude_ids.add(text_tool.editing_entity_id)

        sketch = self.element.sketch

        for fill in sketch.fills:
            if not fill.boundary:
                continue

            geo = self._get_fill_geometry(fill, exclude_ids)
            if geo is None:
                continue

            ctx.new_path()
            geo.to_cairo(ctx)
            ctx.close_path()

            ctx.save()
            self._apply_fill_style(ctx, fill)
            ctx.fill()
            ctx.restore()

        self._draw_text_box_fills(ctx, exclude_ids)

    def _get_fill_geometry(self, fill, exclude_ids):
        """Generate geometry for a single fill."""
        if len(fill.boundary) == 1:
            eid, _ = fill.boundary[0]
            if eid in exclude_ids:
                return None
            entity = self.element.sketch.registry.get_entity(eid)
            if entity:
                return entity.create_fill_geometry(
                    self.element.sketch.registry
                )
            return None

        try:
            first_eid, first_fwd = fill.boundary[0]
            if first_eid in exclude_ids:
                return None
            first_ent = self.element.sketch.registry.get_entity(first_eid)
            if not first_ent:
                return None

            p_ids = first_ent.get_endpoint_ids()
            start_pid = p_ids[0] if first_fwd else p_ids[1]
            start_pt = self.element.sketch.registry.get_point(start_pid)

            from rayforge.core.geo import Geometry

            geo = Geometry()
            geo.move_to(start_pt.x, start_pt.y)

            valid_loop = True
            for eid, fwd in fill.boundary:
                if eid in exclude_ids:
                    valid_loop = False
                    break
                entity = self.element.sketch.registry.get_entity(eid)
                if not entity:
                    valid_loop = False
                    break
                entity.append_to_geometry(
                    geo, self.element.sketch.registry, fwd
                )

            if valid_loop:
                return geo
        except (IndexError, AttributeError):
            pass
        return None

    def _apply_fill_style(self, ctx: cairo.Context, fill):
        """Apply fill color or gradient to the cairo context."""
        if fill.style == FillStyle.SOLID:
            ctx.set_source_rgba(*fill.color)
        elif fill.style == FillStyle.LINEAR_GRADIENT:
            self._apply_linear_gradient(ctx, fill)
        elif fill.style == FillStyle.RADIAL_GRADIENT:
            self._apply_radial_gradient(ctx, fill)
        else:
            ctx.set_source_rgba(*fill.color)

    def _apply_linear_gradient(self, ctx: cairo.Context, fill):
        """Apply a linear gradient fill."""
        ext = ctx.path_extents()
        x1, y1, x2, y2 = ext

        angle_rad = math.radians(fill.gradient_angle)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        half_w = (x2 - x1) / 2
        half_h = (y2 - y1) / 2

        dx = math.cos(angle_rad) * max(half_w, half_h)
        dy = math.sin(angle_rad) * max(half_w, half_h)

        grad = cairo.LinearGradient(cx - dx, cy - dy, cx + dx, cy + dy)

        if fill.gradient_stops:
            for pos, color in fill.gradient_stops:
                grad.add_color_stop_rgba(pos, *color)
        else:
            grad.add_color_stop_rgba(0.0, *fill.color)
            grad.add_color_stop_rgba(1.0, *fill.color)

        ctx.set_source(grad)

    def _apply_radial_gradient(self, ctx: cairo.Context, fill):
        """Apply a radial gradient fill."""
        ext = ctx.path_extents()
        x1, y1, x2, y2 = ext

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        radius = max((x2 - x1) / 2, (y2 - y1) / 2)

        grad = cairo.RadialGradient(cx, cy, 0, cx, cy, radius)

        if fill.gradient_stops:
            for pos, color in fill.gradient_stops:
                grad.add_color_stop_rgba(pos, *color)
        else:
            grad.add_color_stop_rgba(0.0, *fill.color)
            grad.add_color_stop_rgba(1.0, *fill.color)

        ctx.set_source(grad)

    def _draw_text_box_fills(self, ctx: cairo.Context, exclude_ids: set):
        """Draw fills for text box entities."""
        for entity in self.element.sketch.registry.entities:
            if entity.id in exclude_ids:
                continue
            if not entity.construction:
                text_geo = entity.create_text_fill_geometry(
                    self.element.sketch.registry
                )
                if text_geo:
                    ctx.new_path()
                    text_geo.to_cairo(ctx)
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
        select_tool = self.element.tools.get("select")
        hovered_entity_id = (
            select_tool.hovered_entity_id if select_tool else None
        )

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

            # Skip construction geometry if not in edit mode or if hidden
            if entity.construction and (
                not is_editing or not self.element.show_construction_geometry
            ):
                continue

            if entity.invisible:
                continue

            is_sel = entity.id in self.element.selection.entity_ids
            is_hovered = entity.id == hovered_entity_id
            ctx.save()

            # 1. Define the Path
            has_path = False
            if isinstance(entity, Line):
                has_path = self._define_line_path(ctx, entity)
            elif isinstance(entity, Arc):
                has_path = self._define_arc_path(ctx, entity)
            elif isinstance(entity, Bezier):
                has_path = self._define_bezier_path(ctx, entity)
            elif isinstance(entity, Circle):
                has_path = self._define_circle_path(ctx, entity)
            elif isinstance(entity, Ellipse):
                has_path = self._define_ellipse_path(ctx, entity)
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
            elif is_hovered:
                ctx.save()
                ctx.set_dash([])
                ctx.set_source_rgba(1.0, 0.4, 0.2, 0.4)
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
                if is_hovered:
                    ctx.set_source_rgb(1.0, 0.4, 0.2)
                elif entity.constrained:
                    ctx.set_source_rgb(0.2, 0.3, 0.6)
                else:
                    ctx.set_source_rgb(0.3, 0.5, 0.8)
                ctx.stroke()
            else:
                self._set_standard_color(
                    ctx,
                    is_sel or is_hovered,
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

    def _safe_get_point(self, pid: EntityID) -> Optional[Point]:
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

    def _define_ellipse_path(
        self, ctx: cairo.Context, ellipse: Ellipse
    ) -> bool:
        """Defines the path for an ellipse without stroking."""
        center = self._safe_get_point(ellipse.center_idx)
        radius_x_pt = self._safe_get_point(ellipse.radius_x_pt_idx)
        radius_y_pt = self._safe_get_point(ellipse.radius_y_pt_idx)
        if not (center and radius_x_pt and radius_y_pt):
            return False

        rx = math.hypot(radius_x_pt.x - center.x, radius_x_pt.y - center.y)
        ry = math.hypot(radius_y_pt.x - center.x, radius_y_pt.y - center.y)
        if rx < 1e-9 or ry < 1e-9:
            return False

        rotation = math.atan2(
            radius_x_pt.y - center.y, radius_x_pt.x - center.x
        )

        ctx.save()
        ctx.translate(center.x, center.y)
        ctx.rotate(rotation)
        ctx.scale(rx, ry)
        ctx.new_sub_path()
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        return True

    def _define_bezier_path(self, ctx: cairo.Context, bezier: Bezier) -> bool:
        """Defines the path for a bezier curve without stroking."""
        start = self._safe_get_point(bezier.start_idx)
        end = self._safe_get_point(bezier.end_idx)
        if not (start and end):
            return False

        if bezier.is_line(self.element.sketch.registry):
            ctx.move_to(start.x, start.y)
            ctx.line_to(end.x, end.y)
            return True

        cp1_x, cp1_y, cp2_x, cp2_y = bezier.get_control_points_or_endpoints(
            self.element.sketch.registry
        )
        ctx.move_to(start.x, start.y)
        ctx.curve_to(cp1_x, cp1_y, cp2_x, cp2_y, end.x, end.y)
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
        if not self.element.show_constraints:
            return

        # --- Stage 0: Get Hover State ---
        select_tool = self.element.tools.get("select")
        hovered_constraint_idx = (
            select_tool.hovered_constraint_idx if select_tool else None
        )
        if self.element.external_hovered_constraint_idx is not None:
            hovered_constraint_idx = (
                self.element.external_hovered_constraint_idx
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

        # Collect construction entity IDs and their point IDs
        construction_entity_ids = set()
        construction_point_ids = set()
        if not self.element.show_construction_geometry:
            for entity in self.element.sketch.registry.entities:
                if entity.construction:
                    construction_entity_ids.add(entity.id)
                    construction_point_ids.update(entity.get_point_ids())

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

            # Hide constraints referencing construction geometry when hidden
            if construction_entity_ids or construction_point_ids:
                if constr.depends_on_entities(
                    construction_entity_ids
                ) or constr.depends_on_points(construction_point_ids):
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
        self._draw_junctions(
            ctx, to_screen_func, text_box_point_ids, construction_point_ids
        )

    def _draw_junctions(
        self,
        ctx: cairo.Context,
        to_screen: Callable[[GeoPoint], GeoPoint],
        text_box_point_ids: Set[int],
        construction_point_ids: Set[int] | None = None,
    ) -> None:
        if construction_point_ids is None:
            construction_point_ids = set()
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
            elif isinstance(entity, Bezier):
                point_counts[entity.start_idx] += 1
                point_counts[entity.end_idx] += 1
            elif isinstance(entity, Circle):
                point_counts[entity.center_idx] += 1
                point_counts[entity.radius_pt_idx] += 1
            elif isinstance(entity, Ellipse):
                point_counts[entity.center_idx] += 1
                point_counts[entity.radius_x_pt_idx] += 1
                point_counts[entity.radius_y_pt_idx] += 1

        for pid, count in point_counts.items():
            if count > 1:
                # Hide junction visuals for text box points
                if pid in text_box_point_ids:
                    continue

                # Hide junction visuals for construction geometry when hidden
                if pid in construction_point_ids:
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

    def _get_entity_by_id(self, eid: EntityID) -> Optional[Entity]:
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

        if self.element.active_tool_name == "path":
            path_tool = self.element.tools.get("path")
            if path_tool is not None:
                path_hover_pid = path_tool.hovered_point_id
                if path_hover_pid is not None:
                    hover_pid = path_hover_pid

        entity_points = set()

        for eid in self.element.selection.entity_ids:
            ent = self._get_entity_by_id(eid)
            if isinstance(ent, TextBoxEntity):
                entity_points.update(
                    ent.get_all_frame_point_ids(self.element.sketch.registry)
                )
            elif ent:
                entity_points.update(ent.get_point_ids())

        # Collect construction point IDs to hide when construction is hidden
        construction_point_ids = set()
        if not self.element.show_construction_geometry:
            for entity in self.element.sketch.registry.entities:
                if entity.construction:
                    construction_point_ids.update(entity.get_point_ids())

        # Collect hidden point IDs from active tool preview
        hidden_point_ids = set()
        preview_state = self.element.current_tool.get_preview_state()
        if preview_state is not None:
            hidden_point_ids = preview_state.get_hidden_point_ids()

        to_screen = to_screen_matrix.transform_point

        for p in points:
            # Hide points belonging to construction geometry when hidden
            if p.id in construction_point_ids:
                continue
            # Hide points marked as hidden by active tool preview
            if p.id in hidden_point_ids:
                continue
            sx, sy = to_screen((p.x, p.y))

            is_hovered = p.id == hover_pid
            is_explicit_sel = p.id in self.element.selection.point_ids
            is_implicit_sel = p.id in entity_points

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

            if is_hovered:
                ctx.save()
                ctx.set_source_rgba(1.0, 0.4, 0.2, 0.4)
                ctx.arc(sx, sy, r + 5, 0, 2 * math.pi)
                ctx.fill()
                ctx.restore()

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

    def _draw_preview_dimensions(self, ctx: cairo.Context):
        tool = self.element.current_tool
        preview_state = tool.get_preview_state()
        if preview_state is None:
            return

        dimensions = preview_state.get_dimensions(self.element.sketch.registry)
        if not dimensions:
            return

        to_screen_transform = (
            self.element.hittester.get_model_to_screen_transform(self.element)
        )
        to_screen = to_screen_transform.transform_point

        ctx.save()
        ctx.set_font_size(11)
        ctx.set_line_width(1.0)

        dim_input_buffer = getattr(tool, "_dim_input", None)
        dim_input_active = (
            dim_input_buffer is not None and dim_input_buffer.is_active()
        )

        for dim_idx, dim in enumerate(dimensions):
            if not isinstance(dim, DimensionData):
                continue

            sx, sy = to_screen(dim.position)

            has_leader = dim.leader_end is not None

            label = dim.label
            is_editing = False
            if dim_input_active and dim_input_buffer is not None:
                field_text = dim_input_buffer.get_display_text(dim_idx)
                if field_text is not None:
                    label = field_text
                    is_editing = True
                elif dim_input_buffer.field_count == 1:
                    label = dim_input_buffer.get_display_text() or label
                    is_editing = True

            extents = ctx.text_extents(label)
            text_w = extents.width
            text_h = extents.height
            x_bearing = extents.x_bearing

            label_offset_x = 15
            label_offset_y = -15
            label_sx = sx + label_offset_x
            label_sy = sy + label_offset_y

            if has_leader:
                lx, ly = to_screen(dim.leader_end)
                ctx.set_source_rgba(0.2, 0.4, 0.8, 0.9)
                ctx.move_to(lx, ly)
                ctx.line_to(label_sx, label_sy)
                ctx.stroke()

            padding = 3.0
            bg_x = label_sx + x_bearing - padding
            bg_y = label_sy - text_h - padding
            bg_w = text_w + 2 * padding
            bg_h = text_h + 2 * padding

            if is_editing:
                ctx.set_source_rgba(0.9, 0.95, 1.0, 0.95)
            else:
                ctx.set_source_rgba(1.0, 1.0, 1.0, 0.85)
            ctx.rectangle(bg_x, bg_y, bg_w, bg_h)
            ctx.fill()

            if is_editing:
                ctx.set_source_rgba(0.0, 0.2, 0.8, 1.0)
            else:
                ctx.set_source_rgba(0.1, 0.1, 0.1, 1.0)
            ctx.move_to(label_sx, label_sy)
            ctx.show_text(label)

        ctx.restore()

    def _draw_bezier_control_handles(self, ctx: cairo.Context):
        """Draws control handles for bezier preview and selected beziers."""
        to_screen_transform = (
            self.element.hittester.get_model_to_screen_transform(self.element)
        )
        to_screen = to_screen_transform.transform_point

        tool = self.element.current_tool
        preview_state = None
        if isinstance(tool, PathTool):
            preview_state = tool.get_preview_state()

        if isinstance(preview_state, BezierPreviewState):
            if not preview_state.is_line_preview:
                self._draw_bezier_preview_handles(
                    ctx, to_screen, preview_state
                )

        selected_bezier_ids: set[int] = set()
        selected_point_ids: set[int] = set()
        for eid in self.element.selection.entity_ids:
            entity = self._get_entity_by_id(eid)
            if isinstance(entity, Bezier):
                selected_bezier_ids.add(eid)
                selected_point_ids.add(entity.start_idx)
                selected_point_ids.add(entity.end_idx)

        for pid in selected_point_ids:
            waypoint = self._safe_get_point(pid)
            if waypoint is None:
                continue
            self._draw_waypoint_handles(ctx, to_screen, waypoint)

        selected_waypoint_ids: set[int] = set()
        for pid in self.element.selection.point_ids:
            selected_waypoint_ids.add(pid)
        if self.element.selection.junction_pid is not None:
            selected_waypoint_ids.add(self.element.selection.junction_pid)

        for pid in selected_waypoint_ids:
            waypoint = self._safe_get_point(pid)
            if waypoint is None:
                continue
            connected = waypoint.get_connected_beziers(
                self.element.sketch.registry
            )
            if connected:
                self._draw_waypoint_handles(ctx, to_screen, waypoint)

    def _draw_bezier_preview_handles(
        self, ctx: cairo.Context, to_screen, preview_state: BezierPreviewState
    ):
        """Draw control handles for the active bezier preview."""
        if preview_state.end_id is None:
            return

        try:
            start_pt = self.element.sketch.registry.get_point(
                preview_state.start_id
            )
            end_pt = self.element.sketch.registry.get_point(
                preview_state.end_id
            )
        except IndexError:
            return

        if not (start_pt and end_pt):
            return

        start_sx, start_sy = to_screen((start_pt.x, start_pt.y))
        end_sx, end_sy = to_screen((end_pt.x, end_pt.y))

        temp_bezier = None
        if preview_state.temp_entity_id is not None:
            temp_entity = self.element.sketch.registry.get_entity(
                preview_state.temp_entity_id
            )
            if isinstance(temp_entity, Bezier):
                temp_bezier = temp_entity

        ctx.save()
        ctx.set_line_width(1.0)
        ctx.set_source_rgba(0.6, 0.4, 0.8, 0.8)

        if temp_bezier is not None:
            if temp_bezier.cp1 is not None:
                cp1_abs = (
                    start_pt.x + temp_bezier.cp1[0],
                    start_pt.y + temp_bezier.cp1[1],
                )
                cp1_sx, cp1_sy = to_screen(cp1_abs)
                ctx.move_to(start_sx, start_sy)
                ctx.line_to(cp1_sx, cp1_sy)
                ctx.stroke()

            if temp_bezier.cp2 is not None:
                cp2_abs = (
                    end_pt.x + temp_bezier.cp2[0],
                    end_pt.y + temp_bezier.cp2[1],
                )
                cp2_sx, cp2_sy = to_screen(cp2_abs)
                ctx.move_to(end_sx, end_sy)
                ctx.line_to(cp2_sx, cp2_sy)
                ctx.stroke()

        virtual_cp_abs = preview_state.get_virtual_cp_absolute(
            self.element.sketch.registry
        )
        if virtual_cp_abs is not None:
            cp_sx, cp_sy = to_screen(virtual_cp_abs)
            ctx.move_to(end_sx, end_sy)
            ctx.line_to(cp_sx, cp_sy)
            ctx.stroke()

        ctx.set_source_rgba(0.6, 0.4, 0.8, 1.0)

        if temp_bezier is not None:
            if temp_bezier.cp1 is not None:
                cp1_abs = (
                    start_pt.x + temp_bezier.cp1[0],
                    start_pt.y + temp_bezier.cp1[1],
                )
                cp1_sx, cp1_sy = to_screen(cp1_abs)
                ctx.rectangle(cp1_sx - 4, cp1_sy - 4, 8, 8)
                ctx.fill()

            if temp_bezier.cp2 is not None:
                cp2_abs = (
                    end_pt.x + temp_bezier.cp2[0],
                    end_pt.y + temp_bezier.cp2[1],
                )
                cp2_sx, cp2_sy = to_screen(cp2_abs)
                ctx.rectangle(cp2_sx - 4, cp2_sy - 4, 8, 8)
                ctx.fill()

        if virtual_cp_abs is not None:
            cp_sx, cp_sy = to_screen(virtual_cp_abs)
            ctx.rectangle(cp_sx - 4, cp_sy - 4, 8, 8)
            ctx.fill()

        ctx.restore()

    def _draw_waypoint_handles(
        self,
        ctx: cairo.Context,
        to_screen,
        waypoint: Point,
    ):
        """Draw control handles for all beziers connected to a waypoint."""
        sketch = self.element.sketch
        connected_beziers = waypoint.get_connected_beziers(
            sketch.registry, sketch
        )

        if not connected_beziers:
            return

        wp_sx, wp_sy = to_screen((waypoint.x, waypoint.y))

        point_ids = {waypoint.id}
        point_ids.update(sketch.get_coincident_points(waypoint.id))

        ctx.save()
        ctx.set_line_width(1.0)
        ctx.set_source_rgba(0.2, 0.6, 1.0, 0.8)

        for bezier in connected_beziers:
            if bezier.start_idx in point_ids and bezier.cp1 is not None:
                cp_abs = (
                    waypoint.x + bezier.cp1[0],
                    waypoint.y + bezier.cp1[1],
                )
                cp_sx, cp_sy = to_screen(cp_abs)
                ctx.move_to(wp_sx, wp_sy)
                ctx.line_to(cp_sx, cp_sy)
                ctx.stroke()
                ctx.set_source_rgba(0.2, 0.6, 1.0, 1.0)
                ctx.rectangle(cp_sx - 4, cp_sy - 4, 8, 8)
                ctx.fill()
                ctx.set_source_rgba(0.2, 0.6, 1.0, 0.8)

            if bezier.end_idx in point_ids and bezier.cp2 is not None:
                cp_abs = (
                    waypoint.x + bezier.cp2[0],
                    waypoint.y + bezier.cp2[1],
                )
                cp_sx, cp_sy = to_screen(cp_abs)
                ctx.move_to(wp_sx, wp_sy)
                ctx.line_to(cp_sx, cp_sy)
                ctx.stroke()
                ctx.set_source_rgba(0.2, 0.6, 1.0, 1.0)
                ctx.rectangle(cp_sx - 4, cp_sy - 4, 8, 8)
                ctx.fill()
                ctx.set_source_rgba(0.2, 0.6, 1.0, 0.8)

        ctx.restore()

import os
from typing import TYPE_CHECKING, Any, List, Optional, Set

import cairo

from rayforge.core.geo import Point as GeoPoint
from ...core.constraints import (
    CoincidentConstraint,
    PointOnLineConstraint,
    SymmetryConstraint,
)
from ...core.entities import Arc, Circle, Line, Point
from ...core.snap import (
    DragContext,
    SnapLineType,
    SnapResult,
    SNAP_LINE_STYLES,
)
from ...core.snap.types import SnapLineStyle
from .base import SketcherKey

if TYPE_CHECKING:
    from ..sketchelement import SketchElement

DEBUG_SNAPPING = os.environ.get("DEBUG_SNAPPING", "").lower() in (
    "1",
    "true",
    "yes",
)


class SnapMixin:
    """Mixin providing snap functionality for sketch tools.

    This mixin provides:
    - Magnetic snap querying during hover and creation
    - Snap visual feedback rendering
    - Constraint creation from snap results
    - Tab key toggle for magnetic snap

    Usage:
        class MyTool(SnapMixin, SketchTool):
            def on_hover_motion(self, world_x, world_y):
                mx, my = self.element.hittester.screen_to_model(...)
                mx, my, snap_result = self.query_snap_for_creation(
                    self.element, mx, my
                )
                self.current_snap_result = snap_result
                # Use mx, my for snapped position

            def draw_overlay(self, ctx):
                self.draw_snap_feedback(ctx, self.element)
    """

    current_snap_result: Optional[SnapResult] = None
    magnetic_snap_enabled: bool = True

    def toggle_magnetic_snap(self) -> None:
        """Toggle magnetic snap on/off."""
        self.magnetic_snap_enabled = not self.magnetic_snap_enabled

    def is_snap_active(self) -> bool:
        """Check if magnetic snap is currently active and has a result."""
        return (
            self.magnetic_snap_enabled and self.current_snap_result is not None
        )

    def query_snap_for_creation(
        self,
        element: "SketchElement",
        model_x: float,
        model_y: float,
        exclude_points: Optional[Set[int]] = None,
    ) -> GeoPoint:
        """Query snap engine for geometry creation.

        This is used during hover/press in creation tools to get
        snapped positions.

        Args:
            element: The SketchElement
            model_x: Model X coordinate
            model_y: Model Y coordinate
            exclude_points: Optional set of point IDs to exclude from snapping

        Returns:
            Snapped position as GeoPoint (or original position if no snap)
        """
        if not self.magnetic_snap_enabled:
            self.current_snap_result = None
            return (model_x, model_y)

        context = DragContext(
            dragged_point_ids=exclude_points or set(),
            dragged_entity_ids=set(),
            initial_positions={},
        )

        snap_result = self._query_snap_engine(
            element, (model_x, model_y), context
        )

        if snap_result.snapped:
            self.current_snap_result = snap_result
            return snap_result.position
        else:
            self.current_snap_result = None
            return (model_x, model_y)

    def query_snap_for_drag(
        self,
        element: "SketchElement",
        model_x: float,
        model_y: float,
        dragged_point_ids: Optional[Set[int]] = None,
        dragged_entity_ids: Optional[Set[int]] = None,
        initial_positions: Optional[dict] = None,
    ) -> GeoPoint:
        """Query snap engine during drag operations.

        Args:
            element: The SketchElement
            model_x: Model X coordinate (target position)
            model_y: Model Y coordinate (target position)
            dragged_point_ids: Points being dragged (excluded from snap)
            dragged_entity_ids: Entities being dragged (excluded from snap)
            initial_positions: Initial positions for drag context

        Returns:
            Snapped position as GeoPoint (or original position if no snap)
        """
        if not self.magnetic_snap_enabled:
            self.current_snap_result = None
            return (model_x, model_y)

        context = DragContext(
            dragged_point_ids=dragged_point_ids or set(),
            dragged_entity_ids=dragged_entity_ids or set(),
            initial_positions=initial_positions or {},
        )

        snap_result = self._query_snap_engine(
            element, (model_x, model_y), context
        )

        if snap_result.snapped:
            self.current_snap_result = snap_result
            return snap_result.position
        else:
            self.current_snap_result = None
            return (model_x, model_y)

    def _query_snap_engine(
        self,
        element: "SketchElement",
        position: GeoPoint,
        context: DragContext,
    ) -> SnapResult:
        """Internal method to query the snap engine."""
        if element.canvas:
            scale_x, _ = element.canvas.view_transform.get_scale()
            if scale_x > 0:
                element.snap_engine.threshold = 5.0 / scale_x
        return element.snap_engine.query(
            element.sketch.registry, position, context
        )

    def build_snap_constraints(
        self,
        point_id: int,
    ) -> List[Any]:
        """Build constraints from the current snap result.

        Call this when finalizing geometry to create persistent constraints
        from the snap operation.

        Args:
            point_id: The point ID that was snapped

        Returns:
            List of constraints to add to the sketch
        """
        constraints: List[Any] = []

        if (
            not self.current_snap_result
            or not self.current_snap_result.primary_snap_point
        ):
            return constraints

        sp = self.current_snap_result.primary_snap_point

        if sp.line_type == SnapLineType.MIDPOINT:
            if isinstance(sp.source, Line):
                line = sp.source
                if point_id not in (line.p1_idx, line.p2_idx):
                    constraints.append(
                        SymmetryConstraint(
                            p1=line.p1_idx,
                            p2=line.p2_idx,
                            center=point_id,
                        )
                    )

        elif sp.line_type == SnapLineType.ENTITY_POINT:
            if isinstance(sp.source, Point):
                target_point = sp.source
                if target_point.id != point_id:
                    constraints.append(
                        CoincidentConstraint(
                            p1=point_id,
                            p2=target_point.id,
                        )
                    )

        elif sp.line_type == SnapLineType.ON_ENTITY:
            if isinstance(sp.source, (Line, Arc, Circle)):
                entity = sp.source
                constraints.append(
                    PointOnLineConstraint(
                        point_id=point_id,
                        shape_id=entity.id,
                    )
                )

        return constraints

    def get_snapped_point_id(self) -> Optional[int]:
        """Get the point ID if snapped to an existing point.

        Returns:
            Point ID if snapped to ENTITY_POINT, None otherwise
        """
        if (
            self.current_snap_result
            and self.current_snap_result.primary_snap_point
        ):
            sp = self.current_snap_result.primary_snap_point
            if sp.line_type == SnapLineType.ENTITY_POINT:
                if isinstance(sp.source, Point):
                    return sp.source.id
        return None

    def clear_snap_result(self) -> None:
        """Clear the current snap result."""
        self.current_snap_result = None

    def handle_snap_key_event(
        self, key: SketcherKey, is_active: bool = True
    ) -> bool:
        """Handle key events for snap toggle.

        Args:
            key: The key event
            is_active: Whether the tool is in an active state (e.g., dragging)

        Returns:
            True if the key was handled
        """
        if key == SketcherKey.TAB and is_active:
            self.toggle_magnetic_snap()
            return True
        return False

    def draw_snap_feedback(
        self,
        ctx: cairo.Context,
        element: "SketchElement",
    ) -> None:
        """Draw snap lines and snap point indicators.

        Args:
            ctx: Cairo context for drawing
            element: The SketchElement
        """
        if not self.current_snap_result or not element.canvas:
            return

        to_screen = element.hittester.get_model_to_screen_transform(element)
        canvas_width = element.canvas.get_width()
        canvas_height = element.canvas.get_height()

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
            self._draw_snap_point_indicator(ctx, element, to_screen)

        ctx.restore()

    def _draw_snap_point_indicator(
        self,
        ctx: cairo.Context,
        element: "SketchElement",
        to_screen,
    ) -> None:
        """Draw the indicator for the primary snap point."""
        from ...core.constraints.symmetry import draw_symmetry_arrows

        if not self.current_snap_result:
            return

        sp = self.current_snap_result.primary_snap_point
        if not sp:
            return

        sx, sy = to_screen.transform_point((sp.x, sp.y))

        if sp.line_type == SnapLineType.EQUIDISTANT and sp.spacing:
            self._draw_equidistant_indicator(
                ctx, element, to_screen, sp, sx, sy
            )
        elif sp.line_type == SnapLineType.MIDPOINT:
            if isinstance(sp.source, Line):
                line = sp.source
                p1 = element.sketch.registry.get_point(line.p1_idx)
                p2 = element.sketch.registry.get_point(line.p2_idx)
                if p1 and p2:
                    s1 = to_screen.transform_point((p1.x, p1.y))
                    s2 = to_screen.transform_point((p2.x, p2.y))
                    style = SNAP_LINE_STYLES.get(sp.line_type, SnapLineStyle())
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
            ctx.new_path()
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
                element.renderer.draw_entity_highlight(
                    ctx, sp.source, style.color, line_width=3.0 / scale
                )
                ctx.restore()
        else:
            ctx.set_source_rgba(1.0, 0.0, 1.0, 0.8)
            ctx.new_path()
            ctx.arc(sx, sy, 5, 0, 2 * 3.14159)
            ctx.fill()

    def _draw_equidistant_indicator(
        self,
        ctx: cairo.Context,
        element: "SketchElement",
        to_screen,
        sp,
        sx: float,
        sy: float,
    ) -> None:
        """Draw equidistant snap indicator with double arrows."""
        style = SNAP_LINE_STYLES.get(sp.line_type, SnapLineStyle())
        ctx.set_source_rgba(*style.color)
        ctx.set_dash([])
        ctx.set_line_width(2.0)
        scale_x, _ = to_screen.get_scale()
        head_len = min(sp.spacing * scale_x * 0.15, 8) if sp.spacing else 8
        head_width = 4
        tick_len = 6

        coords = (
            sp.pattern_coords
            if sp.pattern_coords
            else (sp.y if sp.is_horizontal else sp.x,)
        )

        def draw_double_arrow(ctx, x1, y1, x2, y2, head_len, head_width):
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
            axis_x = sp.axis_coord if sp.axis_coord is not None else sp.x
            for i in range(len(coords) - 1):
                y1, y2 = coords[i], coords[i + 1]
                if sp.spacing and abs(y2 - y1 - sp.spacing) > 0.5:
                    continue
                sy1 = to_screen.transform_point((axis_x, y1))[1]
                sy2 = to_screen.transform_point((axis_x, y2))[1]
                local_sx = to_screen.transform_point((axis_x, sp.y))[0]
                draw_double_arrow(
                    ctx, local_sx, sy1, local_sx, sy2, head_len, head_width
                )
                ctx.move_to(local_sx - tick_len, sy1)
                ctx.line_to(local_sx + tick_len, sy1)
                ctx.move_to(local_sx - tick_len, sy2)
                ctx.line_to(local_sx + tick_len, sy2)
        else:
            axis_y = sp.axis_coord if sp.axis_coord is not None else sp.y
            for i in range(len(coords) - 1):
                x1, x2 = coords[i], coords[i + 1]
                if sp.spacing and abs(x2 - x1 - sp.spacing) > 0.5:
                    continue
                sx1 = to_screen.transform_point((x1, axis_y))[0]
                sx2 = to_screen.transform_point((x2, axis_y))[0]
                local_sy = to_screen.transform_point((sp.x, axis_y))[1]
                draw_double_arrow(
                    ctx, sx1, local_sy, sx2, local_sy, head_len, head_width
                )
                ctx.move_to(sx1, local_sy - tick_len)
                ctx.line_to(sx1, local_sy + tick_len)
                ctx.move_to(sx2, local_sy - tick_len)
                ctx.line_to(sx2, local_sy + tick_len)
        ctx.stroke()

    def draw_debug_snap_lines(
        self,
        ctx: cairo.Context,
        element: "SketchElement",
    ) -> None:
        """Draw all available snap lines for debugging."""
        if not DEBUG_SNAPPING or not element.canvas:
            return

        to_screen = element.hittester.get_model_to_screen_transform(element)
        canvas_width = element.canvas.get_width()
        canvas_height = element.canvas.get_height()

        center_x = canvas_width / 2
        center_y = canvas_height / 2
        view_transform = element.canvas.view_transform
        model_center = view_transform.invert().transform_point(
            (center_x, center_y)
        )

        old_threshold = element.snap_engine.threshold
        element.snap_engine.threshold = 1e9
        snap_lines = element.snap_engine.get_visible_snap_lines(
            element.sketch.registry,
            model_center,
            DragContext(),
        )
        element.snap_engine.threshold = old_threshold

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

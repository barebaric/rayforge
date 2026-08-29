from __future__ import annotations

import math
from collections.abc import Callable
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

import cairo
from raygeo.geo.types import Point

from ..entities import Bezier, TextBoxEntity
from ..types import EntityID
from .base import Constraint, ConstraintStatus

if TYPE_CHECKING:
    from ..params import ParameterContext
    from ..registry import EntityRegistry
    from ..selection import SketchSelection
    from ..sketch import Sketch

CurveCoeffs = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]

_CURVE_SAMPLES = 24
_NEWTON_ITERATIONS = 12
_MIN_DISTANCE = 1e-9


def _basis(t: float) -> tuple[float, float, float, float]:
    """Cubic Bernstein basis evaluated at t."""
    mt = 1.0 - t
    return (mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t)


def _basis_d1(t: float) -> tuple[float, float, float, float]:
    """First derivative of the cubic Bernstein basis at t."""
    mt = 1.0 - t
    return (
        -3.0 * mt * mt,
        3.0 * mt * (1.0 - 3.0 * t),
        3.0 * t * (2.0 - 3.0 * t),
        3.0 * t * t,
    )


def _basis_d2(t: float) -> tuple[float, float, float, float]:
    """Second derivative of the cubic Bernstein basis at t."""
    return (6.0 * (1.0 - t), 18.0 * t - 12.0, 6.0 - 18.0 * t, 6.0 * t)


def _eval_curve(
    xs: tuple[float, ...], ys: tuple[float, ...], basis: tuple[float, ...]
) -> tuple[float, float]:
    bx = sum(b * x for b, x in zip(basis, xs))
    by = sum(b * y for b, y in zip(basis, ys))
    return bx, by


def closest_point_on_bezier(
    xs: tuple[float, float, float, float],
    ys: tuple[float, float, float, float],
    px: float,
    py: float,
) -> tuple[float, float, float]:
    """
    Finds the point on a cubic Bezier curve closest to (px, py).

    The stationarity condition (B(t) - P) . B'(t) = 0 is solved by
    sampling the curve coarsely and refining the best sample with
    Newton iterations.

    Returns:
        Tuple (t, bx, by): the curve parameter and the coordinates of
        the closest curve point.
    """
    best_t = 0.0
    best_dist_sq = float("inf")
    for i in range(_CURVE_SAMPLES + 1):
        t = i / _CURVE_SAMPLES
        bx, by = _eval_curve(xs, ys, _basis(t))
        dist_sq = (bx - px) ** 2 + (by - py) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_t = t

    t = best_t
    for _newton in range(_NEWTON_ITERATIONS):
        d1 = _eval_curve(xs, ys, _basis_d1(t))
        d2 = _eval_curve(xs, ys, _basis_d2(t))
        bx, by = _eval_curve(xs, ys, _basis(t))
        fx = bx - px
        fy = by - py
        f = fx * d1[0] + fy * d1[1]
        fp = d1[0] * d1[0] + d1[1] * d1[1] + fx * d2[0] + fy * d2[1]
        if abs(fp) < 1e-12:
            break
        step = f / fp
        t = min(1.0, max(0.0, t - step))
        if abs(step) < 1e-12:
            break

    bx, by = _eval_curve(xs, ys, _basis(t))
    return t, bx, by


def _get_curve_coeffs(bez: Bezier, reg: EntityRegistry) -> CurveCoeffs:
    start = reg.get_point(bez.start_idx)
    end = reg.get_point(bez.end_idx)
    cp1x, cp1y, cp2x, cp2y = bez.get_control_points_or_endpoints(reg)
    xs = (start.x, cp1x, cp2x, end.x)
    ys = (start.y, cp1y, cp2y, end.y)
    return xs, ys


class PointOnCurveConstraint(Constraint):
    """Enforces a point lies on a Bezier curve."""

    def __init__(
        self, point_id: EntityID, shape_id: EntityID, user_visible: bool = True
    ):
        super().__init__(user_visible=user_visible)
        self.point_id: EntityID = point_id
        self.shape_id: EntityID = shape_id

    @classmethod
    def get_type_key(cls) -> str:
        return "point_on_curve"

    @classmethod
    def can_apply_to(
        cls, selection: SketchSelection, sketch: Sketch | None = None
    ) -> bool:
        if len(selection.point_ids) != 1 or len(selection.entity_ids) != 1:
            return False
        if sketch is None:
            return False
        entity = sketch.registry.get_entity(selection.entity_ids[0])
        if not isinstance(entity, Bezier):
            return False
        return selection.point_ids[0] not in entity.get_endpoint_ids()

    @staticmethod
    def get_type_name() -> str:
        """Returns to human-readable name of this constraint type."""
        return _("Point on Curve")

    def get_title(self) -> str:
        """Returns a human-readable title for this constraint."""
        return self.get_type_name()

    def get_subtitle(self, registry: EntityRegistry) -> str:
        """Returns subtitle describing constrained entities."""
        pt = registry.get_point(self.point_id)
        if pt:
            return _("Point at {}").format(self._format_coord(pt.x, pt.y))
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "PointOnCurveConstraint",
            "point_id": self.point_id,
            "shape_id": self.shape_id,
            "user_visible": self.user_visible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PointOnCurveConstraint:
        return cls(
            point_id=data["point_id"],
            shape_id=data["shape_id"],
            user_visible=data.get("user_visible", True),
        )

    def _get_bezier(self, reg: EntityRegistry) -> Bezier | None:
        bez = reg.get_entity(self.shape_id)
        return bez if isinstance(bez, Bezier) else None

    def error(self, reg: EntityRegistry, params: ParameterContext) -> float:
        bez = self._get_bezier(reg)
        if bez is None:
            return 0.0
        xs, ys = _get_curve_coeffs(bez, reg)
        pt = reg.get_point(self.point_id)
        _t, bx, by = closest_point_on_bezier(xs, ys, pt.x, pt.y)
        return math.hypot(bx - pt.x, by - pt.y)

    def gradient(
        self, reg: EntityRegistry, params: ParameterContext
    ) -> dict[EntityID, list[Point]]:
        bez = self._get_bezier(reg)
        if bez is None:
            return {}
        xs, ys = _get_curve_coeffs(bez, reg)
        pt = reg.get_point(self.point_id)
        t, bx, by = closest_point_on_bezier(xs, ys, pt.x, pt.y)
        dist = math.hypot(bx - pt.x, by - pt.y)
        if dist < _MIN_DISTANCE:
            return {}
        ux = (bx - pt.x) / dist
        uy = (by - pt.y) / dist

        # The distance to the closest curve point only depends on the
        # curve through the offset direction at the solved parameter
        # (envelope theorem), so plain Bernstein weights suffice here.
        basis = _basis(t)
        w_start = basis[0] + basis[1]
        w_end = basis[2] + basis[3]
        return {
            self.point_id: [(-ux, -uy)],
            bez.start_idx: [(w_start * ux, w_start * uy)],
            bez.end_idx: [(w_end * ux, w_end * uy)],
        }

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: EntityRegistry,
        to_screen: Callable[[Point], Point],
        element: Any,
        threshold: float,
    ) -> bool:
        pt = reg.get_point(self.point_id)
        if pt:
            s_pt = to_screen((pt.x, pt.y))
            return math.hypot(sx - s_pt[0], sy - s_pt[1]) < threshold
        return False

    def draw(
        self,
        ctx: cairo.Context,
        registry: EntityRegistry,
        to_screen: Callable[[Point], Point],
        is_selected: bool = False,
        is_hovered: bool = False,
        point_radius: float = 5.0,
    ) -> None:
        # Hide constraint if its point is part of a text box
        text_box_point_ids = set()
        for entity in registry.entities:
            if isinstance(entity, TextBoxEntity):
                text_box_point_ids.update(
                    entity.get_all_frame_point_ids(registry)
                )
        if self.point_id in text_box_point_ids:
            return

        try:
            p = registry.get_point(self.point_id)
        except IndexError:
            return

        sx, sy = to_screen((p.x, p.y))

        ctx.save()
        ctx.set_line_width(1.5)

        radius = point_radius + 4
        ctx.new_sub_path()
        ctx.arc(sx, sy, radius, 0, 2 * math.pi)

        if is_selected:
            self._draw_selection_underlay(ctx)

        if self.status == ConstraintStatus.CONFLICTING:
            self._draw_conflict_underlay(ctx)

        self._set_color(ctx, is_hovered)
        ctx.stroke()
        ctx.restore()

    def get_draggable_point(self) -> EntityID:
        """Returns the point that lies on the curve."""
        return self.point_id

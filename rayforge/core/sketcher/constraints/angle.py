from __future__ import annotations
import logging
import math
from gettext import gettext as _
from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    List,
    Optional,
    Callable,
    TYPE_CHECKING,
)
from ...geo.primitives import line_intersection, normalize_angle
from ..entities import Line
from .base import Constraint, ConstraintStatus

if TYPE_CHECKING:
    import cairo
    from ..params import ParameterContext
    from ..registry import EntityRegistry

logger = logging.getLogger(__name__)

ANGLE_WEIGHT = 10.0
ARC_RADIUS = 35.0
LABEL_RADIUS = ARC_RADIUS - 10.0


def _get_far_point(ix, iy, p1, p2):
    d1 = (p1.x - ix) ** 2 + (p1.y - iy) ** 2
    d2 = (p2.x - ix) ** 2 + (p2.y - iy) ** 2
    return p1 if d1 > d2 else p2


class AngleConstraint(Constraint):
    """
    Enforces a specific angle between two lines.

    e1 is the anchor line, e2 is the other line.
    The angle is measured CW from anchor direction to other direction,
    using directions pointing away from the lines' intersection toward
    the stored far points.
    """

    def __init__(
        self,
        e1_id: int,
        e2_id: int,
        value: Union[str, float],
        expression: Optional[str] = None,
        user_visible: bool = True,
        e1_far_idx: Optional[int] = None,
        e2_far_idx: Optional[int] = None,
    ):
        super().__init__(user_visible=user_visible)
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.e1_far_idx = e1_far_idx
        self.e2_far_idx = e2_far_idx

        if expression is not None:
            self.expression = expression
            self.value = float(value)
        elif isinstance(value, str):
            self.expression = value
            self.value = 0.0
        else:
            self.expression = None
            self.value = float(value)

    @staticmethod
    def get_type_name() -> str:
        return _("Angle")

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "type": "AngleConstraint",
            "e1_id": self.e1_id,
            "e2_id": self.e2_id,
            "value": self.value,
            "user_visible": self.user_visible,
            "e1_far_idx": self.e1_far_idx,
            "e2_far_idx": self.e2_far_idx,
        }
        if self.expression:
            data["expression"] = self.expression
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AngleConstraint":
        return cls(
            e1_id=data["e1_id"],
            e2_id=data["e2_id"],
            value=data["value"],
            expression=data.get("expression"),
            user_visible=data.get("user_visible", True),
            e1_far_idx=data.get("e1_far_idx"),
            e2_far_idx=data.get("e2_far_idx"),
        )

    def _get_line_params(
        self, reg: "EntityRegistry"
    ) -> Optional[Tuple[Line, Line, Any, Any, Any, Any]]:
        e1 = reg.get_entity(self.e1_id)
        e2 = reg.get_entity(self.e2_id)
        if not (isinstance(e1, Line) and isinstance(e2, Line)):
            return None

        p1 = reg.get_point(e1.p1_idx)
        p2 = reg.get_point(e1.p2_idx)
        p3 = reg.get_point(e2.p1_idx)
        p4 = reg.get_point(e2.p2_idx)

        if not (p1 and p2 and p3 and p4):
            return None

        return e1, e2, p1, p2, p3, p4

    def _get_far_points(
        self, e1: Line, e2: Line, p1, p2, p3, p4, ix: float, iy: float
    ):
        if self.e1_far_idx == e1.p1_idx:
            far1 = p1
        elif self.e1_far_idx == e1.p2_idx:
            far1 = p2
        else:
            far1 = _get_far_point(ix, iy, p1, p2)

        if self.e2_far_idx == e2.p1_idx:
            far2 = p3
        elif self.e2_far_idx == e2.p2_idx:
            far2 = p4
        else:
            far2 = _get_far_point(ix, iy, p3, p4)

        return far1, far2

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> float:
        result = self._get_line_params(reg)
        if result is None:
            logger.warning("_get_line_params returned None")
            return 0.0

        e1, e2, p1, p2, p3, p4 = result

        intersection = line_intersection(
            (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)
        )

        if intersection is None:
            logger.warning("no intersection (parallel lines)")
            return 0.0

        ix, iy = intersection

        far1, far2 = self._get_far_points(e1, e2, p1, p2, p3, p4, ix, iy)

        dx1 = far1.x - ix
        dy1 = far1.y - iy
        dx2 = far2.x - ix
        dy2 = far2.y - iy

        len1_sq = dx1 * dx1 + dy1 * dy1
        len2_sq = dx2 * dx2 + dy2 * dy2

        if len1_sq < 1e-12 or len2_sq < 1e-12:
            logger.debug(
                f"error: zero length, len1_sq={len1_sq}, len2_sq={len2_sq}"
            )
            return 0.0

        anchor_dir = math.atan2(dy1, dx1)
        other_dir = math.atan2(dy2, dx2)

        current = normalize_angle(anchor_dir - other_dir)

        target = math.radians(self.value)

        diff = current - target
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff <= -math.pi:
            diff += 2 * math.pi

        return diff * ANGLE_WEIGHT

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[int, List[Tuple[float, float]]]:
        result = self._get_line_params(reg)
        if result is None:
            return {}

        e1, e2, p1, p2, p3, p4 = result

        intersection = line_intersection(
            (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)
        )

        if intersection is None:
            return {}

        ix, iy = intersection

        far1, far2 = self._get_far_points(e1, e2, p1, p2, p3, p4, ix, iy)

        dx1 = far1.x - ix
        dy1 = far1.y - iy
        dx2 = far2.x - ix
        dy2 = far2.y - iy

        len1_sq = dx1 * dx1 + dy1 * dy1
        len2_sq = dx2 * dx2 + dy2 * dy2

        if len1_sq < 1e-12 or len2_sq < 1e-12:
            return {}

        w = ANGLE_WEIGHT

        d_dir1_d_far1x = -dy1 / len1_sq
        d_dir1_d_far1y = dx1 / len1_sq
        d_dir1_d_near1x = dy1 / len1_sq
        d_dir1_d_near1y = -dx1 / len1_sq

        d_dir2_d_far2x = -dy2 / len2_sq
        d_dir2_d_far2y = dx2 / len2_sq
        d_dir2_d_near2x = dy2 / len2_sq
        d_dir2_d_near2y = -dx2 / len2_sq

        d_error_d_dir1 = w
        d_error_d_dir2 = -w

        grads = {}

        if far1 == p1:
            grads[e1.p1_idx] = [
                (
                    d_error_d_dir1 * d_dir1_d_far1x,
                    d_error_d_dir1 * d_dir1_d_far1y,
                )
            ]
            grads[e1.p2_idx] = [
                (
                    d_error_d_dir1 * d_dir1_d_near1x,
                    d_error_d_dir1 * d_dir1_d_near1y,
                )
            ]
        else:
            grads[e1.p1_idx] = [
                (
                    d_error_d_dir1 * d_dir1_d_near1x,
                    d_error_d_dir1 * d_dir1_d_near1y,
                )
            ]
            grads[e1.p2_idx] = [
                (
                    d_error_d_dir1 * d_dir1_d_far1x,
                    d_error_d_dir1 * d_dir1_d_far1y,
                )
            ]

        if far2 == p3:
            grads[e2.p1_idx] = [
                (
                    d_error_d_dir2 * d_dir2_d_far2x,
                    d_error_d_dir2 * d_dir2_d_far2y,
                )
            ]
            grads[e2.p2_idx] = [
                (
                    d_error_d_dir2 * d_dir2_d_near2x,
                    d_error_d_dir2 * d_dir2_d_near2y,
                )
            ]
        else:
            grads[e2.p1_idx] = [
                (
                    d_error_d_dir2 * d_dir2_d_near2x,
                    d_error_d_dir2 * d_dir2_d_near2y,
                )
            ]
            grads[e2.p2_idx] = [
                (
                    d_error_d_dir2 * d_dir2_d_far2x,
                    d_error_d_dir2 * d_dir2_d_far2y,
                )
            ]

        return grads

    def get_visuals(
        self,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
    ):
        result = self._get_line_params(reg)
        if result is None:
            return None

        e1, e2, p1, p2, p3, p4 = result

        intersection = line_intersection(
            (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)
        )

        if intersection is None:
            mx1 = (p1.x + p2.x) / 2
            my1 = (p1.y + p2.y) / 2
            mx2 = (p3.x + p4.x) / 2
            my2 = (p3.y + p4.y) / 2
            intersection = ((mx1 + mx2) / 2, (my1 + my2) / 2)

        ix, iy = intersection
        sx, sy = to_screen((ix, iy))

        far1, far2 = self._get_far_points(e1, e2, p1, p2, p3, p4, ix, iy)

        far1_screen = to_screen((far1.x, far1.y))
        far2_screen = to_screen((far2.x, far2.y))

        anchor_ang = math.atan2(far1_screen[1] - sy, far1_screen[0] - sx)
        other_ang = math.atan2(far2_screen[1] - sy, far2_screen[0] - sx)

        return sx, sy, anchor_ang, other_ang

    def get_label_pos(
        self,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
    ):
        visuals = self.get_visuals(reg, to_screen)
        if visuals is None:
            return None

        sx, sy, anchor_ang, other_ang = visuals

        cw_diff = normalize_angle(anchor_ang - other_ang)

        mid_angle = other_ang + cw_diff / 2
        radius = 28.0
        label_x = sx + math.cos(mid_angle) * radius
        label_y = sy + math.sin(mid_angle) * radius

        return label_x, label_y

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        element: Any,
        threshold: float,
    ) -> bool:
        visuals = self.get_visuals(reg, to_screen)
        if visuals is None:
            return False

        cx, cy, anchor_ang, other_ang = visuals

        dist = math.hypot(sx - cx, sy - cy)
        if abs(dist - ARC_RADIUS) > threshold:
            return False

        click_ang = math.atan2(sy - cy, sx - cx)

        ccw_diff = normalize_angle(other_ang - anchor_ang)

        click_from_anchor = normalize_angle(click_ang - anchor_ang)

        return click_from_anchor <= ccw_diff

    def draw(
        self,
        ctx: "cairo.Context",
        registry: "EntityRegistry",
        to_screen: Callable[[Tuple[float, float]], Tuple[float, float]],
        is_selected: bool = False,
        is_hovered: bool = False,
        point_radius: float = 5.0,
    ) -> None:
        visuals = self.get_visuals(registry, to_screen)
        if visuals is None:
            return

        sx, sy, anchor_ang, other_ang = visuals

        ctx.save()
        ctx.set_line_width(1.5)

        ctx.new_sub_path()
        ctx.arc(sx, sy, ARC_RADIUS, anchor_ang, other_ang)

        if is_selected:
            self._draw_selection_underlay(ctx)

        if self.status == ConstraintStatus.CONFLICTING:
            self._draw_conflict_underlay(ctx)

        self._set_color(ctx, is_hovered)
        ctx.stroke()

        label = self._format_value() + "°"
        ext = ctx.text_extents(label)

        ccw_diff = normalize_angle(other_ang - anchor_ang)

        mid = anchor_ang + ccw_diff / 2
        label_x = sx + math.cos(mid) * LABEL_RADIUS
        label_y = sy + math.sin(mid) * LABEL_RADIUS

        if is_selected:
            ctx.set_source_rgba(0.2, 0.6, 1.0, 0.4)
        elif is_hovered:
            ctx.set_source_rgba(1.0, 0.95, 0.85, 0.9)
        elif self.status == ConstraintStatus.CONFLICTING:
            ctx.set_source_rgba(1.0, 0.6, 0.6, 0.9)
        elif self.status == ConstraintStatus.ERROR:
            ctx.set_source_rgba(1.0, 0.8, 0.8, 0.9)
        elif self.status == ConstraintStatus.EXPRESSION_BASED:
            ctx.set_source_rgba(1.0, 0.9, 0.7, 0.9)
        else:
            ctx.set_source_rgba(1, 1, 1, 0.8)

        bg_x = label_x - ext.width / 2 - 4
        bg_y = label_y - ext.height / 2 - 4
        ctx.rectangle(bg_x, bg_y, ext.width + 8, ext.height + 8)
        ctx.fill()
        ctx.new_path()

        if self.status in (
            ConstraintStatus.ERROR,
            ConstraintStatus.CONFLICTING,
        ):
            ctx.set_source_rgb(0.8, 0.0, 0.0)
        else:
            ctx.set_source_rgb(0, 0, 0.5)

        ctx.move_to(label_x - ext.width / 2, label_y + ext.height / 2 - 2)
        ctx.show_text(label)
        ctx.new_path()

        ctx.restore()

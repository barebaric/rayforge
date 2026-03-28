from __future__ import annotations
import math
from typing import (
    Optional,
    Union,
    Dict,
    Any,
    List,
    Callable,
    TYPE_CHECKING,
    cast,
)
from gettext import gettext as _
from rayforge.core.geo import Point
from ..entities import Line, Arc, Circle, Ellipse
from ..types import EntityID
from .base import Constraint, ConstraintStatus

if TYPE_CHECKING:
    import cairo
    from ..params import ParameterContext
    from ..registry import EntityRegistry
    from ..selection import SketchSelection
    from ..sketch import Sketch


class EqualLengthConstraint(Constraint):
    """
    Enforces that all entities in a set have the same characteristic length.
    - Line: Length
    - Arc/Circle: Radius
    - Ellipse: Both radii (X and Y)
    """

    def __init__(self, entity_ids: List[EntityID], user_visible: bool = True):
        super().__init__(user_visible=user_visible)
        self.entity_ids = entity_ids

    @classmethod
    def get_type_key(cls) -> str:
        return "equal"

    @classmethod
    def can_apply_to(
        cls, selection: "SketchSelection", sketch: Optional["Sketch"] = None
    ) -> bool:
        if selection.point_ids or len(selection.entity_ids) < 2:
            return False
        if sketch is None:
            return False
        entities = [
            sketch.registry.get_entity(eid) for eid in selection.entity_ids
        ]
        return all(
            isinstance(e, (Line, Arc, Circle, Ellipse)) and e is not None
            for e in entities
        )

    @staticmethod
    def get_type_name() -> str:
        """Returns to human-readable name of this constraint type."""
        return _("Equal Length")

    def get_title(self) -> str:
        """Returns a human-readable title for this constraint."""
        return self.get_type_name()

    def get_subtitle(self, registry: "EntityRegistry") -> str:
        """Returns subtitle describing constrained entities."""
        if len(self.entity_ids) >= 2:
            return _("{} entities").format(len(self.entity_ids))
        return ""

    def targets_segment(
        self, p1: EntityID, p2: EntityID, entity_id: Optional[EntityID]
    ) -> bool:
        if entity_id is not None:
            return entity_id in self.entity_ids
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "EqualLengthConstraint",
            "entity_ids": self.entity_ids,
            "user_visible": self.user_visible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EqualLengthConstraint":
        return cls(
            entity_ids=data["entity_ids"],
            user_visible=data.get("user_visible", True),
        )

    def constrains_radius(
        self, registry: "EntityRegistry", entity_id: EntityID
    ) -> bool:
        return entity_id in self.entity_ids

    @staticmethod
    def _get_length_pairs(entity):
        """Returns point-index pairs defining the entity's length(s)."""
        if isinstance(entity, Line):
            return [(entity.p1_idx, entity.p2_idx)]
        elif isinstance(entity, Arc):
            return [(entity.center_idx, entity.start_idx)]
        elif isinstance(entity, Circle):
            return [(entity.center_idx, entity.radius_pt_idx)]
        elif isinstance(entity, Ellipse):
            return [
                (entity.center_idx, entity.radius_x_pt_idx),
                (entity.center_idx, entity.radius_y_pt_idx),
            ]
        return []

    @staticmethod
    def _pair_dist(pa_idx, pb_idx, reg):
        pa = reg.get_point(pa_idx)
        pb = reg.get_point(pb_idx)
        return math.hypot(pb.x - pa.x, pb.y - pa.y)

    def _get_length(self, entity, reg: "EntityRegistry") -> float:
        if isinstance(entity, Line):
            p1 = reg.get_point(entity.p1_idx)
            p2 = reg.get_point(entity.p2_idx)
            return math.hypot(p2.x - p1.x, p2.y - p1.y)
        elif isinstance(entity, Arc):
            c = reg.get_point(entity.center_idx)
            s = reg.get_point(entity.start_idx)
            return math.hypot(s.x - c.x, s.y - c.y)
        elif isinstance(entity, Circle):
            c = reg.get_point(entity.center_idx)
            r = reg.get_point(entity.radius_pt_idx)
            return math.hypot(r.x - c.x, r.y - c.y)
        elif isinstance(entity, Ellipse):
            rx, ry = entity._get_radii(reg)
            return (rx + ry) / 2.0
        return 0.0

    def error(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> List[float]:
        if len(self.entity_ids) < 2:
            return []

        entities = [reg.get_entity(eid) for eid in self.entity_ids]
        if any(e is None for e in entities):
            return []

        errors = []
        base = entities[0]
        base_pairs = self._get_length_pairs(base)
        base_lens = [self._pair_dist(pa, pb, reg) for pa, pb in base_pairs]

        for i in range(1, len(entities)):
            ent = entities[i]
            ent_pairs = self._get_length_pairs(ent)
            ent_lens = [self._pair_dist(pa, pb, reg) for pa, pb in ent_pairs]

            n = max(len(base_pairs), len(ent_pairs))
            for j in range(n):
                bl = base_lens[j] if j < len(base_lens) else base_lens[0]
                el = ent_lens[j] if j < len(ent_lens) else ent_lens[0]
                errors.append(el - bl)
        return errors

    def gradient(
        self, reg: "EntityRegistry", params: "ParameterContext"
    ) -> Dict[EntityID, List[Point]]:
        if len(self.entity_ids) < 2:
            return {}

        entities = [
            cast(Union[Line, Arc, Circle, Ellipse], reg.get_entity(eid))
            for eid in self.entity_ids
        ]
        if any(e is None for e in entities):
            return {}

        base_pairs = self._get_length_pairs(entities[0])
        num_residuals = 0
        for i in range(1, len(entities)):
            ent_pairs = self._get_length_pairs(entities[i])
            num_residuals += max(len(base_pairs), len(ent_pairs))

        grad = {}

        def add_grad(pid, r_idx, gx, gy):
            if pid not in grad:
                grad[pid] = [(0.0, 0.0)] * num_residuals
            curr = list(grad[pid])
            ox, oy = curr[r_idx]
            curr[r_idx] = (ox + gx, oy + gy)
            grad[pid] = curr

        row = 0
        for i in range(1, len(entities)):
            ent = entities[i]
            ent_pairs = self._get_length_pairs(ent)
            n = max(len(base_pairs), len(ent_pairs))

            for j in range(n):
                bp = base_pairs[j] if j < len(base_pairs) else base_pairs[0]
                ep = ent_pairs[j] if j < len(ent_pairs) else ent_pairs[0]

                b_pta = reg.get_point(bp[0])
                b_ptb = reg.get_point(bp[1])
                b_dx = b_ptb.x - b_pta.x
                b_dy = b_ptb.y - b_pta.y
                b_len = math.hypot(b_dx, b_dy)
                if b_len > 1e-9:
                    b_ux, b_uy = b_dx / b_len, b_dy / b_len
                else:
                    b_ux, b_uy = 0.0, 0.0

                e_pta = reg.get_point(ep[0])
                e_ptb = reg.get_point(ep[1])
                e_dx = e_ptb.x - e_pta.x
                e_dy = e_ptb.y - e_pta.y
                e_len = math.hypot(e_dx, e_dy)
                if e_len > 1e-9:
                    e_ux, e_uy = e_dx / e_len, e_dy / e_len
                else:
                    e_ux, e_uy = 0.0, 0.0

                add_grad(bp[0], row, b_ux, b_uy)
                add_grad(bp[1], row, -b_ux, -b_uy)
                add_grad(ep[0], row, -e_ux, -e_uy)
                add_grad(ep[1], row, e_ux, e_uy)

                row += 1

        return grad

    def _get_symbol_pos(
        self,
        entity,
        reg: "EntityRegistry",
        to_screen: Callable[[Point], Point],
    ):
        """Calculates screen pos for an equality symbol on an entity."""
        # 1. Get anchor point (mid_x, mid_y) and normal_angle in MODEL space
        mid_x, mid_y, normal_angle = 0.0, 0.0, 0.0

        if isinstance(entity, Line):
            p1 = reg.get_point(entity.p1_idx)
            p2 = reg.get_point(entity.p2_idx)
            mid_x = (p1.x + p2.x) / 2.0
            mid_y = (p1.y + p2.y) / 2.0
            tangent_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            normal_angle = tangent_angle - (math.pi / 2.0)
        elif isinstance(entity, (Arc, Circle, Ellipse)):
            midpoint = entity.get_midpoint(reg)
            if not midpoint:
                return None
            mid_x, mid_y = midpoint
            center = reg.get_point(entity.center_idx)
            normal_angle = math.atan2(mid_y - center.y, mid_x - center.x)

        # Estimate scale from transform
        p0 = to_screen((0, 0))
        p1 = to_screen((1, 0))
        scale = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if scale < 1e-9:
            scale = 1.0

        offset_dist_model = 15.0 / scale
        final_x = mid_x + offset_dist_model * math.cos(normal_angle)
        final_y = mid_y + offset_dist_model * math.sin(normal_angle)
        return to_screen((final_x, final_y))

    def is_hit(
        self,
        sx: float,
        sy: float,
        reg: "EntityRegistry",
        to_screen: Callable[[Point], Point],
        element: Any,
        threshold: float,
    ) -> bool:
        for entity_id in self.entity_ids:
            entity = reg.get_entity(entity_id)
            if not entity:
                continue
            pos = self._get_symbol_pos(entity, reg, to_screen)
            if pos:
                esx, esy = pos
                if math.hypot(sx - esx, sy - esy) < threshold:
                    return True
        return False

    def draw(
        self,
        ctx: "cairo.Context",
        registry: "EntityRegistry",
        to_screen: Callable[[Point], Point],
        is_selected: bool = False,
        is_hovered: bool = False,
        point_radius: float = 5.0,
    ) -> None:
        for entity_id in self.entity_ids:
            entity = registry.get_entity(entity_id)
            if not entity:
                continue

            pos = self._get_symbol_pos(entity, registry, to_screen)
            if not pos:
                continue

            sx, sy = pos
            ctx.save()

            if is_selected:
                ctx.set_source_rgba(0.2, 0.6, 1.0, 0.4)
                ctx.arc(sx, sy, 10, 0, 2 * math.pi)
                ctx.fill()

            if self.status == ConstraintStatus.CONFLICTING:
                ctx.set_source_rgba(1.0, 0.2, 0.2, 0.5)
                ctx.arc(sx, sy, 12, 0, 2 * math.pi)
                ctx.fill()

            self._set_color(ctx, is_hovered)
            ctx.set_font_size(16)
            ext = ctx.text_extents("=")
            ctx.move_to(sx - ext.width / 2, sy + ext.height / 2)
            ctx.show_text("=")
            ctx.restore()
            ctx.new_path()

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from raygeo.geo import Geometry
from raygeo.geo.types import Point, Rect

from ..types import EntityID
from .entity import Entity, OffsetPlan

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..registry import EntityRegistry
    from ..sketch import Sketch


class Ellipse(Entity):
    def __init__(
        self,
        id: EntityID,
        center_idx: EntityID,
        radius_x_pt_idx: EntityID,
        radius_y_pt_idx: EntityID,
        construction: bool = False,
        helper_line_ids: list[EntityID] | None = None,
    ):
        super().__init__(id, construction)
        self.center_idx: EntityID = center_idx
        self.radius_x_pt_idx: EntityID = radius_x_pt_idx
        self.radius_y_pt_idx: EntityID = radius_y_pt_idx
        self.helper_line_ids: list[EntityID] = helper_line_ids or []
        self.type = "ellipse"

    def get_point_ids(self) -> list[EntityID]:
        return [self.center_idx, self.radius_x_pt_idx, self.radius_y_pt_idx]

    def get_endpoint_ids(self) -> list[EntityID]:
        return []

    def get_junction_point_ids(self) -> list[EntityID]:
        return [self.center_idx, self.radius_x_pt_idx, self.radius_y_pt_idx]

    def get_rigidly_connected_points(
        self, point_id: EntityID
    ) -> list[EntityID]:
        if point_id == self.center_idx:
            return [
                self.center_idx,
                self.radius_x_pt_idx,
                self.radius_y_pt_idx,
            ]
        return []

    def get_drag_anchor_points(self, point_id: EntityID) -> list[EntityID]:
        if point_id in (self.radius_x_pt_idx, self.radius_y_pt_idx):
            return [self.center_idx]
        return []

    def hit_test(
        self,
        mx: float,
        my: float,
        threshold: float,
        registry: "EntityRegistry",
    ) -> bool:
        center = registry.get_point(self.center_idx)
        radius_x_pt = registry.get_point(self.radius_x_pt_idx)
        radius_y_pt = registry.get_point(self.radius_y_pt_idx)
        if not (center and radius_x_pt and radius_y_pt):
            return False

        rx = math.hypot(radius_x_pt.x - center.x, radius_x_pt.y - center.y)
        ry = math.hypot(radius_y_pt.x - center.x, radius_y_pt.y - center.y)
        if rx < 1e-9 or ry < 1e-9:
            return False

        rotation = self._get_rotation(registry)
        cos_a = math.cos(-rotation)
        sin_a = math.sin(-rotation)

        dx = mx - center.x
        dy = my - center.y
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a

        dist = math.sqrt((local_x / rx) ** 2 + (local_y / ry) ** 2)
        return abs(dist - 1.0) < (threshold / min(rx, ry))

    def get_ignorable_unconstrained_points(self) -> list[EntityID]:
        if self.constrained:
            return [self.radius_x_pt_idx, self.radius_y_pt_idx]
        return []

    def update_constrained_status(
        self, registry: "EntityRegistry", constraints: Sequence["Constraint"]
    ) -> None:
        center_pt = registry.get_point(self.center_idx)
        radius_x_pt = registry.get_point(self.radius_x_pt_idx)
        radius_y_pt = registry.get_point(self.radius_y_pt_idx)

        center_is_constrained = center_pt.constrained
        radii_are_defined = radius_x_pt.constrained and radius_y_pt.constrained

        if not radii_are_defined:
            for constr in constraints:
                if constr.constrains_radius(registry, self.id):
                    radii_are_defined = True
                    break

        self.constrained = center_is_constrained and radii_are_defined

    def _get_radii(self, registry: "EntityRegistry") -> tuple[float, float]:
        center = registry.get_point(self.center_idx)
        radius_x_pt = registry.get_point(self.radius_x_pt_idx)
        radius_y_pt = registry.get_point(self.radius_y_pt_idx)
        rx = math.hypot(radius_x_pt.x - center.x, radius_x_pt.y - center.y)
        ry = math.hypot(radius_y_pt.x - center.x, radius_y_pt.y - center.y)
        return rx, ry

    def _get_rotation(self, registry: "EntityRegistry") -> float:
        center = registry.get_point(self.center_idx)
        radius_x_pt = registry.get_point(self.radius_x_pt_idx)
        return math.atan2(radius_x_pt.y - center.y, radius_x_pt.x - center.x)

    def is_contained_by(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        center = registry.get_point(self.center_idx)
        rx, ry = self._get_radii(registry)
        return (
            (center.x - rx) >= rect[0]
            and (center.y - ry) >= rect[1]
            and (center.x + rx) <= rect[2]
            and (center.y + ry) <= rect[3]
        )

    def intersects_rect(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        center = registry.get_point(self.center_idx)
        rx, ry = self._get_radii(registry)

        if self.is_contained_by(rect, registry):
            return True

        closest_x = max(rect[0], min(center.x, rect[2]))
        closest_y = max(rect[1], min(center.y, rect[3]))
        dx = closest_x - center.x
        dy = closest_y - center.y
        dist_sq = (
            (dx / rx) ** 2 + (dy / ry) ** 2
            if rx > 0 and ry > 0
            else float("inf")
        )

        if dist_sq > 1.0:
            return False

        dx_far = max(abs(rect[0] - center.x), abs(rect[2] - center.x))
        dy_far = max(abs(rect[1] - center.y), abs(rect[3] - center.y))
        dist_sq_far = (
            (dx_far / rx) ** 2 + (dy_far / ry) ** 2
            if rx > 0 and ry > 0
            else float("inf")
        )
        return dist_sq_far >= 1.0

    def to_geometry(self, registry: "EntityRegistry") -> Geometry:
        geo = Geometry()
        center = registry.get_point(self.center_idx)
        radius_x_pt = registry.get_point(self.radius_x_pt_idx)
        radius_y_pt = registry.get_point(self.radius_y_pt_idx)

        rx = math.hypot(radius_x_pt.x - center.x, radius_x_pt.y - center.y)
        ry = math.hypot(radius_y_pt.x - center.x, radius_y_pt.y - center.y)

        if rx < 1e-9 or ry < 1e-9:
            return geo

        cx, cy = center.x, center.y
        rotation = self._get_rotation(registry)
        cos_a = math.cos(rotation)
        sin_a = math.sin(rotation)

        if abs(rx - ry) < 1e-9:
            return self._circle_geometry(cx, cy, rx, cos_a, sin_a)

        # Eccentricity-driven sampling, capped to keep tessellation
        # bounded even for extreme aspect ratios.
        ratio = max(rx, ry) / min(rx, ry)
        num_segments = min(max(32, int(64 * ratio)), 256)
        for i in range(num_segments):
            angle = 2 * math.pi * i / num_segments
            local_x = rx * math.cos(angle)
            local_y = ry * math.sin(angle)
            x = cx + local_x * cos_a - local_y * sin_a
            y = cy + local_x * sin_a + local_y * cos_a
            if i == 0:
                geo.move_to(x, y)
            else:
                geo.line_to(x, y)
        geo.close_path()
        geo.fit_arcs(0.1)
        return geo

    @staticmethod
    def _circle_geometry(
        cx: float,
        cy: float,
        r: float,
        cos_a: float,
        sin_a: float,
    ) -> Geometry:
        geo = Geometry()
        start_x = cx + r * cos_a
        start_y = cy + r * sin_a
        mid_x = cx - r * cos_a
        mid_y = cy - r * sin_a
        i1 = -r * cos_a
        j1 = -r * sin_a
        i2 = r * cos_a
        j2 = r * sin_a
        geo.move_to(start_x, start_y)
        geo.arc_to(mid_x, mid_y, i1, j1, clockwise=False)
        geo.arc_to(start_x, start_y, i2, j2, clockwise=False)
        return geo

    def create_fill_geometry(
        self, registry: "EntityRegistry"
    ) -> Geometry | None:
        return self.to_geometry(registry)

    def as_offset_item(self, sketch: "Sketch") -> "Ellipse":
        """An ellipse offsets on its own, updated in place."""
        return self

    def plan_offset(
        self,
        registry: "EntityRegistry",
        offset: float,
        allocate_id: Callable[[], EntityID],
    ) -> OffsetPlan | None:
        """Updates the ellipse in place: both radius points move
        radially so each semi-axis grows by the offset. The result
        stays an ellipse (an approximation of the true offset)."""
        center = registry.get_point(self.center_idx)
        radius_x_pt = registry.get_point(self.radius_x_pt_idx)
        radius_y_pt = registry.get_point(self.radius_y_pt_idx)
        rx, ry = self._get_radii(registry)
        if rx + offset <= 1e-9 or ry + offset <= 1e-9:
            return None

        plan = OffsetPlan()
        for pid, pt, radius in (
            (self.radius_x_pt_idx, radius_x_pt, rx),
            (self.radius_y_pt_idx, radius_y_pt, ry),
        ):
            scale = (radius + offset) / radius
            plan.point_moves[pid] = (
                center.x + (pt.x - center.x) * scale,
                center.y + (pt.y - center.y) * scale,
            )
        return plan

    def preview_polylines(
        self, registry: "EntityRegistry", offset: float
    ) -> list[list[tuple[float, float]]]:
        """Parametric sampling of the ellipse grown by the offset, for
        live preview (matches plan_offset's in-place update)."""
        center = registry.get_point(self.center_idx)
        rx, ry = self._get_radii(registry)
        if rx + offset <= 1e-9 or ry + offset <= 1e-9:
            return []
        rotation = self._get_rotation(registry)
        cos_a = math.cos(rotation)
        sin_a = math.sin(rotation)
        segments = 64
        return [
            [
                (
                    center.x
                    + (rx + offset) * math.cos(t) * cos_a
                    - (ry + offset) * math.sin(t) * sin_a,
                    center.y
                    + (rx + offset) * math.cos(t) * sin_a
                    + (ry + offset) * math.sin(t) * cos_a,
                )
                for t in (
                    2 * math.pi * i / segments for i in range(segments + 1)
                )
            ]
        ]

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "center_idx": self.center_idx,
                "radius_x_pt_idx": self.radius_x_pt_idx,
                "radius_y_pt_idx": self.radius_y_pt_idx,
                "helper_line_ids": self.helper_line_ids,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Ellipse":
        return cls(
            id=data["id"],
            center_idx=data["center_idx"],
            radius_x_pt_idx=data["radius_x_pt_idx"],
            radius_y_pt_idx=data["radius_y_pt_idx"],
            construction=data.get("construction", False),
            helper_line_ids=data.get("helper_line_ids"),
        )

    def get_midpoint(self, registry: "EntityRegistry") -> Point | None:
        radius_x_pt = registry.get_point(self.radius_x_pt_idx)
        if not radius_x_pt:
            return None
        return radius_x_pt.pos()

    def __repr__(self) -> str:
        return (
            f"Ellipse(id={self.id}, center={self.center_idx}, "
            f"radius_x_pt={self.radius_x_pt_idx}, "
            f"radius_y_pt={self.radius_y_pt_idx}, "
            f"construction={self.construction})"
        )

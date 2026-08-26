import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from raygeo.geo import Geometry
from raygeo.geo.shape.line import does_line_segment_intersect_rect
from raygeo.geo.shape.polygon import is_point_inside_polygon
from raygeo.geo.shape.text import FontConfig, text_to_geometry
from raygeo.geo.types import Rect

from rayforge.core.color import ColorRGBA

from ..types import EntityID
from .entity import Entity
from .line import Line

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..registry import EntityRegistry

logger = logging.getLogger(__name__)


class TextBoxEntity(Entity):
    def __init__(
        self,
        id: EntityID,
        origin_id: EntityID,
        width_id: EntityID,
        height_id: EntityID,
        content: str = "",
        font_config: FontConfig | None = None,
        construction: bool = False,
        construction_line_ids: list[EntityID] | None = None,
    ):
        super().__init__(id, construction)
        self.origin_id: EntityID = origin_id
        self.width_id: EntityID = width_id
        self.height_id: EntityID = height_id
        self.content = content
        self.font_config = font_config or FontConfig()
        self.construction_line_ids: list[EntityID] = (
            construction_line_ids or []
        )
        self.fill_color: ColorRGBA | None = None
        self.type = "text_box"

    def get_point_ids(self) -> list[EntityID]:
        return [self.origin_id, self.width_id, self.height_id]

    def get_endpoint_ids(self) -> list[EntityID]:
        return []

    def get_junction_point_ids(self) -> list[EntityID]:
        return []

    def get_frame_polygon(
        self, registry: "EntityRegistry"
    ) -> list[tuple[float, float]] | None:
        """Returns the 4-corner frame polygon, or None if points are
        missing.  Corner 4 is taken from the construction line when
        present, otherwise computed from the parallelogram identity."""
        p_origin = registry.get_point(self.origin_id)
        p_width = registry.get_point(self.width_id)
        p_height = registry.get_point(self.height_id)
        if not (p_origin and p_width and p_height):
            return None

        p4_id = self.get_fourth_corner_id(registry)
        if p4_id is not None:
            p4 = registry.get_point(p4_id)
            if p4:
                p4_x, p4_y = p4.x, p4.y
            else:
                p4_x = p_width.x + p_height.x - p_origin.x
                p4_y = p_width.y + p_height.y - p_origin.y
        else:
            p4_x = p_width.x + p_height.x - p_origin.x
            p4_y = p_width.y + p_height.y - p_origin.y

        return [
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p4_x, p4_y),
            (p_height.x, p_height.y),
        ]

    def hit_test(
        self,
        mx: float,
        my: float,
        threshold: float,
        registry: "EntityRegistry",
    ) -> bool:
        polygon = self.get_frame_polygon(registry)
        if polygon is None:
            return False
        return is_point_inside_polygon((mx, my), polygon)

    def get_all_frame_point_ids(
        self, registry: "EntityRegistry"
    ) -> list[EntityID]:
        """Returns all 4 corner points of the text box frame."""
        ids = [self.origin_id, self.width_id, self.height_id]
        p4_id = self.get_fourth_corner_id(registry)
        if p4_id is not None:
            ids.append(p4_id)
        return ids

    def get_font_metrics(self) -> tuple[float, float, float]:
        return self.font_config.get_font_metrics()

    def get_natural_size(
        self, content: str | None = None
    ) -> tuple[float, float]:
        """
        Returns the natural (width, height) of the text content.

        If *content* is omitted, uses ``self.content``. An empty or
        None content yields a minimum width of 10.

        The height is derived from the actual geometry bounding box
        so the frame matches the ink extent exactly and map_to_frame
        does not stretch the text.  raygeo returns Y-UP geometry.`
        """
        text = content if content is not None else self.content

        if not text:
            return 10.0, self.font_config.size

        geo = text_to_geometry(text, font_config=self.font_config)
        _, geo_min_y, _, geo_max_y = geo.rect()
        geo_height = geo_max_y - geo_min_y

        return (
            max(self.font_config.get_text_width(text) or 0, 1.0),
            max(geo_height, 1.0),
        )

    def get_fourth_corner_id(
        self, registry: "EntityRegistry"
    ) -> EntityID | None:
        """Finds the 4th point ID of the text box."""
        for eid in self.construction_line_ids:
            entity = registry.get_entity(eid)
            if isinstance(entity, Line):
                if entity.p1_idx == self.width_id and (
                    entity.p2_idx != self.origin_id
                    and entity.p2_idx != self.height_id
                ):
                    return entity.p2_idx
                if entity.p2_idx == self.width_id and (
                    entity.p1_idx != self.origin_id
                    and entity.p1_idx != self.height_id
                ):
                    return entity.p1_idx
        return None

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        if state is not None:
            state["fill_color"] = self.fill_color
        else:
            state = {"fill_color": self.fill_color}
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        if "fill_color" in state:
            self.fill_color = state["fill_color"]

    def update_constrained_status(
        self, registry: "EntityRegistry", constraints: Sequence["Constraint"]
    ) -> None:
        p_origin = registry.get_point(self.origin_id)
        p_width = registry.get_point(self.width_id)
        p_height = registry.get_point(self.height_id)
        self.constrained = (
            p_origin.constrained
            and p_width.constrained
            and p_height.constrained
        )

    def is_contained_by(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        points = self.get_frame_polygon(registry)
        if points is None:
            return False

        return all(
            rect[0] <= px <= rect[2] and rect[1] <= py <= rect[3]
            for px, py in points
        )

    def intersects_rect(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        points = self.get_frame_polygon(registry)
        if points is None:
            return False

        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            if does_line_segment_intersect_rect(p1, p2, rect):
                return True

        return any(
            rect[0] <= px <= rect[2] and rect[1] <= py <= rect[3]
            for px, py in points
        )

    def _build_frame_for_content(
        self,
        registry: "EntityRegistry",
        content: str,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        """
        Builds a frame (origin, p_width, p_height) whose dimensions match
        *content*'s natural size, preserving direction vectors from the
        current frame.

        Returns (origin, p_width, p_height).
        """
        p_origin = registry.get_point(self.origin_id)
        p_width = registry.get_point(self.width_id)
        p_height = registry.get_point(self.height_id)

        if content == self.content or not self.content:
            logger.debug(
                f"_build_frame: no scaling, content == self.content "
                f"({content!r} == {self.content!r}) or empty"
            )
            return (
                (p_origin.x, p_origin.y),
                (p_width.x, p_width.y),
                (p_height.x, p_height.y),
            )

        nat_w, nat_h = self.get_natural_size(content)

        dx = p_width.x - p_origin.x
        dy = p_width.y - p_origin.y
        frame_w = math.hypot(dx, dy)

        if frame_w < 1e-9:
            w_scale = 1.0
        else:
            w_scale = nat_w / frame_w

        logger.debug(
            f"_build_frame: scaling content={content!r} "
            f"nat_w={nat_w:.2f} frame_w={frame_w:.2f} "
            f"w_scale={w_scale:.4f}"
        )

        scaled_width = (
            p_origin.x + dx * w_scale,
            p_origin.y + dy * w_scale,
        )

        hx = p_height.x - p_origin.x
        hy = p_height.y - p_origin.y
        frame_h = math.hypot(hx, hy)

        if frame_h < 1e-9:
            h_scale = 1.0
        else:
            h_scale = nat_h / frame_h

        scaled_height = (
            p_origin.x + hx * h_scale,
            p_origin.y + hy * h_scale,
        )

        return (
            (p_origin.x, p_origin.y),
            scaled_width,
            scaled_height,
        )

    def to_geometry(
        self,
        registry: "EntityRegistry",
        resolved_content: str | None = None,
    ) -> Geometry:
        """Converts the text box to a Geometry object."""
        text = (
            resolved_content if resolved_content is not None else self.content
        )
        origin, pw, ph = self._build_frame_for_content(registry, text)
        txt_geo = text_to_geometry(text, font_config=self.font_config)
        _, geo_min_y, _, geo_max_y = txt_geo.rect()
        advance_width = self.font_config.get_text_width(text) or 1.0
        return txt_geo.map_to_frame(
            origin,
            pw,
            ph,
            anchor_x=0.0,
            stable_src_width=advance_width,
            anchor_y=geo_min_y,
            stable_src_height=geo_max_y - geo_min_y,
        )

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "origin_id": self.origin_id,
                "width_id": self.width_id,
                "height_id": self.height_id,
                "content": self.content,
                "font_config": self.font_config.to_dict(),
                "construction_line_ids": self.construction_line_ids,
            }
        )
        if self.fill_color is not None:
            data["fill_color"] = list(self.fill_color)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextBoxEntity":
        fill_color_raw = data.get("fill_color")
        fill_color = (
            tuple(fill_color_raw) if fill_color_raw is not None else None
        )
        entity = cls(
            id=data["id"],
            origin_id=data["origin_id"],
            width_id=data["width_id"],
            height_id=data["height_id"],
            content=data.get("content", ""),
            font_config=FontConfig.from_dict(data.get("font_config")),
            construction=data.get("construction", False),
            construction_line_ids=data.get("construction_line_ids"),
        )
        entity.fill_color = fill_color
        return entity

    def __repr__(self) -> str:
        return (
            f"TextBoxEntity(id={self.id}, origin={self.origin_id}, "
            f"width={self.width_id}, height={self.height_id}, "
            f"content='{self.content}')"
        )

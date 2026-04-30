import logging
import math
from typing import List, Tuple, Dict, Any, Sequence, Optional, TYPE_CHECKING
from rayforge.core.color import ColorRGBA
from rayforge.core.geo import primitives, Rect
from rayforge.core.geo.geometry import Geometry
from rayforge.core.geo.font_config import FontConfig
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
        font_config: Optional[FontConfig] = None,
        construction: bool = False,
        construction_line_ids: Optional[List[EntityID]] = None,
    ):
        super().__init__(id, construction)
        self.origin_id: EntityID = origin_id
        self.width_id: EntityID = width_id
        self.height_id: EntityID = height_id
        self.content = content
        self.font_config = font_config or FontConfig()
        self.construction_line_ids: List[EntityID] = (
            construction_line_ids or []
        )
        self.fill_color: Optional[ColorRGBA] = None
        self.type = "text_box"

    def get_point_ids(self) -> List[EntityID]:
        return [self.origin_id, self.width_id, self.height_id]

    def get_endpoint_ids(self) -> List[EntityID]:
        return []

    def get_junction_point_ids(self) -> List[EntityID]:
        return []

    def hit_test(
        self,
        mx: float,
        my: float,
        threshold: float,
        registry: "EntityRegistry",
    ) -> bool:
        p_origin = registry.get_point(self.origin_id)
        p_width = registry.get_point(self.width_id)
        p_height = registry.get_point(self.height_id)
        if not (p_origin and p_width and p_height):
            return False

        p4_x = p_width.x + p_height.x - p_origin.x
        p4_y = p_width.y + p_height.y - p_origin.y

        polygon = [
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p4_x, p4_y),
            (p_height.x, p_height.y),
        ]

        return primitives.is_point_in_polygon((mx, my), polygon)

    def get_all_frame_point_ids(
        self, registry: "EntityRegistry"
    ) -> List[EntityID]:
        """Returns all 4 corner points of the text box frame."""
        ids = [self.origin_id, self.width_id, self.height_id]
        p4_id = self.get_fourth_corner_id(registry)
        if p4_id is not None:
            ids.append(p4_id)
        return ids

    def get_font_metrics(self) -> Tuple[float, float, float]:
        return self.font_config.get_font_metrics()

    def get_natural_size(
        self, content: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Returns the natural (width, height) of the text content.

        If *content* is omitted, uses ``self.content``. An empty or
        None content yields a minimum width of 10.
        """
        text = content if content is not None else self.content
        _, _, font_height = self.get_font_metrics()

        if not text:
            return 10.0, font_height

        geo = Geometry.from_text(text, self.font_config)
        geo.flip_y()
        min_x, _, max_x, _ = geo.rect()
        return max(max_x - min_x, 1.0), font_height

    def get_fourth_corner_id(
        self, registry: "EntityRegistry"
    ) -> Optional[EntityID]:
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

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        if state is not None:
            state["fill_color"] = self.fill_color
        else:
            state = {"fill_color": self.fill_color}
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
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
        p_origin = registry.get_point(self.origin_id)
        p_width = registry.get_point(self.width_id)
        p_height = registry.get_point(self.height_id)

        p4_x = p_width.x + p_height.x - p_origin.x
        p4_y = p_width.y + p_height.y - p_origin.y

        points = [
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p4_x, p4_y),
            (p_height.x, p_height.y),
        ]

        return all(
            rect[0] <= px <= rect[2] and rect[1] <= py <= rect[3]
            for px, py in points
        )

    def intersects_rect(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        p_origin = registry.get_point(self.origin_id)
        p_width = registry.get_point(self.width_id)
        p_height = registry.get_point(self.height_id)

        p4_x = p_width.x + p_height.x - p_origin.x
        p4_y = p_width.y + p_height.y - p_origin.y

        points = [
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p4_x, p4_y),
            (p_height.x, p_height.y),
        ]

        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            if primitives.line_segment_intersects_rect(p1, p2, rect):
                return True

        return any(
            rect[0] <= px <= rect[2] and rect[1] <= py <= rect[3]
            for px, py in points
        )

    def _build_frame_for_content(
        self,
        registry: "EntityRegistry",
        content: str,
    ) -> Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        float,
        float,
    ]:
        """
        Builds a frame (origin, p_width, p_height) whose width matches
        *content*'s natural size, preserving direction vectors from the
        current frame.

        Returns (origin, p_width, p_height, descent, font_height).
        """
        p_origin = registry.get_point(self.origin_id)
        p_width = registry.get_point(self.width_id)
        p_height = registry.get_point(self.height_id)
        _, descent, font_height = self.get_font_metrics()

        if content == self.content or not self.content:
            logger.debug(
                f"_build_frame: no scaling, content == self.content "
                f"({content!r} == {self.content!r}) or empty"
            )
            return (
                (p_origin.x, p_origin.y),
                (p_width.x, p_width.y),
                (p_height.x, p_height.y),
                descent,
                font_height,
            )

        nat_w, _ = self.get_natural_size(content)

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
            h_scale = font_height / frame_h

        scaled_height = (
            p_origin.x + hx * h_scale,
            p_origin.y + hy * h_scale,
        )

        return (
            (p_origin.x, p_origin.y),
            scaled_width,
            scaled_height,
            descent,
            font_height,
        )

    def to_geometry(
        self,
        registry: "EntityRegistry",
        resolved_content: Optional[str] = None,
    ) -> Geometry:
        """Converts the text box to a Geometry object."""
        text = (
            resolved_content if resolved_content is not None
            else self.content
        )
        origin, pw, ph, descent, fh = self._build_frame_for_content(
            registry, text
        )
        txt_geo = Geometry.from_text(text, self.font_config)
        txt_geo.flip_y()

        return txt_geo.map_to_frame(
            origin, pw, ph,
            anchor_y=-descent,
            stable_src_height=fh,
        )

    def create_text_fill_geometry(
        self,
        registry: "EntityRegistry",
        resolved_content: Optional[str] = None,
    ) -> Optional[Geometry]:
        """Creates a fill geometry for text entities."""
        text = (
            resolved_content if resolved_content is not None
            else self.content
        )
        origin, pw, ph, descent, fh = self._build_frame_for_content(
            registry, text
        )
        txt_geo = Geometry.from_text(text, self.font_config)
        txt_geo.flip_y()

        return txt_geo.map_to_frame(
            origin, pw, ph,
            anchor_y=-descent,
            stable_src_height=fh,
        )

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "TextBoxEntity":
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

from typing import List, Tuple, Dict, Any, Sequence, Optional, TYPE_CHECKING
from ...geo import primitives
from ...geo.geometry import Geometry
from ...geo.font_config import FontConfig
from .entity import Entity
from .line import Line

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..registry import EntityRegistry


class TextBoxEntity(Entity):
    def __init__(
        self,
        id: int,
        origin_id: int,
        width_id: int,
        height_id: int,
        content: str = "",
        font_config: Optional[FontConfig] = None,
        construction: bool = False,
        construction_line_ids: Optional[List[int]] = None,
    ):
        super().__init__(id, construction)
        self.origin_id = origin_id
        self.width_id = width_id
        self.height_id = height_id
        self.content = content
        self.font_config = font_config or FontConfig()
        self.construction_line_ids = construction_line_ids or []
        self.type = "text_box"

    def get_point_ids(self) -> List[int]:
        return [self.origin_id, self.width_id, self.height_id]

    def get_all_frame_point_ids(self, registry: "EntityRegistry") -> List[int]:
        """Returns all 4 corner points of the text box frame."""
        ids = [self.origin_id, self.width_id, self.height_id]
        p4_id = self.get_fourth_corner_id(registry)
        if p4_id is not None:
            ids.append(p4_id)
        return ids

    def get_font_metrics(self) -> Tuple[float, float, float]:
        return self.font_config.get_font_metrics()

    def get_fourth_corner_id(
        self, registry: "EntityRegistry"
    ) -> Optional[int]:
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
        rect: Tuple[float, float, float, float],
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
        rect: Tuple[float, float, float, float],
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

    def to_geometry(self, registry: "EntityRegistry") -> Geometry:
        """Converts the text box to a Geometry object."""
        p_origin = registry.get_point(self.origin_id)
        p_width = registry.get_point(self.width_id)
        p_height = registry.get_point(self.height_id)
        txt_geo = Geometry.from_text(self.content, self.font_config)
        txt_geo.flip_y()

        _, descent, font_height = self.get_font_metrics()

        return txt_geo.map_to_frame(
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p_height.x, p_height.y),
            anchor_y=-descent,
            stable_src_height=font_height,
        )

    def create_text_fill_geometry(
        self, registry: "EntityRegistry"
    ) -> Optional[Geometry]:
        """Creates a fill geometry for text entities."""
        p_origin = registry.get_point(self.origin_id)
        p_width = registry.get_point(self.width_id)
        p_height = registry.get_point(self.height_id)

        txt_geo = Geometry.from_text(self.content, self.font_config)
        txt_geo.flip_y()

        _, descent, font_height = self.get_font_metrics()

        return txt_geo.map_to_frame(
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p_height.x, p_height.y),
            anchor_y=-descent,
            stable_src_height=font_height,
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
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextBoxEntity":
        return cls(
            id=data["id"],
            origin_id=data["origin_id"],
            width_id=data["width_id"],
            height_id=data["height_id"],
            content=data.get("content", ""),
            font_config=FontConfig.from_dict(data.get("font_config")),
            construction=data.get("construction", False),
            construction_line_ids=data.get("construction_line_ids"),
        )

    def __repr__(self) -> str:
        return (
            f"TextBoxEntity(id={self.id}, origin={self.origin_id}, "
            f"width={self.width_id}, height={self.height_id}, "
            f"content='{self.content}')"
        )

from typing import List, Tuple, Dict, Any, Sequence, Optional, TYPE_CHECKING
from ...geo import primitives
from ...geo.geometry import Geometry
from .entity import Entity

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
        font_params: Optional[Dict[str, Any]] = None,
        construction: bool = False,
        construction_line_ids: Optional[List[int]] = None,
    ):
        super().__init__(id, construction)
        self.origin_id = origin_id
        self.width_id = width_id
        self.height_id = height_id
        self.content = content
        self.font_params = font_params or {
            "family": "sans-serif",
            "size": 10.0,
            "bold": False,
            "italic": False,
        }
        self.construction_line_ids = construction_line_ids or []
        self.type = "text_box"

    def get_point_ids(self) -> List[int]:
        return [self.origin_id, self.width_id, self.height_id]

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
        txt_geo = Geometry.from_text(
            self.content,
            font_family=self.font_params.get("family", "sans-serif"),
            font_size=self.font_params.get("size", 10.0),
            is_bold=self.font_params.get("bold", False),
            is_italic=self.font_params.get("italic", False),
        )
        txt_geo.flip_y()
        return txt_geo.map_to_frame(
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p_height.x, p_height.y),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "origin_id": self.origin_id,
                "width_id": self.width_id,
                "height_id": self.height_id,
                "content": self.content,
                "font_params": self.font_params,
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
            font_params=data.get("font_params"),
            construction=data.get("construction", False),
            construction_line_ids=data.get("construction_line_ids"),
        )

    def __repr__(self) -> str:
        return (
            f"TextBoxEntity(id={self.id}, origin={self.origin_id}, "
            f"width={self.width_id}, height={self.height_id}, "
            f"content='{self.content}')"
        )

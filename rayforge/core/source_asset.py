from __future__ import annotations
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..image.base_renderer import Renderer


@dataclass
class SourceAsset:
    """
    An immutable data record for a raw imported file and its base render.
    This is stored once per file in the document's central asset registry.
    """

    source_file: Path
    original_data: bytes
    renderer: "Renderer"
    base_render_data: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the SourceAsset to a dictionary."""
        return {
            "uid": self.uid,
            "source_file": str(self.source_file),
            "original_data": self.original_data,
            "base_render_data": self.base_render_data,
            "renderer_name": self.renderer.__class__.__name__,
            "metadata": self.metadata,
            "width_px": self.width_px,
            "height_px": self.height_px,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "SourceAsset":
        """Deserializes a dictionary into a SourceAsset instance."""
        from ..image import renderer_by_name

        renderer = renderer_by_name[state["renderer_name"]]

        return cls(
            uid=state["uid"],
            source_file=Path(state["source_file"]),
            original_data=state["original_data"],
            base_render_data=state.get("base_render_data"),
            renderer=renderer,
            metadata=state.get("metadata", {}),
            width_px=state.get("width_px"),
            height_px=state.get("height_px"),
        )

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List

from .geo import Geometry
from .vectorization_spec import VectorizationSpec

# A type alias for a list of serializable modifier configurations.
ImageModifierChain = List[Dict[str, Any]]


@dataclass
class GenerationConfig:
    """
    A self-contained set of instructions for generating a WorkPiece's vectors.
    """

    source_asset_uid: str
    segment_mask_geometry: Geometry
    vectorization_spec: VectorizationSpec
    image_modifier_chain: ImageModifierChain = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the configuration to a dictionary."""
        return {
            "source_asset_uid": self.source_asset_uid,
            "segment_mask_geometry": self.segment_mask_geometry.to_dict(),
            "image_modifier_chain": self.image_modifier_chain,
            "vectorization_spec": self.vectorization_spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        """Deserializes a dictionary into a GenerationConfig instance."""
        return cls(
            source_asset_uid=data["source_asset_uid"],
            segment_mask_geometry=Geometry.from_dict(
                data["segment_mask_geometry"]
            ),
            image_modifier_chain=data.get("image_modifier_chain", []),
            vectorization_spec=VectorizationSpec.from_dict(
                data["vectorization_spec"]
            ),
        )

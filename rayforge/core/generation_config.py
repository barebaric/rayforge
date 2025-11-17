from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

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

    # --- Fields for cropped/traced bitmap rendering ---
    crop_window_px: Optional[Tuple[float, float, float, float]] = None
    source_image_width_px: Optional[int] = None
    source_image_height_px: Optional[int] = None
    cropped_width_mm: Optional[float] = None
    cropped_height_mm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the configuration to a dictionary."""
        return {
            "source_asset_uid": self.source_asset_uid,
            "segment_mask_geometry": self.segment_mask_geometry.to_dict(),
            "image_modifier_chain": self.image_modifier_chain,
            "vectorization_spec": self.vectorization_spec.to_dict(),
            "crop_window_px": self.crop_window_px,
            "source_image_width_px": self.source_image_width_px,
            "source_image_height_px": self.source_image_height_px,
            "cropped_width_mm": self.cropped_width_mm,
            "cropped_height_mm": self.cropped_height_mm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        """Deserializes a dictionary into a GenerationConfig instance."""
        # Handle tuple conversion for crop_window_px if it's a list from JSON
        crop_window = data.get("crop_window_px")
        if isinstance(crop_window, list):
            crop_window = tuple(crop_window)

        return cls(
            source_asset_uid=data["source_asset_uid"],
            segment_mask_geometry=Geometry.from_dict(
                data["segment_mask_geometry"]
            ),
            image_modifier_chain=data.get("image_modifier_chain", []),
            vectorization_spec=VectorizationSpec.from_dict(
                data["vectorization_spec"]
            ),
            crop_window_px=crop_window,
            source_image_width_px=data.get("source_image_width_px"),
            source_image_height_px=data.get("source_image_height_px"),
            cropped_width_mm=data.get("cropped_width_mm"),
            cropped_height_mm=data.get("cropped_height_mm"),
        )

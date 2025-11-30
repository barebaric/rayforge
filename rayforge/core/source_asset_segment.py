from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy

from .geo import Geometry
from .vectorization_spec import VectorizationSpec

# A type alias for a list of serializable modifier configurations.
ImageModifierChain = List[Dict[str, Any]]


@dataclass
class SourceAssetSegment:
    """
    Contains vectors describing the boundaries of a segment in a
    SourceAsset, along with a set of instructions for generating those
    boundary vectors.
    """

    source_asset_uid: str
    segment_mask_geometry: Geometry
    vectorization_spec: VectorizationSpec
    image_modifier_chain: ImageModifierChain = field(default_factory=list)

    # --- Fields for cropped/traced bitmap rendering ---
    crop_window_px: Optional[Tuple[float, float, float, float]] = None
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
            "cropped_width_mm": self.cropped_width_mm,
            "cropped_height_mm": self.cropped_height_mm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceAssetSegment":
        """Deserializes a dictionary into a SourceAssetSegment instance."""
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
            cropped_width_mm=data.get("cropped_width_mm"),
            cropped_height_mm=data.get("cropped_height_mm"),
        )

    def clone_with_geometry(
        self, new_geometry: Geometry
    ) -> "SourceAssetSegment":
        """
        Creates a deep copy of this segment but replaces the mask geometry.

        This is essential for splitting operations. It ensures the new segment
        is independent (deep copied mutable fields) and does not carry the
        memory overhead of the parent's potentially large geometry.
        """
        # Use dataclasses.replace for a shallow copy of scalar fields
        new_segment = replace(self, segment_mask_geometry=new_geometry)

        # Manually deepcopy mutable fields to ensure independence
        new_segment.image_modifier_chain = deepcopy(self.image_modifier_chain)
        new_segment.vectorization_spec = deepcopy(self.vectorization_spec)

        return new_segment

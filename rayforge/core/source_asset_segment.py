from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy

from .geo import Geometry
from .matrix import Matrix
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
    vectorization_spec: VectorizationSpec
    image_modifier_chain: ImageModifierChain = field(default_factory=list)
    layer_id: Optional[str] = None

    # --- Fields for cropped/traced bitmap rendering ---
    crop_window_px: Optional[Tuple[float, float, float, float]] = None
    cropped_width_mm: Optional[float] = None
    cropped_height_mm: Optional[float] = None

    # --- Fields for non-destructive vector import ---
    pristine_geometry: Optional[Geometry] = None
    normalization_matrix: Optional[Matrix] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the configuration to a dictionary."""
        return {
            "source_asset_uid": self.source_asset_uid,
            "image_modifier_chain": self.image_modifier_chain,
            "vectorization_spec": self.vectorization_spec.to_dict(),
            "crop_window_px": self.crop_window_px,
            "cropped_width_mm": self.cropped_width_mm,
            "cropped_height_mm": self.cropped_height_mm,
            "layer_id": self.layer_id,
            "pristine_geometry": self.pristine_geometry.to_dict()
            if self.pristine_geometry
            else None,
            "normalization_matrix": self.normalization_matrix.to_list()
            if self.normalization_matrix
            else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceAssetSegment":
        """Deserializes a dictionary into a SourceAssetSegment instance."""
        # Handle tuple conversion for crop_window_px if it's a list from JSON
        crop_window = data.get("crop_window_px")
        if isinstance(crop_window, list):
            crop_window = tuple(crop_window)

        pristine_geo_data = data.get("pristine_geometry")
        pristine_geometry = (
            Geometry.from_dict(pristine_geo_data)
            if pristine_geo_data
            else None
        )

        norm_matrix_data = data.get("normalization_matrix")
        normalization_matrix = (
            Matrix.from_list(norm_matrix_data) if norm_matrix_data else None
        )

        return cls(
            source_asset_uid=data["source_asset_uid"],
            image_modifier_chain=data.get("image_modifier_chain", []),
            vectorization_spec=VectorizationSpec.from_dict(
                data["vectorization_spec"]
            ),
            crop_window_px=crop_window,
            cropped_width_mm=data.get("cropped_width_mm"),
            cropped_height_mm=data.get("cropped_height_mm"),
            layer_id=data.get("layer_id"),
            pristine_geometry=pristine_geometry,
            normalization_matrix=normalization_matrix,
        )

    def clone_with_geometry(
        self, new_y_down_geometry: Geometry
    ) -> "SourceAssetSegment":
        """
        Creates a deep copy of this segment for use in splitting operations.

        The provided `new_y_down_geometry` is assumed to be normalized and
        becomes the new pristine shape. The normalization matrix is reset to
        identity. This ensures the new workpiece fragment renders correctly.
        """
        # Use dataclasses.replace for a shallow copy of scalar fields.
        # The new geometry becomes the pristine data, and since it's already
        # normalized, the normalization matrix is identity.
        new_segment = replace(
            self,
            pristine_geometry=new_y_down_geometry,
            normalization_matrix=Matrix.identity(),
        )

        # Manually deepcopy mutable fields to ensure independence
        new_segment.image_modifier_chain = deepcopy(self.image_modifier_chain)
        new_segment.vectorization_spec = deepcopy(self.vectorization_spec)

        return new_segment

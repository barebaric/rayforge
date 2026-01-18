import logging
from typing import List, Dict, Optional
from ..core.geo import Geometry
from ..core.item import DocItem
from ..core.layer import Layer
from ..core.source_asset import SourceAsset
from ..core.source_asset_segment import SourceAssetSegment
from ..core.workpiece import WorkPiece
from ..core.vectorization_spec import VectorizationSpec
from .structures import LayoutItem

logger = logging.getLogger(__name__)


class ItemAssembler:
    """
    Phase 3: Object Assembly.

    Factory that instantiates Rayforge domain objects (WorkPieces, Layers)
    based on the LayoutPlan calculated by the Engine.
    """

    def create_items(
        self,
        source_asset: SourceAsset,
        layout_plan: List[LayoutItem],
        spec: VectorizationSpec,
        source_name: str,
        geometries: Dict[Optional[str], Geometry],
        layer_manifest: Optional[List[Dict[str, str]]] = None,
    ) -> List[DocItem]:
        """
        Creates DocItems from the plan.
        """
        if not layout_plan:
            return []

        # If we have multiple items, we generally wrap them in Layers (if
        # requested by spec) or return a list of WorkPieces.
        items: List[DocItem] = []
        layer_names = (
            {m["id"]: m["name"] for m in layer_manifest}
            if layer_manifest
            else {}
        )

        for item in layout_plan:
            # 1. Create the Segment
            # This links the WorkPiece to the specific subset of the source
            # file
            geo = geometries.get(item.layer_id)

            segment = SourceAssetSegment(
                source_asset_uid=source_asset.uid,
                vectorization_spec=spec,
                layer_id=item.layer_id,
                pristine_geometry=geo,
                # Geometry Normalization
                normalization_matrix=item.normalization_matrix,
                # Raster Cropping
                crop_window_px=item.crop_window,
            )

            # Note: We should probably store physical dimensions on the segment
            # for split/crop reference, calculated from world matrix scale.
            w_mm, h_mm = item.world_matrix.get_abs_scale()
            segment.cropped_width_mm = w_mm
            segment.cropped_height_mm = h_mm

            # 2. Create the WorkPiece
            # Prioritize human-readable name from manifest, fallback to ID,
            # then to the overall source name.
            name = (
                layer_names.get(item.layer_id, item.layer_id)
                if item.layer_id
                else source_name
            )
            wp = WorkPiece(name=name, source_segment=segment)

            # 3. Apply Physical Transforms
            wp.matrix = item.world_matrix
            wp.natural_width_mm = w_mm
            wp.natural_height_mm = h_mm

            # 4. Wrap in Layer if splitting is active and meaningful
            if item.layer_id:
                layer = Layer(name=name)
                layer.add_child(wp)
                items.append(layer)
            else:
                items.append(wp)

        return items

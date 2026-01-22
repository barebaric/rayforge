import logging
from typing import List, Dict, Optional, Tuple
from ..core.geo import Geometry
from ..core.item import DocItem
from ..core.layer import Layer
from ..core.source_asset import SourceAsset
from ..core.source_asset_segment import SourceAssetSegment
from ..core.workpiece import WorkPiece
from ..core.vectorization_spec import VectorizationSpec, PassthroughSpec
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
        page_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[DocItem]:
        """
        Creates DocItems from the plan.
        """
        if not layout_plan:
            return []

        # If we have multiple items, we generally wrap them in Layers (if
        # requested by spec) or return a list of WorkPieces.
        items: List[DocItem] = []

        logger.debug(f"ItemAssembler: page_bounds={page_bounds}")

        for item in layout_plan:
            # 1. Create the Segment
            # This links the WorkPiece to the specific subset of the source
            # file
            geo: Optional[Geometry] = None
            if item.layer_id is not None:
                # Split strategy: get geometry for the specific layer
                geo = geometries.get(item.layer_id)
            else:
                # Merge strategy: combine all available geometries
                if geometries:
                    merged_geo = Geometry()
                    for g in geometries.values():
                        if g and not g.is_empty():
                            merged_geo.extend(g)
                    if not merged_geo.is_empty():
                        geo = merged_geo

            # The `item.crop_window` is in absolute native coordinates.
            # For rendering trimmed vector files (like SVG), the renderer
            # needs an absolute viewBox to render the correct portion of the
            # source file. We pass this through directly.
            # NOTE: For raster sources, this field contains pixel coordinates.
            # For vector sources (SVG), it contains native user-units. The
            # respective WorkPiece/Renderer logic must handle this distinction.
            logger.debug(
                f"ItemAssembler: item.crop_window={item.crop_window}, "
                f"layer_id={item.layer_id}"
            )

            segment = SourceAssetSegment(
                source_asset_uid=source_asset.uid,
                vectorization_spec=spec,
                layer_id=item.layer_id,
                pristine_geometry=geo,
                normalization_matrix=item.normalization_matrix,
                crop_window_px=item.crop_window,
            )

            # Note: We should probably store physical dimensions on the segment
            # for split/crop reference, calculated from the world matrix scale.
            w_mm, h_mm = item.world_matrix.get_abs_scale()
            segment.cropped_width_mm = w_mm
            segment.cropped_height_mm = h_mm

            # 2. Create the WorkPiece
            # Prioritize human-readable name from layout item, fallback to ID,
            # then to the overall source name.
            name = (
                item.layer_name
                if item.layer_name
                else (item.layer_id if item.layer_id else source_name)
            )
            wp = WorkPiece(name=name, source_segment=segment)

            # 3. Apply Physical Transforms
            wp.matrix = item.world_matrix
            wp.natural_width_mm = w_mm
            wp.natural_height_mm = h_mm

            # 4. Wrap in Layer if splitting is active and meaningful
            # Hack: Sketches (layer_id="__default__") should never be wrapped
            # in a layer
            if (
                item.layer_id
                and item.layer_id != "__default__"
                and isinstance(spec, PassthroughSpec)
                and spec.create_new_layers
            ):
                layer = Layer(name=name)
                layer.add_child(wp)
                items.append(layer)
            else:
                items.append(wp)

        return items

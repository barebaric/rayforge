import logging
from typing import Any

from raygeo.geo import Geometry
from raygeo.geo.types import Rect

from ..core.color import pick_unused_color
from ..core.item import DocItem
from ..core.source_asset import SourceAsset
from ..core.source_asset_segment import SourceAssetSegment
from ..core.step_registry import step_registry
from ..core.vectorization_spec import (
    LayerImportMode,
    PassthroughSpec,
    VectorizationSpec,
)
from ..core.workpiece import WorkPiece
from .structures import LayoutItem, SourceLayerInfo

logger = logging.getLogger(__name__)


class ItemAssembler:
    """
    Phase 5: Object Assembly.

    Factory that instantiates Rayforge domain objects (WorkPieces)
    based on the LayoutPlan calculated by the NormalizationEngine.

    Coordinate System Contract:
    --------------------------
    Input (LayoutItem):
    - crop_window: Native Coordinates (file-specific units)
    - normalization_matrix: Transforms Native -> Normalized (0-1, Y-Up)
    - world_matrix: Transforms Normalized (0-1, Y-Up) -> World (mm, Y-Up)

    Output (DocItems):
    - WorkPieces are positioned in World Coordinates (mm, Y-Up)
    - The transformation matrices are applied to the WorkPiece.matrix
    - Origin (0,0) is at the bottom-left of the workpiece

    Frame of Reference:
    ------------------
    - crop_window is absolute in the document's native coordinate space
    - The world_matrix positions the workpiece in the world coordinate system
    - All output DocItems are ready for insertion into the document

    Error Handling:
    ---------------
    This class does not collect errors. It assumes valid input from the
    NormalizationEngine. Invalid inputs may produce undefined results.
    """

    def create_items(
        self,
        source_asset: SourceAsset,
        layout_plan: list[LayoutItem],
        spec: VectorizationSpec,
        source_name: str,
        geometries: dict[str | None, Geometry],
        document_bounds: Rect | None = None,
    ) -> tuple[list[DocItem], list[SourceLayerInfo]]:
        """
        Creates WorkPieces from the layout plan.

        Instantiates WorkPieces based on the LayoutItem configurations
        calculated by the NormalizationEngine. For imports that split
        per source layer, the source layers' metadata is returned
        alongside so the caller can create a workflow per source layer.

        Coordinate System:
        ------------------
        Input:
        - layout_plan: LayoutItems with transformation matrices
        - geometries: Geometry in Native Coordinates
        - document_bounds: Optional bounds in Native Coordinates

        Output:
        - WorkPieces positioned in World Coordinates (mm, Y-Up) with
          transformation matrices applied

        Args:
            source_asset: The SourceAsset representing the imported file.
            layout_plan: List of LayoutItem configurations from
                         NormalizationEngine. Each contains transformation
                         matrices and crop windows.
            spec: VectorizationSpec describing the vectorization approach.
            source_name: Base name for the resulting DocItems.
            geometries: Dict of Geometry objects keyed by layer ID.
                        Geometry is in Native Coordinates.
            document_bounds: Optional document bounds in Native Coordinates.
                            Used for debugging and reference.

        Returns:
            A (items, source_layers) tuple. items is the list of
            WorkPieces ready for insertion into the document; may be
            empty if layout_plan is empty. source_layers carries the
            per-source-layer metadata in import order.

        Frame of Reference:
        ------------------
        - crop_window in LayoutItem is absolute in native coordinate space
        - For raster sources: crop_window contains pixel coordinates
        - For vector sources (SVG): crop_window contains native user-units
        - The WorkPiece/Renderer logic handles the distinction

        Error Handling:
        ---------------
        This method assumes valid input from the NormalizationEngine.
        Invalid inputs may produce undefined results.
        """
        if not layout_plan:
            return [], []

        items: list[DocItem] = []
        source_layers: list[SourceLayerInfo] = []
        seen_layer_ids: set[str] = set()
        used_colors = {item.color for item in layout_plan if item.color}

        logger.debug(f"ItemAssembler: document_bounds={document_bounds}")

        for item in layout_plan:
            # 1. Create the Segment
            # This links the WorkPiece to the specific subset of the source
            # file
            geo: Geometry | None = None
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
            items.append(wp)

            # 4. Track the source layer's metadata so the caller can
            # create a workflow per source layer. Sketches
            # (layer_id="__default__") are not source layers.
            if (
                item.layer_id
                and item.layer_id != "__default__"
                and item.layer_id not in seen_layer_ids
                and isinstance(spec, PassthroughSpec)
                and spec.layer_import_mode != LayerImportMode.FLATTEN
            ):
                seen_layer_ids.add(item.layer_id)
                color = item.color or pick_unused_color(used_colors)
                used_colors.add(color)
                source_layers.append(
                    SourceLayerInfo(
                        layer_id=item.layer_id,
                        name=name,
                        color=color,
                        settings=item.settings,
                    )
                )

        return items, source_layers

    @staticmethod
    def apply_settings_to_workflow(
        workflow, name: str, settings: dict[str, Any]
    ) -> None:
        """Create a configured step on the workflow from importer
        settings.

        Source layers tagged ``_is_image_layer`` (e.g. LightBurn
        ``CutSetting_Img``) get an ``EngraveStep``; everything else gets
        a ``ContourStep``.

        The settings dict uses the step's own attribute names
        (canonicalised by the importer); the step applies the domain
        settings it owns via :meth:`Step.apply_import_settings`. Only
        the structural ``passes`` value (a transformer config) is
        handled here.
        """
        is_image = bool(settings.get("_is_image_layer"))
        if is_image:
            step_name, typelabel = "EngraveStep", "Engrave"
        else:
            step_name, typelabel = "ContourStep", "Contour"
        cls = step_registry.get(step_name)
        if cls is None:
            return

        try:
            step = cls(typelabel=typelabel, name=name)
            per_wp, per_step = cls.get_default_transformers_dicts()
            step.per_workpiece_transformers_dicts = per_wp
            step.per_step_transformers_dicts = per_step
        except Exception:
            logger.exception("Failed to set up step from settings")
            return

        step.apply_import_settings(settings)

        passes = settings.get("passes")
        if passes is not None:
            for t in step.per_step_transformers_dicts:
                if t.get("name") == "MultiPassTransformer":
                    t["passes"] = passes
                    break

        workflow.add_step(step)

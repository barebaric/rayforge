import logging
from typing import Any, Dict, List, Optional

from raygeo.geo import Geometry
from raygeo.geo.types import Rect

from ..core.item import DocItem
from ..core.layer import Layer
from ..core.source_asset import SourceAsset
from ..core.source_asset_segment import SourceAssetSegment
from ..core.step_registry import step_registry
from ..core.vectorization_spec import (
    LayerImportMode,
    PassthroughSpec,
    VectorizationSpec,
)
from ..core.workpiece import WorkPiece
from .structures import LayoutItem

logger = logging.getLogger(__name__)


class ItemAssembler:
    """
    Phase 5: Object Assembly.

    Factory that instantiates Rayforge domain objects (WorkPieces, Layers)
    based on the LayoutPlan calculated by the NormalizationEngine.

    Coordinate System Contract:
    --------------------------
    Input (LayoutItem):
    - crop_window: Native Coordinates (file-specific units)
    - normalization_matrix: Transforms Native -> Normalized (0-1, Y-Up)
    - world_matrix: Transforms Normalized (0-1, Y-Up) -> World (mm, Y-Up)

    Output (DocItems):
    - WorkPieces and Layers are positioned in World Coordinates (mm, Y-Up)
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
        layout_plan: List[LayoutItem],
        spec: VectorizationSpec,
        source_name: str,
        geometries: Dict[Optional[str], Geometry],
        document_bounds: Optional[Rect] = None,
    ) -> List[DocItem]:
        """
        Creates DocItems from the layout plan.

        Instantiates WorkPieces and Layers based on the LayoutItem
        configurations calculated by the NormalizationEngine.

        Coordinate System:
        ------------------
        Input:
        - layout_plan: LayoutItems with transformation matrices
        - geometries: Geometry in Native Coordinates
        - document_bounds: Optional bounds in Native Coordinates

        Output:
        - DocItems (WorkPieces or Layers) positioned in World Coordinates
          (mm, Y-Up) with transformation matrices applied

        Args:
            source_asset: The SourceAsset representing the imported file.
            layout_plan: List of LayoutItem configurations from
                         NormalizationEngine. Each contains transformation
                         matrices and crop windows.
            spec: VectorizationSpec describing the vectorization approach.
                  Determines whether to create Layers for split items.
            source_name: Base name for the resulting DocItems.
            geometries: Dict of Geometry objects keyed by layer ID.
                        Geometry is in Native Coordinates.
            document_bounds: Optional document bounds in Native Coordinates.
                            Used for debugging and reference.

        Returns:
            List of DocItems (WorkPieces or Layers) ready for insertion
            into the document. May be empty if layout_plan is empty.

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
            return []

        # If we have multiple items, we generally wrap them in Layers (if
        # requested by spec) or return a list of WorkPieces.
        items: List[DocItem] = []

        logger.debug(f"ItemAssembler: document_bounds={document_bounds}")

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
                and spec.layer_import_mode != LayerImportMode.FLATTEN
            ):
                layer = Layer(name=name)
                layer.add_child(wp)
                # If the importer provided settings, pre-populate a step
                # on the layer's workflow so that add_default_steps_for_layers
                # will skip it (workflow.has_steps() -> True).
                if item.settings:
                    self._apply_settings(layer, item.settings)
                items.append(layer)
            else:
                items.append(wp)

        return items

    @staticmethod
    def _apply_settings(layer: Layer, settings: Dict[str, Any]) -> None:
        """Create a configured step on the layer from importer settings.

        Layers tagged ``_is_image_layer`` (e.g. LightBurn
        ``CutSetting_Img``) get an ``EngraveStep``; everything else gets
        a ``ContourStep``.
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
            step = cls(typelabel=typelabel, name=layer.name)
            if cls.PRODUCER_CLASS is not None:
                step.opsproducer_dict = cls.PRODUCER_CLASS().to_dict()
            per_wp, per_step = cls.get_default_transformers_dicts()
            step.per_workpiece_transformers_dicts = per_wp
            step.per_step_transformers_dicts = per_step
        except Exception:
            logger.exception("Failed to set up step from settings")
            return

        power = settings.get("power")
        if power is not None:
            try:
                step.set_power(power)
            except Exception:
                step.power = power

        cut_speed = settings.get("cut_speed")
        if cut_speed is not None:
            try:
                step.set_cut_speed(cut_speed)
            except Exception:
                step.cut_speed = cut_speed

        kerf_mm = settings.get("kerf_mm")
        if kerf_mm is not None:
            try:
                step.set_kerf_mm(kerf_mm)
            except Exception:
                step.kerf_mm = kerf_mm

        passes = settings.get("passes")
        if passes is not None:
            for t in step.per_step_transformers_dicts:
                if t.get("name") == "MultiPassTransformer":
                    t["passes"] = passes
                    break

        if is_image:
            # EngraveStep modulates between min_power/max_power; the base
            # Step.power attribute set above is unused by raster output,
            # so mirror the value (already maxPower/100) into max_power.
            if power is not None and hasattr(step, "max_power"):
                setattr(step, "max_power", power)
            min_power = settings.get("min_power")
            if min_power is not None and hasattr(step, "min_power"):
                setattr(step, "min_power", min_power)
            dot_width = settings.get("dot_width_correction_mm")
            if dot_width is not None and hasattr(
                step, "dot_width_correction_mm"
            ):
                setattr(step, "dot_width_correction_mm", dot_width)
            line_interval = settings.get("line_interval_mm")
            if line_interval is not None and hasattr(
                step, "line_interval_mm"
            ):
                setattr(step, "line_interval_mm", line_interval)
            scan_angle = settings.get("scan_angle")
            if scan_angle is not None and hasattr(step, "scan_angle"):
                setattr(step, "scan_angle", scan_angle)

        workflow = layer.workflow
        if workflow is not None:
            workflow.add_step(step)

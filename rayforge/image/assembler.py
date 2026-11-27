import logging
from gettext import gettext as _
from typing import Any

from raygeo.geo import Geometry, Matrix
from raygeo.geo.types import Rect

from ..core.color import pick_unused_color
from ..core.group import Group
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
        layout_plan: list[LayoutItem],
        spec: VectorizationSpec,
        source_name: str,
        geometries: dict[str | None, Geometry],
        document_bounds: Rect | None = None,
    ) -> list[DocItem]:
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
        items: list[DocItem] = []
        used_colors = {item.color for item in layout_plan if item.color}
        layers_by_id: dict[str, Layer] = {}
        groups_by_key: dict[tuple[str | None, str], Group] = {}

        logger.debug(f"ItemAssembler: document_bounds={document_bounds}")

        for item in layout_plan:
            # 1. Create the Segment
            # This links the WorkPiece to the specific subset of the source
            # file
            geo = item.geometry
            if geo is None:
                geo = self._lookup_geometry(item, geometries)

            # The `item.crop_window` is in absolute native coordinates.
            # For rendering trimmed vector files (like SVG), the renderer
            # needs an absolute viewBox to render the correct portion of the
            # source file. We pass this through directly.
            # NOTE: For raster sources, this field contains pixel coordinates.
            # For vector sources (SVG), it contains native user-units.
            # The respective WorkPiece/Renderer logic must handle this
            # distinction.
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
            # Prioritize the group name, then the human-readable layer name,
            # then the layer ID, then the overall source name.
            name = (
                item.group_name
                or item.layer_name
                or (item.layer_id if item.layer_id else source_name)
            )
            wp = WorkPiece(name=name, source_segment=segment)

            # 3. Apply Physical Transforms
            wp.matrix = item.world_matrix
            wp.natural_width_mm = w_mm
            wp.natural_height_mm = h_mm

            # 4. Wrap in Layer if splitting is active and meaningful
            # Hack: Sketches (layer_id="__default__") should never be wrapped
            # in a layer
            split = bool(
                item.layer_id
                and item.layer_id != "__default__"
                and isinstance(spec, PassthroughSpec)
                and spec.layer_import_mode != LayerImportMode.FLATTEN
            )
            owner: Layer | None = None
            if split:
                assert item.layer_id is not None
                owner = layers_by_id.get(item.layer_id)
                if owner is None:
                    layer_name = (
                        item.layer_name or item.layer_id or source_name
                    )
                    owner = Layer(name=layer_name)
                    owner.set_color(
                        item.color or pick_unused_color(used_colors)
                    )
                    used_colors.add(owner.color)
                    # If the importer provided settings, pre-populate a step
                    # on the layer's workflow so that
                    # add_default_steps_for_layers will skip it
                    # (workflow.has_steps() -> True).
                    if item.settings:
                        self._apply_settings(owner, item.settings)
                    layers_by_id[item.layer_id] = owner
                    items.append(owner)

            # 5. Attach the WorkPiece, grouping it when the importer asked
            # for that. Groups live on one layer, so pieces of a group that
            # spans layers become separate groups in their own layers.
            if item.group_id:
                key = (item.layer_id if split else None, item.group_id)
                group = groups_by_key.get(key)
                if group is None:
                    group = Group(name=item.group_name or _("Group"))
                    groups_by_key[key] = group
                    if owner is not None:
                        owner.add_child(group)
                    else:
                        items.append(group)
                group.add_child(wp)
            elif owner is not None:
                owner.add_child(wp)
            else:
                items.append(wp)

        # Groups are positioned on the document's unit square like every
        # other DocItem, so their children become relative to that square.
        for group in groups_by_key.values():
            self._fit_group_to_children(group)

        return items

    @staticmethod
    def _fit_group_to_children(group: Group) -> None:
        """Maps a group's unit square onto the bounds of its children.

        WorkPieces are created with world matrices so they can be placed
        on a layer directly. A group wraps some of them, so its own
        matrix has to take over their common position and size, and the
        children's matrices become relative to the group.
        """
        rects: list[tuple[float, float, float, float]] = []
        for child in group.children:
            ItemAssembler._collect_rects(child, Matrix.identity(), rects)
        if not rects:
            return

        min_x = min(rect[0] for rect in rects)
        min_y = min(rect[1] for rect in rects)
        max_x = max(rect[0] + rect[2] for rect in rects)
        max_y = max(rect[1] + rect[3] for rect in rects)
        # A block may be a single line, so its bounds can be degenerate.
        # Keep the mapping invertible rather than leaving the group with
        # an identity matrix and world-space children.
        width = max(max_x - min_x, 1e-9)
        height = max(max_y - min_y, 1e-9)

        group_matrix = Matrix.translation(min_x, min_y) @ Matrix.scale(
            width, height
        )
        try:
            inverse = group_matrix.invert()
        except ValueError:
            return

        children = [(child, child.matrix.copy()) for child in group.children]
        group.matrix = group_matrix
        for child, old_matrix in children:
            child.matrix = inverse @ old_matrix

    @staticmethod
    def _collect_rects(
        item: DocItem,
        parent_transform: Matrix,
        rects: list[tuple[float, float, float, float]],
    ) -> None:
        """Collects the bounding rects of an item's WorkPieces."""
        transform = parent_transform @ item.matrix
        if isinstance(item, Group):
            for child in item.children:
                ItemAssembler._collect_rects(child, transform, rects)
            return
        if not isinstance(item, WorkPiece):
            return
        local = item.get_local_bbox()
        if not local:
            return
        corners = [
            (local[0], local[1]),
            (local[0] + local[2], local[1]),
            (local[0] + local[2], local[1] + local[3]),
            (local[0], local[1] + local[3]),
        ]
        world = [transform.transform_point(point) for point in corners]
        min_x = min(p[0] for p in world)
        min_y = min(p[1] for p in world)
        max_x = max(p[0] for p in world)
        max_y = max(p[1] for p in world)
        rects.append((min_x, min_y, max_x - min_x, max_y - min_y))

    @staticmethod
    def _lookup_geometry(
        item: LayoutItem,
        geometries: dict[str | None, Geometry],
    ) -> Geometry | None:
        """The geometry a layout item refers to when it carries none."""
        if item.layer_id is not None:
            # Split strategy: get geometry for the specific layer
            return geometries.get(item.layer_id)
        # Merge strategy: combine all available geometries
        if not geometries:
            return None
        merged_geo = Geometry()
        for geo in geometries.values():
            if geo and not geo.is_empty():
                merged_geo.extend(geo)
        return merged_geo if not merged_geo.is_empty() else None

    @staticmethod
    def _apply_settings(layer: Layer, settings: dict[str, Any]) -> None:
        """Create a configured step on the layer from importer settings.

        Layers tagged ``_is_image_layer`` (e.g. LightBurn
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
            step = cls(typelabel=typelabel, name=layer.name)
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

        workflow = layer.workflow
        if workflow is not None:
            workflow.add_step(step)

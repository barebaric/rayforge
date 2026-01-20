import logging
from typing import List, Tuple, Optional
from ..core.matrix import Matrix
from ..core.vectorization_spec import (
    VectorizationSpec,
    PassthroughSpec,
    TraceSpec,
)
from .structures import (
    ParsingResult,
    LayoutItem,
    LayerGeometry,
    VectorizationResult,
)

logger = logging.getLogger(__name__)


class NormalizationEngine:
    """
    Phase 2: Layout Engine.

    Pure logic component that calculates how to map Native Coordinates
    (ParsingResult) to Rayforge World Coordinates (LayoutPlan) based on
    user intent (VectorizationSpec).
    """

    def calculate_layout(
        self,
        vec_result: VectorizationResult,
        spec: Optional[VectorizationSpec],
    ) -> List[LayoutItem]:
        """
        Calculates the layout plan.
        """
        result = vec_result.source_parse_result
        spec = spec or PassthroughSpec()

        # For traced results, the definitive bounds come from the new geometry.
        if isinstance(spec, TraceSpec):
            all_rects = []
            for geo in vec_result.geometries_by_layer.values():
                if geo and not geo.is_empty():
                    min_x, min_y, max_x, max_y = geo.rect()
                    all_rects.append(
                        (min_x, min_y, max_x - min_x, max_y - min_y)
                    )

            if not all_rects:
                # Fallback to page bounds if tracing produced no geometry
                return [
                    self._create_item_from_bounds(
                        result.page_bounds, result, layer_id=None
                    )
                ]

            union_rect = self._calculate_union_rect(all_rects)

            if union_rect[2] <= 0 or union_rect[3] <= 0:
                union_rect = result.page_bounds

            return [
                self._create_item_from_bounds(
                    union_rect, result, layer_id=None
                )
            ]

        # For direct vector imports (PassthroughSpec), bounds from parse phase
        # are authoritative.
        split_layers = False
        active_layers = None
        if isinstance(spec, PassthroughSpec):
            if spec.active_layer_ids:
                split_layers = True
                active_layers = set(spec.active_layer_ids)

        # Filter relevant layers
        target_layers: List[LayerGeometry] = result.layers
        if active_layers:
            target_layers = [
                geo for geo in result.layers if geo.layer_id in active_layers
            ]

        if not target_layers:
            # Fallback for empty files or no matching layers: use page bounds.
            return [
                self._create_item_from_bounds(
                    result.page_bounds, result, layer_id=None
                )
            ]

        if split_layers:
            # Strategy: Each layer gets its own workpiece, sized to its
            # content and positioned correctly in the world.
            plan = []
            for layer in target_layers:
                # Create the item based on its individual bounds to get the
                # correct world_matrix and normalization_matrix.
                plan.append(
                    self._create_item_from_bounds(
                        layer.content_bounds, result, layer_id=layer.layer_id
                    )
                )
            return plan
        else:
            # Strategy: Merged (Union Rect)
            # Calculate union of all content bounds
            union_rect = self._calculate_union_rect(
                [geo.content_bounds for geo in target_layers]
            )

            # If union is zero/invalid (e.g. empty layers), fallback to page
            if union_rect[2] <= 0 or union_rect[3] <= 0:
                union_rect = result.page_bounds

            return [
                self._create_item_from_bounds(
                    union_rect, result, layer_id=None
                )
            ]

    def _calculate_union_rect(
        self, rects: List[Tuple[float, float, float, float]]
    ) -> Tuple[float, float, float, float]:
        if not rects:
            return (0.0, 0.0, 0.0, 0.0)

        min_x = rects[0][0]
        min_y = rects[0][1]
        max_x = min_x + rects[0][2]
        max_y = min_y + rects[0][3]

        for x, y, w, h in rects[1:]:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _create_item_from_bounds(
        self,
        bounds: Tuple[float, float, float, float],
        result: ParsingResult,
        layer_id: Optional[str],
    ) -> LayoutItem:
        """
        Generates the matrices for a specific bounding box (x, y, w, h)
        in Native Coordinates.
        """
        bx, by, bw, bh = bounds

        # Protect against degenerate bounds
        if bw <= 0:
            bw = 1.0
        if bh <= 0:
            bh = 1.0

        # 1. Normalization Matrix: Native -> Unit Square (0-1)
        # Scale to 1, and only translate if the geometry is in the global
        # coordinate system.
        scale_matrix = Matrix.scale(1.0 / bw, 1.0 / bh)
        if result.geometry_is_relative_to_bounds:
            # Geometry is already at its local origin (0,0) due to trimming.
            # No translation needed for normalization.
            norm_matrix = scale_matrix
        else:
            # Geometry is in global coords. Translate it to its origin first.
            norm_matrix = scale_matrix @ Matrix.translation(-bx, -by)

        # If the source coordinate system is Y-Up, we
        # must flip the normalized output to match the Y-Down contract
        # expected by the WorkPiece.
        if not result.is_y_down:
            flip_matrix = Matrix.translation(0, 1) @ Matrix.scale(1, -1)
            norm_matrix = flip_matrix @ norm_matrix

        # 2. World Matrix: Unit Square (0-1) -> Physical World (mm)

        # Calculate physical dimensions of the content
        width_mm = bw * result.native_unit_to_mm
        height_mm = bh * result.native_unit_to_mm

        # X Position is a direct scaling of the absolute X coordinate.
        pos_x_mm = bx * result.native_unit_to_mm

        # Y Position depends on the coordinate system and the reference frame.
        # Rayforge World is Y-Up (0 at bottom).
        ref_bounds = result.untrimmed_page_bounds or result.page_bounds
        ref_h_native = ref_bounds[3]

        if result.is_y_down:
            # Native is Y-Down (0 at top). We invert relative to the full page.
            # Bottom of content in native coords = by + bh.
            # Distance from page bottom = ref_h_native - (by + bh).
            dist_from_bottom_native = ref_h_native - (by + bh)
            pos_y_mm = dist_from_bottom_native * result.native_unit_to_mm
        else:
            # Native is Y-Up (DXF). Origin is already at the bottom.
            pos_y_mm = by * result.native_unit_to_mm

        # World Matrix combines scale and the absolute calculated position.
        # Note: WorkPiece applies this matrix to a Y-Up 0-1 geometry.
        world_matrix = Matrix.translation(pos_x_mm, pos_y_mm) @ Matrix.scale(
            width_mm, height_mm
        )

        logger.debug(
            f"NormalizationEngine: layer_id={layer_id}, bounds={bounds}, "
            f"crop_window={bounds}, "
            f"native_unit_to_mm={result.native_unit_to_mm}"
        )
        return LayoutItem(
            layer_id=layer_id,
            world_matrix=world_matrix,
            normalization_matrix=norm_matrix,
            crop_window=bounds,
        )

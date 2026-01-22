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

    @staticmethod
    def calculate_layout_item(
        bounds: Tuple[float, float, float, float],
        parse_result: ParsingResult,
        layer_id: Optional[str] = None,
        layer_name: Optional[str] = None,
    ) -> LayoutItem:
        """
        Generates the matrices for a specific bounding box (x, y, w, h)
        in Native Coordinates. This is the central layout algorithm.
        """
        bx, by, bw, bh = bounds

        # Protect against degenerate bounds
        if bw <= 0:
            bw = 1.0
        if bh <= 0:
            bh = 1.0

        # 1. Normalization Matrix: Native -> Unit Square (0-1, Y-Up)
        scale_matrix = Matrix.scale(1.0 / bw, 1.0 / bh)
        if parse_result.geometry_is_relative_to_bounds:
            # Geometry is already at its local origin (0,0) due to trimming.
            # No translation needed for normalization.
            norm_matrix = scale_matrix
        else:
            # Geometry is in global coords. Translate it to its origin first.
            norm_matrix = scale_matrix @ Matrix.translation(-bx, -by)

        # The contract is that the normalization_matrix MUST produce a Y-UP,
        # 0-1 coordinate space for the WorkPiece.
        if parse_result.is_y_down:
            # Source (SVG, PNG) is Y-Down. We need to flip it to become Y-Up.
            flip_matrix = Matrix.translation(0, 1) @ Matrix.scale(1, -1)
            norm_matrix = flip_matrix @ norm_matrix

        # 2. World Matrix: Unit Square (0-1, Y-Up) -> Physical World (mm, Y-Up)
        width_mm = bw * parse_result.native_unit_to_mm
        height_mm = bh * parse_result.native_unit_to_mm

        pos_x_mm = bx * parse_result.native_unit_to_mm

        # The frame of reference for Y-inversion is the original,
        # untrimmed page.
        ref_bounds = (
            parse_result.untrimmed_document_bounds
            or parse_result.document_bounds
        )
        ref_x_native, ref_y_native, ref_w_native, ref_h_native = ref_bounds

        if parse_result.is_y_down:
            # Native is Y-Down (0 at top). We invert relative to the full page.
            # The bottom of the content in native coords is by + bh.
            # The bottom of the reference frame in native coords is
            # ref_y + ref_h
            dist_from_bottom_native = (ref_y_native + ref_h_native) - (by + bh)
            pos_y_mm = dist_from_bottom_native * parse_result.native_unit_to_mm
        else:
            # Native is Y-Up (DXF). Origin is already at the bottom.
            pos_y_mm = by * parse_result.native_unit_to_mm

        world_matrix = Matrix.translation(pos_x_mm, pos_y_mm) @ Matrix.scale(
            width_mm, height_mm
        )

        return LayoutItem(
            layer_id=layer_id,
            layer_name=layer_name,
            world_matrix=world_matrix,
            normalization_matrix=norm_matrix,
            crop_window=bounds,
        )

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

        if isinstance(spec, TraceSpec):
            # Check if we have any valid geometries
            has_valid_geo = any(
                geo and not geo.is_empty()
                for geo in vec_result.geometries_by_layer.values()
            )
            if not has_valid_geo:
                return []

            # For traced results, the coordinate system and overall bounds are
            # defined by the bitmap that was rendered for tracing. This is
            # described in the source_parse_result. The actual vector geometry
            # is just content within that frame. Using the document_bounds
            # ensures the final workpiece size matches the background image
            # size.
            bounds_to_use = result.document_bounds

            # Fallback in case the document bounds are invalid
            if bounds_to_use[2] <= 1e-6 or bounds_to_use[3] <= 1e-6:
                all_rects = []
                for geo in vec_result.geometries_by_layer.values():
                    if geo and not geo.is_empty():
                        min_x, min_y, max_x, max_y = geo.rect()
                        all_rects.append(
                            (min_x, min_y, max_x - min_x, max_y - min_y)
                        )
                if not all_rects:
                    # No geometry and no valid page bounds, return empty plan
                    return []
                bounds_to_use = self._calculate_union_rect(all_rects)

            return [
                self.calculate_layout_item(
                    bounds_to_use, result, layer_id=None, layer_name=None
                )
            ]

        # For direct vector imports (PassthroughSpec), bounds from parse phase
        # are authoritative.
        split_layers = False
        active_layers = None
        if isinstance(spec, PassthroughSpec):
            # The decision to split is now based on the spec flag.
            split_layers = spec.create_new_layers
            if spec.active_layer_ids:
                active_layers = set(spec.active_layer_ids)

        # Filter relevant layers
        target_layers: List[LayerGeometry] = result.layers
        if active_layers:
            target_layers = [
                geo for geo in result.layers if geo.layer_id in active_layers
            ]

        if not target_layers:
            # Fallback for empty files or no matching layers: use page bounds.
            # But if page bounds are effectively zero-sized, return empty plan.
            bx, by, bw, bh = result.document_bounds
            if bw <= 1e-6 or bh <= 1e-6:
                return []
            return [
                self.calculate_layout_item(
                    result.document_bounds,
                    result,
                    layer_id=None,
                    layer_name=None,
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
                    self.calculate_layout_item(
                        layer.content_bounds,
                        result,
                        layer_id=layer.layer_id,
                        layer_name=layer.name,
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
                union_rect = result.document_bounds

            return [
                self.calculate_layout_item(
                    union_rect, result, layer_id=None, layer_name=None
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

"""
Implements a pixel-based layout strategy for dense packing of workpieces.
"""

from __future__ import annotations
import math
import logging
from typing import List, Sequence, Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

import cairo
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve
from ...core.matrix import Matrix
from ...core.group import Group
from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from .base import LayoutStrategy

if TYPE_CHECKING:
    from ...shared.tasker.context import ExecutionContext


logger = logging.getLogger(__name__)


@dataclass
class WorkpieceVariant:
    """Represents a pre-rendered, rotated version of a DocItem."""

    item: DocItem  # The original DocItem (WorkPiece or Group)
    mask: np.ndarray  # Dilated mask for collision detection
    local_bbox: Tuple[float, float, float, float]  # Bbox in local coords
    angle_offset: int  # Rotation applied to create this variant
    unrotated_size_mm: Tuple[float, float]  # The size of the source shape


@dataclass
class PlacedItem:
    """Represents a workpiece variant placed on the packing canvas."""

    variant: WorkpieceVariant
    position_px: Tuple[int, int]  # (y, x) position on the canvas


class PixelPerfectLayoutStrategy(LayoutStrategy):
    """
    Arranges workpieces for maximum density using their rendered shapes.

    This strategy operates in three main phases:
    1.  **Preparation**: Each workpiece is rendered into a pixel mask for
        each allowed rotation. A margin is added by dilating the mask.
    2.  **Packing**: The masks are placed one-by-one onto a large virtual
        canvas using a greedy first-fit algorithm. The goal is to keep
        the total bounding box of all placed items as small as possible.
    3.  **Transformation**: The final pixel positions are translated back
        into world-coordinate transformation matrices for each workpiece.
    """

    def __init__(
        self,
        items: Sequence[DocItem],
        margin_mm: float = 0.5,
        resolution_px_per_mm: float = 8.0,
        allow_rotation: bool = True,
    ):
        """
        Initializes the pixel-perfect layout strategy.

        Args:
            items: The list of DocItems to arrange.
            margin_mm: The safety margin to add around each workpiece.
            resolution_px_per_mm: The resolution for rendering shapes.
                Higher values lead to more accurate but slower packing.
            allow_rotation: Whether to allow 90-degree rotations.
        """
        super().__init__(items)
        self.margin_mm = margin_mm
        self.resolution = resolution_px_per_mm
        self.allow_rotation = allow_rotation

    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[DocItem, Matrix]:
        """
        Calculates the transform for each workpiece for a dense layout. The
        final arrangement is centered relative to the center of the initial
        selection's bounding box.
        """
        if not self.items:
            return {}
        logger.info("Starting pixel-perfect layout...")

        # 1. Get initial selection bounding box and its center.
        selection_bbox = self._get_selection_world_bbox()
        if not selection_bbox:
            return {}
        min_x_world, min_y_world, max_x_world, max_y_world = selection_bbox
        initial_center = (
            (min_x_world + max_x_world) / 2,
            (min_y_world + max_y_world) / 2,
        )

        if context:
            context.set_message("Preparing workpiece variants...")

        prepared_items, total_area = self._prepare_variants()
        if not prepared_items:
            return {}

        if context:
            context.set_progress(0.1)
            context.set_message("Packing items...")

        # 2. Create packing canvas and pack items.
        canvas = self._create_packing_canvas(total_area, prepared_items)
        logger.info(
            f"Using packing canvas of {canvas.shape[1]}x{canvas.shape[0]} px."
        )

        placements, placed_bounds_px = self._pack_items(
            prepared_items, canvas, context
        )
        if not placements:
            return {}

        if context:
            context.set_progress(0.9)
            context.set_message("Calculating final positions...")

        # 3. Calculate the bounding box and center of the new packed layout.
        final_min_x_px = min(b[0] for b in placed_bounds_px)
        final_min_y_px = min(b[1] for b in placed_bounds_px)
        final_max_x_px = max(b[2] for b in placed_bounds_px)
        final_max_y_px = max(b[3] for b in placed_bounds_px)

        final_center_px = (
            (final_min_x_px + final_max_x_px) / 2,
            (final_min_y_px + final_max_y_px) / 2,
        )

        # 4. Calculate the world offset needed to align the final layout's
        #    center with the initial selection's center. This offset is the
        #    world coordinate that corresponds to pixel (0,0) on the canvas.
        group_offset = (
            initial_center[0] - (final_center_px[0] / self.resolution),
            initial_center[1] - (final_center_px[1] / self.resolution),
        )

        # 5. Compute the final transformation deltas using this new offset.
        deltas = self._compute_deltas_from_placements(placements, group_offset)

        logger.info("Pixel-perfect layout complete.")
        return deltas

    def _prepare_variants(
        self,
    ) -> Tuple[List[List[WorkpieceVariant]], int]:
        """
        Generates rotated and dilated masks for all DocItems.

        Returns:
            A tuple containing:
            - A list of item groups, where each group is a list of
              variants (rotations) for a single workpiece, sorted by size.
            - The total pixel area of all dilated masks.
        """
        groups = []
        total_area_px = 0
        rotations = [0, 90, 180, 270] if self.allow_rotation else [0]
        margin_px = int(self.margin_mm * self.resolution)

        for item in self.items:
            variants = []
            for angle in rotations:
                render = self._render_and_mask(item, angle)
                if not (render and np.sum(render[0]) > 0):
                    continue

                mask, local_bbox, unrotated_size = render

                if margin_px > 0:
                    # Pad the mask array to create physical space for the
                    # margin. The dilated mask will be larger than the
                    # original mask.
                    padded_mask = np.pad(
                        mask,
                        pad_width=margin_px,
                        mode="constant",
                        constant_values=False,
                    )
                    # Dilate the padded mask. Using iterations is an efficient
                    # way to expand the shape by `margin_px` pixels.
                    # The default 3x3 cross-shaped structure is used.
                    dilated_mask = binary_dilation(
                        padded_mask, iterations=margin_px
                    )
                else:
                    dilated_mask = mask

                variants.append(
                    WorkpieceVariant(
                        item, dilated_mask, local_bbox, angle, unrotated_size
                    )
                )
                total_area_px += np.sum(dilated_mask)

            if variants:
                groups.append(variants)

        # Sort workpieces by the max dimension of their first variant's mask
        # (heuristic for placing largest items first).
        groups.sort(key=lambda v_group: -max(v_group[0].mask.shape))
        return groups, total_area_px

    def _create_packing_canvas(
        self, total_area_px: int, items: List[List[WorkpieceVariant]]
    ) -> np.ndarray:
        """
        Creates a boolean numpy array to serve as the packing surface.

        Args:
            total_area_px: The sum of the pixel areas of all items.
            items: The prepared workpiece variants.

        Returns:
            A 2D boolean numpy array initialized to False.
        """
        # Estimate canvas side length with a 50% buffer for inefficiency.
        canvas_side = math.ceil(math.sqrt(total_area_px * 1.5))
        # Ensure canvas is at least as large as the largest item.
        max_dim = max(items[0][0].mask.shape) if items else 0
        canvas_h = canvas_w = max(canvas_side, max_dim) + 1
        return np.full((canvas_h, canvas_w), False, dtype=bool)

    def _pack_items(
        self,
        item_groups: List[List[WorkpieceVariant]],
        canvas: np.ndarray,
        context: Optional[ExecutionContext] = None,
    ) -> Tuple[List[PlacedItem], List[Tuple[int, int, int, int]]]:
        """
        Places workpiece variants onto the canvas greedily.

        Args:
            item_groups: A list of variant lists, one for each workpiece.
            canvas: The 2D numpy array to pack items onto.
            context: The execution context for reporting progress.

        Returns:
            A tuple containing:
            - A list of final `PlacedItem` instances.
            - A list of their bounding boxes in pixels (x0, y0, x1, y1).
        """
        placements: List[PlacedItem] = []
        placed_bounds_px: List[Tuple[int, int, int, int]] = []
        total_items = len(item_groups)

        for i, variants in enumerate(item_groups):
            wp_name = variants[0].item.name
            logger.debug(f"Placing item: {wp_name}")

            placement = self._find_best_placement(
                variants, canvas, placed_bounds_px
            )

            if placement:
                item, pos = placement.variant, placement.position_px
                y_px, x_px = pos
                h_px, w_px = item.mask.shape

                canvas[y_px : y_px + h_px, x_px : x_px + w_px] |= item.mask
                placed_bounds_px.append((x_px, y_px, x_px + w_px, y_px + h_px))
                placements.append(placement)

                if context:
                    # Calculate progress within the 0.1 to 0.9 range allocated
                    # for the packing phase (an 80% span).
                    pack_progress = (i + 1) / total_items
                    total_progress = 0.1 + (pack_progress * 0.8)
                    context.set_progress(total_progress)
                    context.set_message(
                        f"Packing item {i + 1} of {total_items}..."
                    )
            else:
                logger.warning(f"Could not place item {wp_name}.")

        return placements, placed_bounds_px

    def _find_best_placement(
        self,
        variants: List[WorkpieceVariant],
        canvas: np.ndarray,
        placed_bounds: List[Tuple[int, int, int, int]],
    ) -> Optional[PlacedItem]:
        """
        Finds the best rotation and position for an item.

        The "best" placement is the one that results in the smallest
        overall bounding box for all items placed so far.

        Args:
            variants: A list of possible rotations for a workpiece.
            canvas: The packing canvas.
            placed_bounds: A list of bounding boxes for already-placed items.

        Returns:
            The best `PlacedItem` if a fit is found, otherwise None.
        """
        best_fit: Optional[Dict] = None
        best_score = float("inf")

        for variant in variants:
            pos_px = self._find_first_fit(canvas, variant.mask)
            if not pos_px:
                continue

            # Score the placement by the area of the new total bounding box.
            score = self._calculate_placement_score(
                pos_px, variant.mask.shape, placed_bounds
            )

            if score < best_score:
                best_score = score
                best_fit = {"pos": pos_px, "variant": variant}

        if best_fit:
            logger.debug(
                f"  - Best fit: offset {best_fit['variant'].angle_offset}°, "
                f"pos {best_fit['pos']}, score {best_score:.0f}"
            )
            return PlacedItem(
                variant=best_fit["variant"], position_px=best_fit["pos"]
            )
        return None

    @staticmethod
    def _calculate_placement_score(
        pos_px: Tuple[int, int],
        mask_shape: Tuple[int, int],
        placed_bounds: List[Tuple[int, int, int, int]],
    ) -> float:
        """
        Calculates the area of the bounding box of a potential placement.

        Args:
            pos_px: The (y, x) position of the new item's top-left corner.
            mask_shape: The (h, w) shape of the new item's mask.
            placed_bounds: Bboxes of items already on the canvas, as
                           (x0, y0, x1, y1) tuples.

        Returns:
            The total area of the new combined bounding box.
        """
        y_px, x_px = pos_px
        h_px, w_px = mask_shape
        temp_bounds = placed_bounds + [(x_px, y_px, x_px + w_px, y_px + h_px)]
        min_x = min(b[0] for b in temp_bounds)
        min_y = min(b[1] for b in temp_bounds)
        max_x = max(b[2] for b in temp_bounds)
        max_y = max(b[3] for b in temp_bounds)
        return (max_x - min_x) * (max_y - min_y)

    def _compute_deltas_from_placements(
        self, placements: List[PlacedItem], group_offset: Tuple[float, float]
    ) -> Dict[DocItem, Matrix]:
        """
        Converts the list of pixel placements into transform deltas.

        Args:
            placements: The list of `PlacedItem`s.
            group_offset: The (x, y) world coordinate of the packing origin.

        Returns:
            A dictionary mapping each DocItem to its required delta matrix.
        """
        deltas: Dict[DocItem, Matrix] = {}
        for item in placements:
            doc_item, delta = self._create_delta_for_placement(
                item, group_offset
            )
            deltas[doc_item] = delta
        return deltas

    def _create_delta_for_placement(
        self, item: PlacedItem, group_offset: Tuple[float, float]
    ) -> Tuple[DocItem, Matrix]:
        """
        Calculates the final matrix and delta for a single placed item.

        Args:
            item: The `PlacedItem` to process.
            group_offset: The (x, y) world coordinate of the packing origin.

        Returns:
            A tuple of (DocItem, delta_Matrix).
        """
        doc_item = item.variant.item
        y_px, x_px = item.position_px
        margin_px = int(self.margin_mm * self.resolution)
        group_offset_x, group_offset_y = group_offset

        # 1. Calculate final position of the rotated bbox corner in world space
        true_x_px = x_px + margin_px
        true_y_px = y_px + margin_px
        packed_x = group_offset_x + (true_x_px / self.resolution)
        packed_y = group_offset_y + (true_y_px / self.resolution)

        # 2. Determine final position of the item's origin from bbox data
        bbox_off_x, bbox_off_y = (
            item.variant.local_bbox[0],
            item.variant.local_bbox[1],
        )
        final_x = packed_x - bbox_off_x
        final_y = packed_y - bbox_off_y
        T = Matrix.translation(final_x, final_y)

        # 3. Determine final rotation
        target_angle = item.variant.angle_offset

        # 4. Determine the final scale matrix (S) and rotation center.
        # This is the critical step where Groups are treated differently.
        if isinstance(doc_item, Group):
            # For a Group, we MUST preserve its existing world scale to avoid
            # deforming its children. The packer's job is only to move and
            # rotate the group as a rigid object.
            current_w, current_h = (
                doc_item.get_world_transform().get_abs_scale()
            )
            S = Matrix.scale(current_w, current_h)
            center_for_rot = (current_w / 2, current_h / 2)
        else:
            # For a WorkPiece, we create a new scale based on its original
            # world size, as it's being laid out from scratch.
            w_mm, h_mm = item.variant.unrotated_size_mm
            S = Matrix.scale(w_mm, h_mm)
            center_for_rot = (w_mm / 2, h_mm / 2)

        R = Matrix.rotation(target_angle, center=center_for_rot)

        # 5. Construct the final world matrix from T, R, and S.
        final_matrix = T @ R @ S

        # 6. Calculate the delta required to achieve this new world matrix.
        # W_new = P @ (Delta @ L_old) => Delta = P_inv @ W_new @ L_old_inv
        old_local_matrix = doc_item.matrix
        if old_local_matrix.has_zero_scale():
            logger.warning(f"Item {doc_item.name} has zero scale, skipping.")
            return doc_item, Matrix.identity()
        old_local_inv = old_local_matrix.invert()

        parent_inv = Matrix.identity()
        if doc_item.parent:
            parent_world_transform = doc_item.parent.get_world_transform()
            if not parent_world_transform.has_zero_scale():
                parent_inv = parent_world_transform.invert()

        delta = parent_inv @ final_matrix @ old_local_inv
        return doc_item, delta

    def _render_and_mask(
        self, item: DocItem, angle_offset: int
    ) -> Optional[
        Tuple[
            np.ndarray, Tuple[float, float, float, float], Tuple[float, float]
        ]
    ]:
        """
        Renders a DocItem to a pixel mask at a specific orientation.

        Returns a tuple: (mask, local_bbox_of_rotated_shape,
            unrotated_shape_size).
        """
        source_surface: Optional[cairo.ImageSurface] = None
        unrotated_w_mm, unrotated_h_mm = 0.0, 0.0

        if isinstance(item, WorkPiece):
            unrotated_w_mm, unrotated_h_mm = (
                item.get_world_transform().get_abs_scale()
            )
            if unrotated_w_mm <= 0 or unrotated_h_mm <= 0:
                return None
            source_surface = item.importer.render_to_pixels(
                width=int(unrotated_w_mm * self.resolution),
                height=int(unrotated_h_mm * self.resolution),
            )
        elif isinstance(item, Group):
            # For a group, render its contents based on its world AABB.
            bbox = self._get_item_world_bbox(item)
            if not bbox:
                return None
            min_x_world, min_y_world, max_x_world, max_y_world = bbox
            unrotated_w_mm = max_x_world - min_x_world
            unrotated_h_mm = max_y_world - min_y_world

            if unrotated_w_mm <= 0 or unrotated_h_mm <= 0:
                return None

            width_px = int(unrotated_w_mm * self.resolution)
            height_px = int(unrotated_h_mm * self.resolution)
            source_surface = cairo.ImageSurface(
                cairo.FORMAT_A8, width_px, height_px
            )
            ctx = cairo.Context(source_surface)

            for wp in item.get_descendants(of_type=WorkPiece):
                ctx.save()
                wp_w, wp_h = wp.get_world_transform().get_abs_scale()
                if wp_w <= 0 or wp_h <= 0:
                    ctx.restore()
                    continue

                wp_surf = wp.importer.render_to_pixels(
                    width=int(wp_w * self.resolution),
                    height=int(wp_h * self.resolution),
                )
                if not wp_surf:
                    ctx.restore()
                    continue

                # Get the workpiece's world transform relative to the
                # group's AABB.
                world_transform = wp.get_world_transform()
                tx, ty, angle, _, _, _ = world_transform.decompose()
                x_pos_px = (tx - min_x_world) * self.resolution

                # Correct for Cairo's Y-down coordinate system. We must invert
                # the Y position relative to the canvas height.
                y_pos_px = height_px - ((ty - min_y_world) * self.resolution)

                center_x_px, center_y_px = (
                    wp_surf.get_width() / 2,
                    wp_surf.get_height() / 2,
                )

                # Cairo's rotation center is also affected by the Y-inversion.
                # The translation must place the workpiece's bottom-left
                # corner, then adjust for rotation around its center.
                ctx.translate(x_pos_px, y_pos_px)
                ctx.translate(center_x_px, -center_y_px)
                ctx.rotate(math.radians(angle))
                ctx.translate(-center_x_px, -(-center_y_px))
                ctx.translate(
                    0, -wp_surf.get_height()
                )  # Move to top-left corner for painting

                ctx.set_source_surface(wp_surf, 0, 0)
                ctx.paint()
                ctx.restore()

        if not source_surface:
            return None

        # The rest of the logic rotates this source surface
        transform = Matrix.rotation(
            angle_offset, center=(unrotated_w_mm / 2, unrotated_h_mm / 2)
        )
        corners = [
            (0, 0),
            (unrotated_w_mm, 0),
            (unrotated_w_mm, unrotated_h_mm),
            (0, unrotated_h_mm),
        ]
        world_corners = [transform.transform_point(p) for p in corners]
        min_x, min_y = (
            min(p[0] for p in world_corners),
            min(p[1] for p in world_corners),
        )
        max_x, max_y = (
            max(p[0] for p in world_corners),
            max(p[1] for p in world_corners),
        )
        local_bbox = (min_x, min_y, max_x, max_y)

        width_mm, height_mm = max_x - min_x, max_y - min_y
        if width_mm <= 0 or height_mm <= 0:
            return None
        width_px, height_px = (
            round(width_mm * self.resolution),
            round(height_mm * self.resolution),
        )
        if not source_surface:
            return None

        # 3. Create a destination surface and draw the rotated source onto it.
        final_surface = cairo.ImageSurface(
            cairo.FORMAT_A8, width_px, height_px
        )
        ctx = cairo.Context(final_surface)
        src_w, src_h = source_surface.get_width(), source_surface.get_height()

        # Center the rotated image via translate-rotate-translate.
        ctx.translate(width_px / 2, height_px / 2)
        ctx.rotate(math.radians(angle_offset))
        ctx.translate(-src_w / 2, -src_h / 2)
        ctx.set_source_surface(source_surface, 0, 0)
        ctx.paint()

        # 4. Extract mask data from cairo surface into a numpy array.
        buf = final_surface.get_data()
        mask = np.frombuffer(buf, dtype=np.uint8).reshape(
            (height_px, final_surface.get_stride())
        )
        # We only care about the actual width, not the stride.
        mask = mask[:, :width_px] > 0
        return np.flipud(mask), local_bbox, (unrotated_w_mm, unrotated_h_mm)

    @staticmethod
    def _find_first_fit(
        canvas: np.ndarray, item_mask: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        Finds the first top-left position where an item fits on the canvas.

        This method uses FFT-based convolution to quickly find all
        collision-free locations, then returns the first one (top-most, then
        left-most). This is a significant optimization over a naive
        pixel-by-pixel scan, especially on large canvases.

        Args:
            canvas: The boolean 2D array representing occupied space.
            item_mask: The boolean 2D array of the item to place.

        Returns:
            A tuple (y, x) of the top-left corner for placement, or None
            if no fit is found.
        """
        canvas_h, canvas_w = canvas.shape
        item_h, item_w = item_mask.shape

        if item_h > canvas_h or item_w > canvas_w:
            return None

        # The core of the check is a 2D cross-correlation:
        # result(y, x) = sum(canvas[y:y+h, x:x+w] * item_mask)
        # We look for a (y,x) where the result is 0.
        # fftconvolve computes convolution, which is correlation with a
        # flipped kernel.
        # We use floating point numbers for fftconvolve performance.
        canvas_f = canvas.astype(np.float32)
        # The kernel must be flipped for cross-correlation.
        item_mask_f = np.flip(item_mask.astype(np.float32))

        # `mode='valid'` ensures the output size is correct for checking
        # every possible top-left placement. The result is a map where each
        # pixel value is the sum of products of overlapping areas.
        collision_map = fftconvolve(canvas_f, item_mask_f, mode="valid")

        # Due to floating point inaccuracies, results may not be exactly zero.
        # We round to the nearest integer to check for collisions. A collision
        # exists if the sum of overlapping pixels is > 0.
        collision_map_int = np.round(collision_map).astype(np.int32)

        # Find the coordinates of the first zero (no collision).
        # np.argwhere finds all non-zero elements. We want the first zero.
        potential_fits = np.argwhere(collision_map_int == 0)

        if potential_fits.size > 0:
            # np.argwhere returns results sorted first by row, then by column,
            # so the first result is the top-most, left-most fit.
            y, x = potential_fits[0]
            return int(y), int(x)

        return None

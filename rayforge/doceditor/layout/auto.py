"""
Implements a pixel-based layout strategy for dense packing of workpieces.
"""

from __future__ import annotations
import math
import logging
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

import cairo
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve
from ...core.matrix import Matrix
from ...core.workpiece import WorkPiece
from .base import LayoutStrategy

if TYPE_CHECKING:
    from ...shared.tasker.context import ExecutionContext


logger = logging.getLogger(__name__)


@dataclass
class WorkpieceVariant:
    """Represents a pre-rendered, rotated version of a workpiece."""

    workpiece: WorkPiece
    mask: np.ndarray  # Dilated mask for collision detection
    local_bbox: Tuple[float, float, float, float]  # Bbox in local coords
    angle_offset: int  # Rotation applied to create this variant


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
        workpieces: List[WorkPiece],
        margin_mm: float = .5,
        resolution_px_per_mm: float = 8.0,
        allow_rotation: bool = True,
    ):
        """
        Initializes the pixel-perfect layout strategy.

        Args:
            workpieces: The list of workpieces to arrange.
            margin_mm: The safety margin to add around each workpiece.
            resolution_px_per_mm: The resolution for rendering shapes.
                Higher values lead to more accurate but slower packing.
            allow_rotation: Whether to allow 90-degree rotations.
        """
        super().__init__(workpieces)
        self.margin_mm = margin_mm
        self.resolution = resolution_px_per_mm
        self.allow_rotation = allow_rotation

    def calculate_deltas(
        self, context: Optional[ExecutionContext] = None
    ) -> Dict[WorkPiece, Matrix]:
        """
        Calculates the transform for each workpiece for a dense layout. The
        final arrangement is centered relative to the center of the initial
        selection's bounding box.
        """
        if not self.workpieces:
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
        Generates rotated and dilated masks for all workpieces.

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

        for wp in self.workpieces:
            variants = []
            for angle in rotations:
                render = self._render_and_mask(wp, angle)
                if not (render and np.sum(render[0]) > 0):
                    continue

                mask, local_bbox = render

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
                    WorkpieceVariant(wp, dilated_mask, local_bbox, angle)
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
            wp_name = variants[0].workpiece.name
            logger.debug(f"Placing item: {wp_name}")

            placement = self._find_best_placement(
                variants, canvas, placed_bounds_px
            )

            if placement:
                item, pos = placement.variant, placement.position_px
                y_px, x_px = pos
                h_px, w_px = item.mask.shape

                canvas[y_px:y_px + h_px, x_px:x_px + w_px] |= item.mask
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
    ) -> Dict[WorkPiece, Matrix]:
        """
        Converts the list of pixel placements into transform deltas.

        Args:
            placements: The list of `PlacedItem`s.
            group_offset: The (x, y) world coordinate of the packing origin.

        Returns:
            A dictionary mapping each workpiece to its required delta matrix.
        """
        deltas = {}
        for item in placements:
            wp, delta = self._create_delta_for_placement(item, group_offset)
            deltas[wp] = delta
        return deltas

    def _create_delta_for_placement(
        self, item: PlacedItem, group_offset: Tuple[float, float]
    ) -> Tuple[WorkPiece, Matrix]:
        """
        Calculates the final matrix and delta for a single placed item.

        Args:
            item: The `PlacedItem` to process.
            group_offset: The (x, y) world coordinate of the packing origin.

        Returns:
            A tuple of (WorkPiece, delta_Matrix).
        """
        wp = item.variant.workpiece
        y_px, x_px = item.position_px
        margin_px = int(self.margin_mm * self.resolution)
        group_offset_x, group_offset_y = group_offset

        # The position (x_px, y_px) is for the top-left of the DILATED
        # mask. Account for the margin to find the true object position.
        true_x_px = x_px + margin_px
        true_y_px = y_px + margin_px

        # Convert the true bbox corner from canvas pixels to world coords.
        packed_x = group_offset_x + (true_x_px / self.resolution)
        packed_y = group_offset_y + (true_y_px / self.resolution)

        # The final position of the workpiece's origin is its packed
        # bbox position minus the bbox's offset from the origin.
        bbox_off_x, bbox_off_y = (
            item.variant.local_bbox[0],
            item.variant.local_bbox[1],
        )
        final_x = packed_x - bbox_off_x
        final_y = packed_y - bbox_off_y

        # Construct the final transformation matrix.
        # The target angle is the one determined by the packer. The original
        # angle of the workpiece is irrelevant and should be overwritten,
        # not added to.
        target_angle = item.variant.angle_offset
        w_mm, h_mm = wp.size
        S = Matrix.scale(w_mm, h_mm)
        R = Matrix.rotation(target_angle, center=(w_mm / 2, h_mm / 2))
        T = Matrix.translation(final_x, final_y)
        final_matrix = T @ R @ S

        # Calculate the delta needed to move from original to final matrix.
        delta = final_matrix @ wp.matrix.invert()
        return wp, delta

    def _render_and_mask(
        self, wp: WorkPiece, angle_offset: int
    ) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        """
        Renders a workpiece to a pixel mask at a specific orientation.

        Args:
            wp: The WorkPiece to render.
            angle_offset: The rotation to apply (0, 90, 180, or 270).

        Returns:
            A tuple containing the boolean numpy mask and the workpiece's
            bounding box in its local, rotated coordinate system, or None
            if rendering fails.
        """
        # 1. Calculate the transformation and the resulting bounding box.
        # The packing algorithm must work with the base shape of the
        # workpiece, testing canonical rotations. The workpiece's initial
        # angle is ignored here and handled at the end.
        w_mm, h_mm = wp.size
        transform = Matrix.rotation(
            angle_offset, center=(w_mm / 2, h_mm / 2)
        ) @ Matrix.scale(w_mm, h_mm)

        corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        world_corners = [transform.transform_point(p) for p in corners]
        min_x = min(p[0] for p in world_corners)
        min_y = min(p[1] for p in world_corners)
        max_x = max(p[0] for p in world_corners)
        max_y = max(p[1] for p in world_corners)

        local_bbox = (min_x, min_y, max_x, max_y)
        width_mm, height_mm = max_x - min_x, max_y - min_y
        if width_mm <= 0 or height_mm <= 0:
            return None

        width_px = round(width_mm * self.resolution)
        height_px = round(height_mm * self.resolution)

        # 2. Render the unrotated source image from the importer.
        source_surface = wp.importer.render_to_pixels(
            width=int(w_mm * self.resolution),
            height=int(h_mm * self.resolution),
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

        # Cairo's Y-axis points down. Flip to align with numpy indexing.
        return np.flipud(mask), local_bbox

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

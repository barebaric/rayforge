"""
Implements a pixel-based layout strategy for dense packing of workpieces.
"""

from __future__ import annotations
import math
import logging
from typing import (
    List,
    Sequence,
    Dict,
    Optional,
    Tuple,
    TYPE_CHECKING,
)
from dataclasses import dataclass
import cairo
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve
from ...context import get_context
from ...core.geo.linearize import linearize_arc
from ...core.group import Group
from ...core.matrix import Matrix
from ...core.item import DocItem
from ...core.stock import StockItem
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
    unrotated_size_mm: Tuple[float, float]  # The size of source shape


@dataclass
class PlacedItem:
    """Represents a workpiece variant placed on the packing canvas."""

    variant: WorkpieceVariant
    position_px: Tuple[int, int]  # (y, x) position on canvas


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
        Initializes pixel-perfect layout strategy.

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

        if context:
            context.set_message("Preparing workpiece variants...")

        prepared_items, total_area = self._prepare_variants()
        if not prepared_items:
            self.unplaced_items = list(self.items)
            return {}

        if context:
            context.set_progress(0.1)

        # Stock-aware Logic
        stock_item: Optional[StockItem] = None
        doc = self.items[0].doc
        # Get the stock item from the active layer
        if doc and doc.active_layer:
            stock_item = doc.active_layer.stock_item

        # Use stock as boundary if it exists, otherwise use whole surface.
        if stock_item:
            logger.info("Stock item found, using it as layout boundary.")
            if context:
                context.set_message("Using stock as boundary...")

            stock_bbox = stock_item.bbox
            canvas_origin_world = (stock_bbox[0], stock_bbox[1])
            canvas_w_mm, canvas_h_mm = stock_bbox[2], stock_bbox[3]
            canvas_w_px = round(canvas_w_mm * self.resolution)
            canvas_h_px = round(canvas_h_mm * self.resolution)

            # Create a mask of the valid area from the stock's geometry.
            allowed_area_mask = self._render_stock_to_mask(
                stock_item, canvas_w_px, canvas_h_px
            )
            # Initialize the canvas with invalid areas already marked.
            canvas = np.logical_not(allowed_area_mask)
            group_offset = canvas_origin_world

            placements, self.unplaced_items = self._pack_items(
                prepared_items, canvas, context
            )

        else:
            # Use whole surface (machine dimensions) as boundary
            logger.info("Using whole surface as layout boundary.")
            if context:
                context.set_message("Using whole surface as boundary...")

            # Get machine dimensions
            machine_w, machine_h = 200.0, 200.0  # Fallback
            machine = get_context().machine
            if machine:
                machine_w, machine_h = machine.dimensions

            canvas_origin_world = (0.0, 0.0)
            canvas_w_mm, canvas_h_mm = machine_w, machine_h
            canvas_w_px = round(canvas_w_mm * self.resolution)
            canvas_h_px = round(canvas_h_mm * self.resolution)

            # Create a full mask for the entire machine surface
            allowed_area_mask = np.ones((canvas_h_px, canvas_w_px), dtype=bool)
            # Initialize the canvas with all areas marked as valid
            canvas = np.zeros((canvas_h_px, canvas_w_px), dtype=bool)
            group_offset = canvas_origin_world

            placements, self.unplaced_items = self._pack_items(
                prepared_items, canvas, context
            )

        if self.unplaced_items:
            item_names = ", ".join(item.name for item in self.unplaced_items)
            message = _(
                "Could not fit the following items: {item_names}"
            ).format(item_names=item_names)
            self.error_reported.send(self, message=message)

        if context:
            context.set_progress(0.9)
            context.set_message("Calculating final positions...")

        # 5. Compute the final transformation deltas for successfully
        # placed items.
        deltas = self._compute_deltas_from_placements(placements, group_offset)

        # 6. If any items were unplaced and stock exists, move them
        #    outside the stock area.
        if self.unplaced_items and stock_item and stock_item.bbox:
            logger.info(
                f"Moving {len(self.unplaced_items)} unplaced items "
                "outside stock area."
            )
            # Calculate a collective bounding box for all unplaced items
            unplaced_bboxes = [
                self._get_item_world_bbox(item) for item in self.unplaced_items
            ]
            valid_bboxes = [b for b in unplaced_bboxes if b]
            if valid_bboxes:
                min_x = min(b[0] for b in valid_bboxes)
                min_y = min(b[1] for b in valid_bboxes)
                max_x = max(b[2] for b in valid_bboxes)
                max_y = max(b[3] for b in valid_bboxes)
                unplaced_coll_bbox = (min_x, min_y, max_x, max_y)

                # Determine the target position for the collective bbox's
                # top-left corner.
                stock_bbox = stock_item.bbox
                target_x = stock_bbox[0] + stock_bbox[2] + self.margin_mm * 4
                target_y = stock_bbox[1] + stock_bbox[3]

                # Calculate a single (dx, dy) offset for the whole group
                dx = target_x - unplaced_coll_bbox[0]
                dy = target_y - unplaced_coll_bbox[3]

                for item in self.unplaced_items:
                    # Reset rotation and apply to collective translation
                    old_world_transform = item.get_world_transform()
                    tx_old, ty_old = old_world_transform.decompose()[:2]

                    final_x = tx_old + dx
                    final_y = ty_old + dy

                    T = Matrix.translation(final_x, final_y)
                    scale_w, scale_h = old_world_transform.get_abs_scale()
                    S = Matrix.scale(scale_w, scale_h)
                    final_matrix = T @ S  # Rotation is reset to identity

                    # Calculate the local delta to achieve this
                    old_local_matrix = item.matrix
                    if old_local_matrix.has_zero_scale():
                        continue
                    old_local_inv = old_local_matrix.invert()
                    parent_inv = Matrix.identity()
                    if item.parent:
                        parent_tfm = item.parent.get_world_transform()
                        if not parent_tfm.has_zero_scale():
                            parent_inv = parent_tfm.invert()

                    delta = parent_inv @ final_matrix @ old_local_inv
                    deltas[item] = delta

        logger.info("Pixel-perfect layout complete.")
        return deltas

    def _render_stock_to_mask(
        self, stock_item: StockItem, width_px: int, height_px: int
    ) -> np.ndarray:
        """
        Renders the stock's transformed geometry to a boolean mask that
        defines the valid area for packing.

        Args:
            stock_item: The stock item to render.
            width_px: The width of the target canvas in pixels.
            height_px: The height of the target canvas in pixels.

        Returns:
            A 2D boolean numpy array where True represents a valid area.
        """
        # 1. Get the transform that maps the stock's local geometry space to
        #    world, then to the canvas's local pixel space.
        stock_world_transform = stock_item.get_world_transform()
        canvas_origin_world = (stock_item.bbox[0], stock_item.bbox[1])
        translation_to_canvas = Matrix.translation(
            -canvas_origin_world[0], -canvas_origin_world[1]
        )
        final_transform_mm = translation_to_canvas @ stock_world_transform

        # 2. Apply this transform to a copy of the geometry.
        # The Geometry.transform method expects a 4x4 NumPy array.
        geometry_for_render = stock_item.geometry.copy()
        m4x4 = np.identity(4)
        m4x4[:2, :2] = final_transform_mm.m[:2, :2]
        m4x4[:2, 3] = final_transform_mm.m[:2, 2]
        geometry_for_render.transform(m4x4)

        # 3. Render the transformed geometry onto a cairo surface.
        surface = cairo.ImageSurface(cairo.FORMAT_A8, width_px, height_px)
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(1, 1, 1)  # Use white for valid area
        ctx.scale(self.resolution, self.resolution)  # Scale context to mm

        # Draw path from geometry data.
        last_pos = (0.0, 0.0, 0.0)
        for cmd_type, x, y, z, i, j, cw in geometry_for_render.iter_commands():
            end = (x, y, z)

            if cmd_type == 0:  # CMD_TYPE_MOVE
                ctx.move_to(end[0], end[1])
            elif cmd_type == 1:  # CMD_TYPE_LINE
                ctx.line_to(end[0], end[1])
            elif cmd_type == 2:  # CMD_TYPE_ARC
                segments = linearize_arc(
                    (cmd_type, x, y, z, i, j, cw), last_pos
                )
                for _, p2 in segments:
                    ctx.line_to(p2[0], p2[1])
            last_pos = end
        ctx.fill()

        # 4. Extract the pixel data into a NumPy array.
        buf = surface.get_data()
        mask = np.frombuffer(buf, dtype=np.uint8).reshape(
            (height_px, surface.get_stride())
        )
        # We flip Y-axis (np.flipud) because Cairo's origin is top-left,
        # while our application's world space is bottom-left.
        return np.flipud(mask[:, :width_px] > 0)

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
        return groups, int(total_area_px)

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
    ) -> Tuple[List[PlacedItem], List[DocItem]]:
        """
        Places workpiece variants onto the canvas greedily.

        Args:
            item_groups: A list of variant lists, one for each workpiece.
            canvas: The 2D numpy array to pack items onto.
            context: The execution context for reporting progress.

        Returns:
            A tuple containing:
            - A list of final `PlacedItem` instances.
            - A list of `DocItem`s that could not be placed.
        """
        placements: List[PlacedItem] = []
        placed_bounds_px: List[Tuple[int, int, int, int]] = []
        # Create a dictionary of all items to be placed, for easy removal.
        item_dict = {group[0].item.uid: group[0].item for group in item_groups}
        total_items = len(item_groups)

        for i, variants in enumerate(item_groups):
            item_obj = variants[0].item
            logger.debug(f"Placing item: {item_obj.name}")

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
                # Remove successfully placed item from the dictionary.
                del item_dict[item_obj.uid]

                if context:
                    # Calculate progress within 0.1 to 0.9 range allocated
                    # for the packing phase (an 80% span).
                    pack_progress = (i + 1) / total_items
                    total_progress = 0.1 + (pack_progress * 0.8)
                    context.set_progress(total_progress)
                    context.set_message(
                        f"Packing item {i + 1} of {total_items}..."
                    )
            else:
                logger.warning(f"Could not place item {item_obj.name}.")

        # Any items remaining in the dictionary are the ones that failed.
        unplaced_items = list(item_dict.values())
        return placements, unplaced_items

    @staticmethod
    def _get_placement_bounds(
        placement: PlacedItem,
    ) -> Tuple[int, int, int, int]:
        """Calculates the (x0, y0, x1, y1) bounds of a placed item."""
        y_px, x_px = placement.position_px
        h_px, w_px = placement.variant.mask.shape
        return (x_px, y_px, x_px + w_px, y_px + h_px)

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
        Converts a list of pixel placements into transform deltas.

        Args:
            placements: The list of `PlacedItem`s.
            group_offset: The (x, y) world coordinate of the packing origin.

        Returns:
            A dictionary mapping each DocItem to its required delta matrix.
        """
        deltas: Dict[DocItem, Matrix] = {}
        if not placements:
            return deltas
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

        # 1. Calculate the final position of the rotated bbox corner in world
        # space
        true_x_px = x_px + margin_px
        true_y_px = y_px + margin_px

        if isinstance(doc_item, Group):
            # A Group is a rigid body. Calculate a pure
            # rotation/translation delta to move it from its old state to
            # the new packed state without altering its internal scale/shear.
            W_old = doc_item.get_world_transform()
            _, _, angle_old, _, _, _ = W_old.decompose()
            old_bbox = self._get_item_world_bbox(doc_item)
            if not old_bbox:
                return doc_item, Matrix.identity()
            C_old = (
                (old_bbox[0] + old_bbox[2]) / 2,
                (old_bbox[1] + old_bbox[3]) / 2,
            )

            target_angle = item.variant.angle_offset
            angle_delta = target_angle - angle_old

            rotated_bbox = item.variant.local_bbox
            w_mm = rotated_bbox[2] - rotated_bbox[0]
            h_mm = rotated_bbox[3] - rotated_bbox[1]
            C_new_px = (
                true_x_px + (w_mm * self.resolution) / 2,
                true_y_px + (h_mm * self.resolution) / 2,
            )
            C_new = (
                group_offset_x + C_new_px[0] / self.resolution,
                group_offset_y + C_new_px[1] / self.resolution,
            )

            # Create a world-space delta transform: translate, then rotate
            delta_T = Matrix.translation(
                C_new[0] - C_old[0], C_new[1] - C_old[1]
            )
            delta_R = Matrix.rotation(angle_delta, center=C_old)
            delta_world = delta_T @ delta_R

            final_matrix = delta_world @ W_old
        else:
            # For a WorkPiece, reconstruct its transform from scratch.
            packed_x = group_offset_x + (true_x_px / self.resolution)
            packed_y = group_offset_y + (true_y_px / self.resolution)
            bbox_off_x, bbox_off_y = (
                item.variant.local_bbox[0],
                item.variant.local_bbox[1],
            )
            final_x = packed_x - bbox_off_x
            final_y = packed_y - bbox_off_y
            T = Matrix.translation(final_x, final_y)
            target_angle = item.variant.angle_offset
            w_mm, h_mm = item.variant.unrotated_size_mm
            S = Matrix.scale(w_mm, h_mm)
            center_for_rot = (w_mm / 2, h_mm / 2)
            R = Matrix.rotation(target_angle, center=center_for_rot)
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
            # Use the item's own render method which now delegates to the hub
            source_surface = item.render_to_pixels(
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

                wp_surf = wp.render_to_pixels(
                    width=int(wp_w * self.resolution),
                    height=int(wp_h * self.resolution),
                )
                if not wp_surf:
                    ctx.restore()
                    continue

                # Get the workpiece's world transform.
                world_transform = wp.get_world_transform()

                # Robustly get child's world center and map it to the
                # Y-down group canvas to create an accurate snapshot.
                wp_bbox = self._get_item_world_bbox(wp)
                if not wp_bbox:
                    ctx.restore()
                    continue
                wp_center_x = (wp_bbox[0] + wp_bbox[2]) / 2
                wp_center_y = (wp_bbox[1] + wp_bbox[3]) / 2

                x_pos_px = (wp_center_x - min_x_world) * self.resolution
                y_pos_px = (max_y_world - wp_center_y) * self.resolution

                _, _, angle, _, _, _ = world_transform.decompose()

                # Standard translate-rotate-translate pattern around the center
                ctx.translate(x_pos_px, y_pos_px)
                ctx.rotate(
                    -math.radians(angle)
                )  # Negate for Cairo's clockwise
                ctx.translate(
                    -wp_surf.get_width() / 2, -wp_surf.get_height() / 2
                )

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
        ctx.rotate(-math.radians(angle_offset))
        ctx.translate(-src_w / 2, -src_h / 2)
        ctx.set_source_surface(source_surface, 0, 0)
        ctx.paint()

        # 4. Extract the mask data from the cairo surface into a numpy array.
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

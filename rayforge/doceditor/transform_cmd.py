from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Tuple, Optional
from ..context import get_context
from ..core.item import DocItem
from ..core.matrix import Matrix
from ..core.undo import ChangePropertyCommand

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class TransformCmd:
    """Handles undoable transformations of document items."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def create_transform_transaction(
        self,
        changes: List[Tuple[DocItem, Matrix, Matrix]],
    ):
        """
        Creates a single, undoable transaction for a list of matrix changes
        that have already been calculated.

        Args:
            changes: A list of tuples, where each tuple contains
                (DocItem_to_change, old_matrix, new_matrix).
        """
        history_manager = self._editor.history_manager
        if not changes:
            return

        logger.debug(
            f"Creating transform transaction for {len(changes)} item(s)."
        )

        with history_manager.transaction(_("Transform item(s)")) as t:
            for doc_item, old_matrix, new_matrix in changes:
                if old_matrix.is_close(new_matrix):
                    continue

                cmd = ChangePropertyCommand(
                    target=doc_item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

    def nudge_items(
        self,
        items: List[DocItem],
        dx_mm: float,
        dy_mm: float,
    ):
        """
        Moves a list of DocItems by a given delta in world coordinates,
        creating a single undoable transaction for the operation.

        Args:
            items: The list of DocItems to move.
            dx_mm: The distance to move along the X-axis in millimeters.
            dy_mm: The distance to move along the Y-axis in millimeters.
        """
        history_manager = self._editor.history_manager
        if not items or (dx_mm == 0.0 and dy_mm == 0.0):
            return

        with history_manager.transaction(_("Move item(s)")) as t:
            for item in items:
                old_matrix = item.matrix.copy()
                # Nudge must be pre-multiplied to apply the translation in
                # world space, not local space.
                delta = Matrix.translation(dx_mm, dy_mm)
                new_matrix = delta @ old_matrix

                if old_matrix.is_close(new_matrix):
                    continue

                cmd = ChangePropertyCommand(
                    target=item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

    def flip_horizontal(self, items: List[DocItem]):
        """
        Flips a list of DocItems horizontally (mirrors along the Y-axis),
        creating a single undoable transaction for the operation.

        Args:
            items: The list of DocItems to flip horizontally.
        """
        history_manager = self._editor.history_manager
        if not items:
            return

        with history_manager.transaction(_("Flip Horizontal")) as t:
            for item in items:
                old_matrix = item.matrix.copy()
                # Get the world center of the item before transformation
                # This ensures we always flip around the same point
                world_center = item.get_world_transform().transform_point(
                    (0.5, 0.5)
                )

                # Create a flip matrix (scale by -1 on X-axis) around world
                # center
                flip_matrix = Matrix.flip_horizontal(center=world_center)
                new_matrix = flip_matrix @ old_matrix

                if old_matrix.is_close(new_matrix):
                    continue

                cmd = ChangePropertyCommand(
                    target=item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

    def flip_vertical(self, items: List[DocItem]):
        """
        Flips a list of DocItems vertically (mirrors along the X-axis),
        creating a single undoable transaction for the operation.

        Args:
            items: The list of DocItems to flip vertically.
        """
        history_manager = self._editor.history_manager
        if not items:
            return

        with history_manager.transaction(_("Flip Vertical")) as t:
            for item in items:
                old_matrix = item.matrix.copy()
                # Get the world center of the item before transformation
                # This ensures we always flip around the same point
                world_center = item.get_world_transform().transform_point(
                    (0.5, 0.5)
                )

                # Create a flip matrix (scale by -1 on Y-axis) around world
                # center
                flip_matrix = Matrix.flip_vertical(center=world_center)
                new_matrix = flip_matrix @ old_matrix

                if old_matrix.is_close(new_matrix):
                    continue

                cmd = ChangePropertyCommand(
                    target=item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

    def set_position(self, items: List[DocItem], x: float, y: float):
        """
        Sets the position of one or more items using Machine Coordinates.
        The coordinates are converted to World Coordinates based on the
        active machine configuration.

        Args:
            items: List of items to move.
            x: Target X position in machine coordinates.
            y: Target Y position in machine coordinates.
        """
        history_manager = self._editor.history_manager
        if not items:
            return

        machine = get_context().machine

        with history_manager.transaction(_("Move item(s)")) as t:
            for item in items:
                old_matrix = item.matrix.copy()

                # Convert target Machine Coordinate to World Coordinate
                # We need the item's size for correct conversion if origin is
                # right/top.
                size_world = item.size

                if machine:
                    x_world, y_world = machine.machine_to_world(
                        (x, y), size_world
                    )
                else:
                    # Fallback to direct mapping if no machine context
                    x_world, y_world = x, y

                current_pos = item.pos
                dx = x_world - current_pos[0]
                dy = y_world - current_pos[1]

                # Apply translation to matrix
                new_matrix = Matrix.translation(dx, dy) @ old_matrix

                if old_matrix.is_close(new_matrix):
                    continue

                cmd = ChangePropertyCommand(
                    target=item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

    def set_size(
        self,
        items: List[DocItem],
        width: Optional[float] = None,
        height: Optional[float] = None,
        fixed_ratio: bool = False,
        sizes: Optional[List[Tuple[float, float]]] = None,
    ):
        """
        Sets the size of one or more items, creating a single undoable
        transaction for the operation.

        Args:
            items: The list of DocItems to resize.
            width: The target width. Ignored if `sizes` is provided.
            height: The target height. Ignored if `sizes` is provided.
            fixed_ratio: If True, calculates the missing dimension based on
                         aspect ratio if one dimension is None.
            sizes: A list of (width, height) tuples, one for each item.
                   If provided, this takes precedence over `width` and
                   `height`.
        """
        history_manager = self._editor.history_manager
        if not items:
            return

        if sizes is not None and len(sizes) != len(items):
            logger.error(
                "Length of sizes list must match length of items list."
            )
            return

        def _calculate_missing_dim(
            item: DocItem, w: Optional[float], h: Optional[float]
        ) -> Tuple[float, float]:
            """Calculates final width and height handling aspect ratio."""
            current_w, current_h = item.size
            final_w = w if w is not None else current_w
            final_h = h if h is not None else current_h

            if fixed_ratio:
                aspect_ratio = item.get_current_aspect_ratio()
                if aspect_ratio:
                    if w is not None and h is None:
                        final_h = final_w / aspect_ratio
                    elif h is not None and w is None:
                        final_w = final_h * aspect_ratio

            return final_w, final_h

        with history_manager.transaction(_("Resize item(s)")) as t:
            for i, item in enumerate(items):
                old_matrix = item.matrix.copy()

                if sizes is not None:
                    new_width, new_height = sizes[i]
                else:
                    new_width, new_height = _calculate_missing_dim(
                        item, width, height
                    )

                # The set_size method will rebuild the matrix,
                # preserving pos/angle
                item.set_size(new_width, new_height)
                new_matrix = item.matrix.copy()

                if old_matrix.is_close(new_matrix):
                    continue

                cmd = ChangePropertyCommand(
                    target=item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

    def set_angle(self, items: List[DocItem], angle: float):
        """
        Sets the angle of one or more items, creating a single undoable
        transaction for the operation.
        """
        history_manager = self._editor.history_manager
        if not items:
            return

        with history_manager.transaction(_("Change item angle")) as t:
            for item in items:
                old_matrix = item.matrix.copy()
                item.angle = angle
                new_matrix = item.matrix.copy()

                if old_matrix.is_close(new_matrix):
                    continue

                cmd = ChangePropertyCommand(
                    target=item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

    def set_shear(self, items: List[DocItem], shear: float):
        """
        Sets the shear of one or more items, creating a single undoable
        transaction for the operation.
        """
        history_manager = self._editor.history_manager
        if not items:
            return

        with history_manager.transaction(_("Change item shear")) as t:
            for item in items:
                old_matrix = item.matrix.copy()
                item.shear = shear
                new_matrix = item.matrix.copy()

                if old_matrix.is_close(new_matrix):
                    continue

                cmd = ChangePropertyCommand(
                    target=item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

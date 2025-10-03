from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Tuple, Optional
from ..core.item import DocItem
from ..core.workpiece import WorkPiece
from ..core.stock import StockItem
from ..core.matrix import Matrix
from ..undo import ChangePropertyCommand
from ..config import config

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

                cmd = ChangePropertyCommand(
                    target=item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

    def set_position(self, items: List[DocItem], x: float, y: float):
        """
        Sets the position of one or more items, creating a single undoable
        transaction for the operation.
        """
        history_manager = self._editor.history_manager
        if not items:
            return

        with history_manager.transaction(_("Move item(s)")) as t:
            for item in items:
                old_matrix = item.matrix.copy()

                # Calculate the target Y in world coordinates (Y-up)
                size_world = item.size
                x_world = x

                if config.machine and config.machine.y_axis_down:
                    machine_height = config.machine.dimensions[1]
                    y_world = machine_height - y - size_world[1]
                else:
                    y_world = y

                new_pos_world = (x_world, y_world)
                item.pos = new_pos_world
                new_matrix = item.matrix.copy()

                if old_matrix == new_matrix:
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
            fixed_ratio: If True, maintains aspect ratio based on the first
                         item. Ignored if `sizes` is provided.
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

        def _calculate_new_size_with_ratio(
            item: DocItem, value: float, changed_dim: str
        ) -> Tuple[Optional[float], Optional[float]]:
            """Calculates new width and height maintaining aspect ratio."""
            aspect_ratio = None
            if isinstance(item, (WorkPiece, StockItem)):
                aspect_ratio = item.get_current_aspect_ratio()

            if not aspect_ratio:
                return None, None

            if changed_dim == "width":
                new_width = value
                new_height = new_width / aspect_ratio
            else:  # changed_dim == 'height'
                new_height = value
                new_width = new_height * aspect_ratio

            return new_width, new_height

        with history_manager.transaction(_("Resize item(s)")) as t:
            for i, item in enumerate(items):
                old_matrix = item.matrix.copy()

                if sizes is not None:
                    new_width, new_height = sizes[i]
                else:
                    new_width, new_height = item.size
                    if width is not None:
                        new_width = width
                    if height is not None:
                        new_height = height

                    if fixed_ratio:
                        w, h = _calculate_new_size_with_ratio(
                            item, new_width, "width"
                        )
                        if w is not None and h is not None:
                            new_width, new_height = w, h

                # The set_size method will rebuild the matrix,
                # preserving pos/angle
                item.set_size(new_width, new_height)
                new_matrix = item.matrix.copy()

                if old_matrix == new_matrix:
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

                if old_matrix == new_matrix:
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

                if old_matrix == new_matrix:
                    continue

                cmd = ChangePropertyCommand(
                    target=item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

from __future__ import annotations

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional, Tuple

from raygeo.geo import Matrix

from ..context import get_context
from ..core.item import DocItem
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

    @staticmethod
    def group_bbox_world(
        items: List[DocItem],
    ) -> Tuple[float, float, float, float]:
        """Returns ``(min_x, min_y, max_x, max_y)`` of the items' combined
        axis-aligned bounding box in world space.

        A DocItem's local coordinate space is the unit square
        ``[0, 1] x [0, 1]``; the item's world transform encodes its size
        as a scale factor, so we sample the four unit-square corners and
        transform them into world space.
        """
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")
        for item in items:
            world_transform = item.get_world_transform()
            for lx, ly in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                wx, wy = world_transform.transform_point((lx, ly))
                min_x, max_x = min(min_x, wx), max(max_x, wx)
                min_y, max_y = min(min_y, wy), max(max_y, wy)
        return min_x, min_y, max_x, max_y

    @classmethod
    def group_center_world(cls, items: List[DocItem]) -> Tuple[float, float]:
        """World-space centre of the items' combined bounding box."""
        min_x, min_y, max_x, max_y = cls.group_bbox_world(items)
        return (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

    @staticmethod
    def _world_to_local_matrix(
        item: DocItem, world_transform: "Matrix"
    ) -> "Matrix":
        """Convert a world-space transform back to the item's local matrix
        by cancelling out the parent's world transform."""
        if item.parent:
            parent_world = item.parent.get_world_transform()
            try:
                return parent_world.invert() @ world_transform
            except Exception:
                return item.matrix.copy()
        return world_transform

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
        Sets the position of **every** item individually using machine
        coordinates.  Each item's top-left corner is moved to the world
        position derived from ``(x, y)`` and that item's own size, so
        items with different sizes land at different world positions.

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
                    space = machine.get_coordinate_space()
                    x_world, y_world = space.machine_item_to_world(
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

    def set_angle(self, items: List[DocItem], angle: float):
        """Sets **every** item's local rotation angle to *angle* degrees,
        preserving each item's own world-space center."""
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
        """Sets **every** item's local shear angle to *shear* degrees,
        preserving each item's own world-space center."""
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

    def set_size(
        self,
        items: List[DocItem],
        width: Optional[float] = None,
        height: Optional[float] = None,
        fixed_ratio: bool = False,
        sizes: Optional[List[Tuple[float, float]]] = None,
    ):
        """Sets the size of each item individually.

        Args:
            items: The list of DocItems to resize.
            width: The target width. Ignored if ``sizes`` is provided.
            height: The target height. Ignored if ``sizes`` is provided.
            fixed_ratio: If True, calculates the missing dimension based on
                         aspect ratio if one dimension is None.
            sizes: A list of ``(width, height)`` tuples, one per item.
                   If provided this takes precedence over ``width``/``height``.
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

    def set_position_group(self, items: List[DocItem], x: float, y: float):
        """Moves the selection so its **combined bounding box** reaches
        the given target in machine coordinates.

        The ``(x, y)`` target is the desired machine-space position of the
        bounding-box corner that the machine origin refers to (top-left,
        top-right, bottom-left or bottom-right).  The conversion uses the
        combined size of all items, so the whole group lands exactly at
        the target.
        """
        history_manager = self._editor.history_manager
        if not items:
            return

        bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = self.group_bbox_world(
            items
        )
        group_size = (
            bbox_max_x - bbox_min_x,
            bbox_max_y - bbox_min_y,
        )

        machine = get_context().machine
        if machine:
            space = machine.get_coordinate_space()
            target_world = space.machine_item_to_world((x, y), group_size)
        else:
            target_world = (x, y)

        dx = target_world[0] - bbox_min_x
        dy = target_world[1] - bbox_min_y

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return

        with history_manager.transaction(_("Move item(s)")) as t:
            for item in items:
                old_matrix = item.matrix.copy()
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

    def set_angle_group(self, items: List[DocItem], angle: float):
        """Rotates the whole selection so the anchor item (``items[0]``)
        reaches *angle* degrees.

        The rotation is applied as a world-space delta around the group's
        bounding-box centre.  Every item receives the same delta, so
        relative positions are preserved and the group rotates as a whole.
        """
        history_manager = self._editor.history_manager
        if not items:
            return

        anchor = items[0]
        delta_angle = angle - anchor.angle
        if abs(delta_angle - round(delta_angle / 360.0) * 360.0) < 1e-9:
            return

        center = self.group_center_world(items)

        with history_manager.transaction(_("Change item angle")) as t:
            for item in items:
                old_matrix = item.matrix.copy()
                world_transform_old = item.get_world_transform()
                rotate_transform_world = Matrix.rotation(
                    delta_angle, center=center
                )
                world_transform_new = (
                    rotate_transform_world @ world_transform_old
                )

                new_matrix = self._world_to_local_matrix(
                    item, world_transform_new
                )

                if old_matrix.is_close(new_matrix):
                    continue

                cmd = ChangePropertyCommand(
                    target=item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

    def set_shear_group(self, items: List[DocItem], shear: float):
        """Shears the whole selection so the anchor item (``items[0]``)
        reaches *shear* degrees.

        The shear delta is applied to each item's local matrix rather than
        as a world-space shear around the group centre.  This ensures the
        operation is stateless and idempotent: shear does not commute with
        translation, so composing world-space deltas would accumulate
        decomposed drift between pieces and prevent reset from converging.
        Every item's local shear changes by the same delta so they all
        display the same value.
        """
        history_manager = self._editor.history_manager
        if not items:
            return

        anchor = items[0]
        delta_deg = shear - anchor.shear
        if abs(delta_deg) < 1e-9:
            return

        with history_manager.transaction(_("Change item shear")) as t:
            for item in items:
                target_shear = item.shear + delta_deg
                old_matrix = item.matrix.copy()
                item.shear = target_shear
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

    def set_size_group(
        self,
        items: List[DocItem],
        width: Optional[float] = None,
        height: Optional[float] = None,
        fixed_ratio: bool = False,
    ):
        """Resizes the whole selection uniformly so the combined bounding
        box reaches the given *width* and *height* in world space.

        Each item is scaled around the group's centre by the same
        (scale_x, scale_y) factor, preserving item-to-item offsets.

        When only one dimension is provided and *fixed_ratio* is ``True``,
        the missing dimension is calculated from the group's current aspect
        ratio.
        """
        history_manager = self._editor.history_manager
        if not items:
            return

        bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = self.group_bbox_world(
            items
        )
        cur_w = bbox_max_x - bbox_min_x
        cur_h = bbox_max_y - bbox_min_y

        if cur_w < 1e-9 or cur_h < 1e-9:
            return

        if width is None and height is None:
            return

        final_w: float = cur_w
        final_h: float = cur_h

        if width is not None:
            final_w = width
        if height is not None:
            final_h = height

        if fixed_ratio:
            if width is not None and height is None:
                final_h = final_w * cur_h / cur_w
            elif height is not None and width is None:
                final_w = final_h * cur_w / cur_h

        scale_x = final_w / cur_w
        scale_y = final_h / cur_h

        center = (
            (bbox_min_x + bbox_max_x) / 2.0,
            (bbox_min_y + bbox_max_y) / 2.0,
        )

        with history_manager.transaction(_("Resize item(s)")) as t:
            for item in items:
                old_matrix = item.matrix.copy()
                world_old = item.get_world_transform()

                # Build world-space scale around centre
                scale_matrix = (
                    Matrix.identity()
                    .post_translate(center[0], center[1])
                    .post_scale(scale_x, scale_y)
                    .post_translate(-center[0], -center[1])
                )
                world_new = scale_matrix @ world_old
                new_matrix = self._world_to_local_matrix(item, world_new)

                if old_matrix.is_close(new_matrix):
                    continue

                cmd = ChangePropertyCommand(
                    target=item,
                    property_name="matrix",
                    new_value=new_matrix,
                    old_value=old_matrix,
                )
                t.execute(cmd)

    def reset_position(self, items: List[DocItem]):
        """Moves every item's top-left corner to machine (0, 0)."""
        self.set_position(items, 0.0, 0.0)

    def reset_position_group(self, items: List[DocItem]):
        """Moves the selection's bounding box so its origin-corner sits
        at machine (0, 0)."""
        self.set_position_group(items, 0.0, 0.0)

    def reset_angle(self, items: List[DocItem]):
        """Sets every item's angle to 0°."""
        self.set_angle(items, 0.0)

    def reset_angle_group(self, items: List[DocItem]):
        """Resets the group's rotation to 0°."""
        self.set_angle_group(items, 0.0)

    def reset_shear(self, items: List[DocItem]):
        """Sets every item's shear to 0°."""
        self.set_shear(items, 0.0)

    def reset_shear_group(self, items: List[DocItem]):
        """Resets the group's shear to 0°."""
        self.set_shear_group(items, 0.0)

    @classmethod
    def get_position_group(
        cls, items: List[DocItem]
    ) -> Optional[Tuple[float, float]]:
        """Machine-coordinate position of the group's bounding-box
        origin-corner (the corner the machine origin refers to).

        Returns ``None`` when the list is empty.
        """
        if not items:
            return None
        machine = get_context().machine
        min_x, min_y, max_x, max_y = cls.group_bbox_world(items)
        gw, gh = max_x - min_x, max_y - min_y
        if machine:
            space = machine.get_coordinate_space()
            return space.world_item_to_machine((min_x, min_y), (gw, gh))
        return (min_x, min_y)

    @classmethod
    def get_size_group(
        cls, items: List[DocItem]
    ) -> Optional[Tuple[float, float]]:
        """World-space (width, height) of the group bounding box."""
        if not items:
            return None
        min_x, min_y, max_x, max_y = cls.group_bbox_world(items)
        return (max_x - min_x, max_y - min_y)

    @classmethod
    def get_angle_group(cls, items: List[DocItem]) -> Optional[float]:
        """Angle (degrees) of the anchor item, representing the group."""
        if not items:
            return None
        return items[0].angle

    @classmethod
    def get_shear_group(cls, items: List[DocItem]) -> Optional[float]:
        """Shear (degrees) of the anchor item, representing the group."""
        if not items:
            return None
        return items[0].shear

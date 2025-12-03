from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from typing import (
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Iterable,
    TYPE_CHECKING,
    overload,
    Dict,
)
import logging
import numpy as np
from blinker import Signal
from .matrix import Matrix

if TYPE_CHECKING:
    from .asset import IAsset
    from .doc import Doc

logger = logging.getLogger(__name__)

# For generic type hinting in add_child, etc.
T = TypeVar("T", bound="DocItem")
# For generic type hinting in get_descendants
T_Desc = TypeVar("T_Desc", bound="DocItem")


class DocItem(ABC):
    """
    An abstract base class for any item that can exist in a document's
    hierarchy. Implements the Composite design pattern for tree management
    and automatic signal bubbling.
    """

    def __init__(self, name: str = ""):
        self.uid: str = str(uuid.uuid4())
        self._name: str = name
        self._parent: Optional[DocItem] = None
        self.children: List[DocItem] = []
        self._matrix: Matrix = Matrix.identity()

        # Signals
        # Fired when this item's own data (not transform or children) changes.
        self.updated = Signal()
        # Fired when this item's own transform changes.
        self.transform_changed = Signal()

        # Bubbled Signals
        # Fired when a descendant is added anywhere in the subtree.
        self.descendant_added = Signal()
        # Fired when a descendant is removed anywhere in the subtree.
        self.descendant_removed = Signal()
        # Fired when a descendant's `updated` signal is fired.
        self.descendant_updated = Signal()
        # Fired when a descendant's `transform_changed` signal is fired.
        self.descendant_transform_changed = Signal()

        self._natural_size: Tuple[float, float] = (0.0, 0.0)

    @property
    def name(self) -> str:
        """The user-facing name of the item."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        """Sets the item name and sends an update signal if changed."""
        if self._name != new_name:
            self._name = new_name
            self.updated.send(self)

    def depends_on_asset(self, asset: "IAsset") -> bool:
        """
        Checks if this item has a direct dependency on the given asset.
        Subclasses should override this to check their specific asset links.
        By default, items have no asset dependencies.
        """
        return False

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """
        The world-space bounding box of the item as (x, y, width, height).
        """
        x, y = self.pos
        w, h = self.size
        return x, y, w, h

    @abstractmethod
    def to_dict(self) -> Dict:
        """Serializes the item to a dictionary."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict) -> "DocItem":
        """Deserializes the item from a dictionary."""
        raise NotImplementedError

    def __iter__(self):
        """
        Provides a non-recursive iterator over the item's direct children.
        """
        return iter(self.children)

    @property
    def parent(self) -> Optional[DocItem]:
        """The parent DocItem in the hierarchy."""
        return self._parent

    @parent.setter
    def parent(self, new_parent: Optional[DocItem]):
        """
        Sets the parent of this item. This is typically managed by the
        parent's add/remove_child methods and should not be set directly.
        """
        self._parent = new_parent

    @property
    def doc(self) -> Optional["Doc"]:
        """The root Doc object, accessed via the parent hierarchy."""
        if self.parent:
            return self.parent.doc
        return None

    @property
    def pos(self) -> Tuple[float, float]:
        """
        The position (in mm) of the items's top-left corner in world space.
        """
        # The position is the world-space location of the local origin (0,0).
        return self.get_world_transform().transform_point((0.0, 0.0))

    @pos.setter
    def pos(self, new_pos_world: Tuple[float, float]):
        """
        Sets the world-space position of the items's top-left corner
        by manipulating the matrix's translation component.
        """
        world_transform_old = self.get_world_transform()
        current_pos_world = world_transform_old.transform_point((0.0, 0.0))
        delta_x = new_pos_world[0] - current_pos_world[0]
        delta_y = new_pos_world[1] - current_pos_world[1]

        if abs(delta_x) < 1e-9 and abs(delta_y) < 1e-9:
            return

        # Create the translation in world coordinates
        translate_transform_world = Matrix.translation(delta_x, delta_y)

        # Calculate the new desired world transform
        world_transform_new = translate_transform_world @ world_transform_old

        # Back-calculate the new local matrix
        if self.parent:
            parent_world_transform = self.parent.get_world_transform()
            try:
                parent_world_inv = parent_world_transform.invert()
                new_local_matrix = parent_world_inv @ world_transform_new
            except np.linalg.LinAlgError:
                logger.warning(
                    "Cannot set pos: parent transform is not invertible."
                )
                return
        else:
            new_local_matrix = world_transform_new

        self.matrix = new_local_matrix

    @property
    def size(self) -> Tuple[float, float]:
        """
        The world-space size (width, height) in mm, as absolute values,
        decomposed from the world transformation matrix.
        """
        return self.get_world_transform().get_abs_scale()

    def set_size(self, width_mm: float, height_mm: float):
        """
        Sets the item size in mm while preserving its world-space center
        point. This manipulates the existing matrix.
        """
        # Guard against zero dimensions to prevent singular matrices in Cairo
        width_mm = max(abs(width_mm), 1e-9)
        height_mm = max(abs(height_mm), 1e-9)

        world_transform_old = self.get_world_transform()
        current_w, current_h = world_transform_old.get_abs_scale()

        if (
            abs(width_mm - current_w) < 1e-9
            and abs(height_mm - current_h) < 1e-9
        ):
            return

        # Decompose the existing world transform to preserve its properties
        tx, ty, angle, sx, sy, skew = world_transform_old.decompose()

        # Preserve any reflection by checking the sign of the original scale
        new_sx = width_mm * (1 if sx >= 0 else -1)
        new_sy = height_mm * (1 if sy >= 0 else -1)

        # Compose a new world matrix with the new scale, but without its
        # translation corrected yet.
        # We use (0,0) for translation initially, as we will correct the
        # center point manually.
        world_transform_new_uncorrected = Matrix.compose(
            0, 0, angle, new_sx, new_sy, skew
        )

        # The old center point that we must maintain
        center_world_old = world_transform_old.transform_point((0.5, 0.5))

        # The center point of our newly composed matrix (before translation)
        center_world_new_uncorrected = (
            world_transform_new_uncorrected.transform_point((0.5, 0.5))
        )

        # Calculate the required translation to move the new center to the old
        # center's position.
        final_tx = center_world_old[0] - center_world_new_uncorrected[0]
        final_ty = center_world_old[1] - center_world_new_uncorrected[1]

        # Create the final desired world matrix
        world_transform_new = world_transform_new_uncorrected.set_translation(
            final_tx, final_ty
        )

        # Back-calculate the new local matrix
        if self.parent:
            parent_world_transform = self.parent.get_world_transform()
            try:
                parent_world_inv = parent_world_transform.invert()
                new_local_matrix = parent_world_inv @ world_transform_new
            except np.linalg.LinAlgError:
                logger.warning(
                    "Cannot set size: parent transform is not invertible."
                )
                return
        else:
            new_local_matrix = world_transform_new

        self.matrix = new_local_matrix

    @property
    def natural_size(self) -> Tuple[float, float]:
        """
        Returns the natural size (untransformed width and height) of this item.

        For generic items (like Groups), this is the size of the bounding box
        that encloses all children in the item's local coordinate space,
        calculated at the time of the last structural change (adding or
        removing children).

        If an item has no children and provides no intrinsic size (like
        WorkPiece or StockItem do), this returns None.
        """
        return self._natural_size

    def get_local_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns the bounding box of the item in its own local coordinate space
        (before the local matrix is applied).

        Returns:
            (min_x, min_y, width, height) or None
        """
        if self.natural_size:
            return (0.0, 0.0, self.natural_size[0], self.natural_size[1])
        return None

    def _recalculate_natural_size(self):
        """
        Recalculates the natural size based on the current children.
        """
        if not self.children:
            self._natural_size = (0.0, 0.0)
            return

        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
        has_valid_children = False

        for child in self.children:
            child_bbox = child.get_local_bbox()
            if child_bbox is None:
                continue

            bx, by, bw, bh = child_bbox
            # The child's local bounds.
            # We transform these corners to the parent's (self) local space.
            corners = [
                (bx, by),
                (bx + bw, by),
                (bx + bw, by + bh),
                (bx, by + bh),
            ]
            transformed_corners = [
                child.matrix.transform_point(c) for c in corners
            ]

            has_valid_children = True
            for x, y in transformed_corners:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

        if has_valid_children:
            self._natural_size = (max_x - min_x, max_y - min_y)
        else:
            self._natural_size = (0.0, 0.0)

    def get_current_aspect_ratio(self) -> Optional[float]:
        w, h = self.size
        return w / h if h else None

    @property
    def angle(self) -> float:
        """
        The rotation angle (in degrees) of the item.
        This is decomposed from the local transformation matrix.
        """
        return self.matrix.get_rotation()

    @angle.setter
    def angle(self, new_angle_deg: float):
        """
        Sets the local rotation angle to a new value, preserving the item's
        world-space center point.
        """
        current_angle = self.angle
        delta_angle = new_angle_deg - current_angle

        if abs(delta_angle - round(delta_angle / 360.0) * 360.0) < 1e-9:
            return

        # Get the current world transform and the world-space center point
        # around which the rotation should occur.
        world_transform_old = self.get_world_transform()
        center_world = world_transform_old.transform_point((0.5, 0.5))

        # Create a rotation transformation that will be applied in world space
        rotate_transform_world = Matrix.rotation(
            delta_angle, center=center_world
        )

        # Calculate the new desired world transform by applying the rotation
        # to the old one.
        world_transform_new = rotate_transform_world @ world_transform_old

        # Now, back-calculate the new local matrix that will result in this
        # new world transform.
        if self.parent:
            parent_world_transform = self.parent.get_world_transform()
            try:
                parent_world_inv = parent_world_transform.invert()
                new_local_matrix = parent_world_inv @ world_transform_new
            except np.linalg.LinAlgError:
                logger.warning(
                    "Cannot set angle: parent transform is not invertible."
                )
                return
        else:
            # If there's no parent, the local matrix is the world matrix.
            new_local_matrix = world_transform_new

        self.matrix = new_local_matrix

    @property
    def shear(self) -> float:
        """
        The shear angle (in degrees) of the item.
        This is decomposed from the local transformation matrix.
        """
        # decompose returns (tx, ty, angle_deg, sx, sy, skew_angle_deg)
        return self.matrix.decompose()[5]

    @shear.setter
    def shear(self, new_shear_deg: float):
        """
        Sets the local shear angle to a new value, preserving the item's
        world-space center point.
        """
        old_shear_deg = self.shear
        if abs(new_shear_deg - old_shear_deg) < 1e-9:
            return

        # Get world center before change
        world_transform_old = self.get_world_transform()
        center_world_old = world_transform_old.transform_point((0.5, 0.5))

        # Decompose local matrix to get its non-shear components
        tx, ty, angle, sx, sy, _ = self.matrix.decompose()

        # Recompose local matrix with the new shear value
        new_local_matrix = Matrix.compose(tx, ty, angle, sx, sy, new_shear_deg)

        # Calculate the new world center based on the temporary new matrix
        parent_world_transform = (
            self.parent.get_world_transform()
            if self.parent
            else Matrix.identity()
        )
        world_transform_new_uncorrected = (
            parent_world_transform @ new_local_matrix
        )
        center_world_new = world_transform_new_uncorrected.transform_point(
            (0.5, 0.5)
        )

        # Calculate the world-space correction needed to restore the center
        delta_x = center_world_old[0] - center_world_new[0]
        delta_y = center_world_old[1] - center_world_new[1]

        # If there's no significant change, just set the matrix
        if abs(delta_x) < 1e-9 and abs(delta_y) < 1e-9:
            self.matrix = new_local_matrix
            return

        # To correct the center point, we apply a world-space translation
        # *after* the uncorrected new world transform. Then we back-calculate
        # the final local matrix.
        translate_transform_world = Matrix.translation(delta_x, delta_y)
        world_transform_new_corrected = (
            translate_transform_world @ world_transform_new_uncorrected
        )

        # Back-calculate final local matrix
        if self.parent:
            try:
                parent_world_inv = parent_world_transform.invert()
                final_local_matrix = (
                    parent_world_inv @ world_transform_new_corrected
                )
            except np.linalg.LinAlgError:
                logger.warning(
                    "Cannot set shear: parent transform is not invertible."
                )
                return
        else:
            final_local_matrix = world_transform_new_corrected

        self.matrix = final_local_matrix

    def add_child(self, child: T, index: Optional[int] = None) -> T:
        if child in self.children:
            return child

        if child.parent:
            child.parent.remove_child(child)

        if index is None:
            self.children.append(child)
        else:
            self.children.insert(index, child)

        child.parent = self
        self._connect_child_signals(child)
        self._recalculate_natural_size()
        self.descendant_added.send(self, origin=child, parent_of_origin=self)
        return child

    def remove_child(self, child: DocItem):
        if child not in self.children:
            return

        self.children.remove(child)
        child.parent = None
        self.descendant_removed.send(self, origin=child, parent_of_origin=self)
        self._disconnect_child_signals(child)
        self._recalculate_natural_size()

    def add_children(
        self, children_to_add: Iterable[DocItem], index: Optional[int] = None
    ):
        """
        Adds multiple children in a bulk operation to improve performance,
        sending a single `updated` signal after completion. It quietly
        re-parents the children if they already belong to another parent.

        Args:
            children_to_add: An iterable of DocItems to add.
            index: The index at which to insert the children. If None, they
                   are appended.
        """
        children_list = list(children_to_add)
        if not children_list:
            return

        # Quietly detach from any existing parents first.
        for child in children_list:
            if child.parent:
                try:
                    # Manually remove from old parent's list without signals
                    child.parent.children.remove(child)
                    child.parent._disconnect_child_signals(child)
                    child.parent._recalculate_natural_size()
                except (ValueError, AttributeError):
                    pass  # Failsafe if tree is in an inconsistent state
            child.parent = None

        # Add to self's children list
        if index is None:
            self.children.extend(children_list)
        else:
            self.children[index:index] = children_list

        # Update parent pointers and connect signals
        for child in children_list:
            child.parent = self
            self._connect_child_signals(child)

        self._recalculate_natural_size()
        self.updated.send(self)

    def remove_children(self, children_to_remove: Iterable[DocItem]):
        """
        Removes multiple children in a bulk operation to improve performance,
        sending a single `updated` signal after all are removed.
        """
        # Use UIDs for safe comparison in the set
        to_remove_uids = {c.uid for c in children_to_remove}
        if not to_remove_uids:
            return

        removed_items = [c for c in self.children if c.uid in to_remove_uids]
        if not removed_items:
            return

        # Rebuild the list, excluding the removed items
        self.children = [
            c for c in self.children if c.uid not in to_remove_uids
        ]

        # Update parent pointers and disconnect signals for removed items
        for child in removed_items:
            child.parent = None
            self._disconnect_child_signals(child)

        self._recalculate_natural_size()
        self.updated.send(self)

    def set_children(self, new_children: Iterable[DocItem]):
        """
        Correctly updates the list of children by mutating state first,
        then sending notifications.
        """
        old_children = list(self.children)
        new_children_list = list(new_children)

        # 1. Mutate the state immediately.
        self.children = new_children_list

        # 2. Calculate differences based on the old and new states.
        old_set = set(old_children)
        new_set = set(new_children_list)

        # 3. Process removals and notify.
        for child in old_set - new_set:
            child.parent = None
            self.descendant_removed.send(
                self, origin=child, parent_of_origin=self
            )
            self._disconnect_child_signals(child)

        # 4. Process additions and notify.
        for child in new_set - old_set:
            if child.parent:
                child.parent.remove_child(child)
            child.parent = self
            self._connect_child_signals(child)
            self.descendant_added.send(
                self, origin=child, parent_of_origin=self
            )

        self._recalculate_natural_size()

    def get_depth(self) -> int:
        """
        Calculates the depth of this item in the document hierarchy by
        counting its DocItem ancestors.

        A direct child has a depth of 1.
        An item inside that item would have a depth of 2, and so on.

        Returns:
            The integer depth of the item.
        """
        depth = 0
        current_item = self
        while current_item.parent and isinstance(current_item.parent, DocItem):
            depth += 1
            current_item = current_item.parent
        return depth

    @overload
    def get_descendants(self) -> List["DocItem"]: ...

    @overload
    def get_descendants(self, of_type: Type[T_Desc]) -> List[T_Desc]: ...

    def get_descendants(self, of_type: Optional[Type[T_Desc]] = None) -> List:
        """
        Recursively finds and returns a flattened list of all descendant
        DocItems, optionally filtered by type.
        """
        all_descendants: List[DocItem] = []
        for child in self.children:
            all_descendants.append(child)
            # This recursive call unambiguously matches the first overload.
            all_descendants.extend(child.get_descendants())

        if of_type:
            # The list comprehension correctly narrows the type for the return.
            return [
                item for item in all_descendants if isinstance(item, of_type)
            ]

        return all_descendants

    def get_child_by_uid(self, uid: str) -> Optional["DocItem"]:
        """
        Finds a direct child of this item by its unique identifier.

        Args:
            uid: The unique identifier to search for.

        Returns:
            The DocItem if found, otherwise None.
        """
        for child in self.children:
            if child.uid == uid:
                return child
        return None

    def find_descendant_by_uid(self, uid: str) -> Optional[DocItem]:
        """
        Recursively searches the subtree for a descendant with a matching UID.

        Args:
            uid: The unique identifier to search for.

        Returns:
            The DocItem if found, otherwise None.
        """
        for child in self.children:
            if child.uid == uid:
                return child
            found = child.find_descendant_by_uid(uid)
            if found:
                return found
        return None

    def _connect_child_signals(self, child: DocItem):
        child.updated.connect(self._on_child_updated)
        child.transform_changed.connect(self._on_child_transform_changed)
        child.descendant_added.connect(self._on_descendant_added)
        child.descendant_removed.connect(self._on_descendant_removed)
        child.descendant_updated.connect(self._on_descendant_updated)
        child.descendant_transform_changed.connect(
            self._on_descendant_transform_changed
        )

    def _disconnect_child_signals(self, child: DocItem):
        child.updated.disconnect(self._on_child_updated)
        child.transform_changed.disconnect(self._on_child_transform_changed)
        child.descendant_added.disconnect(self._on_descendant_added)
        child.descendant_removed.disconnect(self._on_descendant_removed)
        child.descendant_updated.disconnect(self._on_descendant_updated)
        child.descendant_transform_changed.disconnect(
            self._on_descendant_transform_changed
        )

    def _on_child_updated(self, sender: DocItem):
        self.descendant_updated.send(
            self, origin=sender, parent_of_origin=self
        )

    def _on_child_transform_changed(self, sender: DocItem):
        self.descendant_transform_changed.send(
            self, origin=sender, parent_of_origin=self
        )

    def _on_descendant_added(
        self, sender: DocItem, *, origin: DocItem, parent_of_origin: DocItem
    ):
        self.descendant_added.send(
            self, origin=origin, parent_of_origin=parent_of_origin
        )

    def _on_descendant_removed(
        self, sender: DocItem, *, origin: DocItem, parent_of_origin: DocItem
    ):
        self.descendant_removed.send(
            self, origin=origin, parent_of_origin=parent_of_origin
        )

    def _on_descendant_updated(
        self, sender: DocItem, *, origin: DocItem, parent_of_origin: DocItem
    ):
        self.descendant_updated.send(
            self, origin=origin, parent_of_origin=parent_of_origin
        )

    def _on_descendant_transform_changed(
        self, sender: DocItem, *, origin: DocItem, parent_of_origin: DocItem
    ):
        self.descendant_transform_changed.send(
            self, origin=origin, parent_of_origin=parent_of_origin
        )

    @property
    def matrix(self) -> "Matrix":
        """The 3x3 local transformation matrix for this item."""
        return self._matrix

    @matrix.setter
    def matrix(self, value: "Matrix"):
        if self._matrix == value:
            return
        self._matrix = value
        self.transform_changed.send(self)

    def get_world_transform(self) -> "Matrix":
        """
        Calculates the cumulative transformation matrix for this item,
        which transforms it from its local coordinate space into the
        document's world space.
        """
        if self.parent:
            parent_transform = self.parent.get_world_transform()
            return parent_transform @ self.matrix
        return self.matrix

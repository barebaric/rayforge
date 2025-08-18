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
        self.name: str = name
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
        current_pos_world = self.pos
        delta_x = new_pos_world[0] - current_pos_world[0]
        delta_y = new_pos_world[1] - current_pos_world[1]

        if abs(delta_x) < 1e-9 and abs(delta_y) < 1e-9:
            return

        # Apply the translation in world space
        self.matrix = self.matrix.pre_translate(delta_x, delta_y)

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
        current_w, current_h = self.size
        if (
            abs(width_mm - current_w) < 1e-9
            and abs(height_mm - current_h) < 1e-9
        ):
            return

        # Calculate scale factors to apply in world space
        scale_x = width_mm / current_w if current_w > 1e-9 else 0
        scale_y = height_mm / current_h if current_h > 1e-9 else 0

        # Get the current world transform and world-space center
        world_transform_old = self.get_world_transform()
        center_world = world_transform_old.transform_point((0.5, 0.5))

        # Create the scaling transformation in world coordinates
        scale_transform_world = Matrix.scale(
            scale_x, scale_y, center=center_world
        )

        # Calculate the new desired world transform
        world_transform_new = scale_transform_world @ world_transform_old

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
        self.descendant_added.send(self, origin=child)
        return child

    def remove_child(self, child: DocItem):
        if child not in self.children:
            return

        self.children.remove(child)
        child.parent = None
        self.descendant_removed.send(self, origin=child)
        self._disconnect_child_signals(child)

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
            self.descendant_removed.send(self, origin=child)
            self._disconnect_child_signals(child)

        # 4. Process additions and notify.
        for child in new_set - old_set:
            if child.parent:
                child.parent.remove_child(child)
            child.parent = self
            self._connect_child_signals(child)
            self.descendant_added.send(self, origin=child)

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

    def _on_child_updated(self, sender: DocItem, **kwargs):
        self.descendant_updated.send(self, origin=sender)

    def _on_child_transform_changed(self, sender: DocItem, **kwargs):
        self.descendant_transform_changed.send(self, origin=sender)

    def _on_descendant_added(self, sender: DocItem, *, origin: DocItem):
        self.descendant_added.send(self, origin=origin)

    def _on_descendant_removed(self, sender: DocItem, *, origin: DocItem):
        self.descendant_removed.send(self, origin=origin)

    def _on_descendant_updated(self, sender: DocItem, *, origin: DocItem):
        self.descendant_updated.send(self, origin=origin)

    def _on_descendant_transform_changed(
        self, sender: DocItem, *, origin: DocItem
    ):
        self.descendant_transform_changed.send(self, origin=origin)

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

from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Iterable, TYPE_CHECKING
from blinker import Signal
from .matrix import Matrix

if TYPE_CHECKING:
    from .workpiece import WorkPiece
    from .doc import Doc

# For generic type hinting in add_child, etc.
T = TypeVar("T", bound="DocItem")


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

    @abstractmethod
    def get_all_workpieces(self) -> List["WorkPiece"]:
        """
        Recursively finds and returns a flattened list of all WorkPiece
        objects contained within this item.
        """
        pass

from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
from blinker import Signal
from .matrix import Matrix

if TYPE_CHECKING:
    from .workpiece import WorkPiece
    from .doc import Doc
    from .layer import Layer


class DocItem(ABC):
    """
    An abstract base class for any item that can exist in a document's
    hierarchy, such as a WorkPiece or a group of items.
    """

    def __init__(self, name: str):
        """
        Initializes the DocItem.

        Args:
            name: The user-facing name of the item.
        """
        self.uid: str = str(uuid.uuid4())
        self.name = name
        self._parent: Optional[Layer] = None
        self._matrix: Matrix = Matrix.identity()

        # Signals for notifying of model changes
        self.changed = Signal()
        self.transform_changed = Signal()

    @property
    def matrix(self) -> "Matrix":
        """The 3x3 local transformation matrix for this item."""
        return self._matrix

    @matrix.setter
    def matrix(self, value: "Matrix"):
        """
        Sets the local transformation matrix.

        This is a transform-only operation that fires the `transform_changed`
        signal.
        """
        if self._matrix == value:
            return
        self._matrix = value
        self.transform_changed.send(self)

    @property
    def parent(self) -> Optional[Layer]:
        """The container object for this item (e.g., a Layer)."""
        return self._parent

    @parent.setter
    def parent(self, value: Optional[Layer]):
        self._parent = value

    @property
    def doc(self) -> Optional["Doc"]:
        """The root Doc object, accessed via the parent hierarchy."""
        if self.parent:
            return self.parent.doc
        return None

    def get_world_transform(self) -> "Matrix":
        """
        Calculates the cumulative transformation matrix for this item,
        which transforms it from its local coordinate space into the
        document's world space.
        """
        # If parent is a DocItem, recurse. Otherwise, this is the top-level
        # transform. A Layer is not a DocItem.
        if self.parent and isinstance(self.parent, DocItem):
            parent_transform = self.parent.get_world_transform()
            # Use matrix multiplication for composition.
            return parent_transform @ self.matrix

        return self.matrix

    @abstractmethod
    def get_all_workpieces(self) -> List["WorkPiece"]:
        """
        Recursively finds and returns a flattened list of all WorkPiece
        objects contained within this item.
        """
        pass

from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
from blinker import Signal

if TYPE_CHECKING:
    import numpy as np
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

        # Signals for notifying of model changes
        self.changed = Signal()
        self.transform_changed = Signal()

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

    @abstractmethod
    def get_world_transform(self) -> "np.ndarray":
        """
        Calculates the cumulative transformation matrix for this item,
        which transforms it from its local coordinate space into the
        document's world space.
        """
        pass

    @abstractmethod
    def get_all_workpieces(self) -> List["WorkPiece"]:
        """
        Recursively finds and returns a flattened list of all WorkPiece
        objects contained within this item.
        """
        pass

from __future__ import annotations
import logging
from typing import Optional, Dict, TYPE_CHECKING
from .item import DocItem
from .layer import Layer
from .stock import StockItem
from .workflow import Workflow

if TYPE_CHECKING:
    # For generic type hinting in add_child
    from .item import T

logger = logging.getLogger(__name__)


class StockLayer(Layer):
    """
    A specialized Layer that contains StockItems. It has no Workflow and
    serves as a container for reference geometry.
    """

    def __init__(self, name: str = "Stock"):
        """
        Initializes a StockLayer instance, bypassing the parent Layer's
        workflow creation.
        """
        # Call DocItem's __init__ directly to avoid creating a workflow.
        DocItem.__init__(self, name=name)
        self.visible: bool = True
        # No signals to set up here as we have no workflow.

    def to_dict(self) -> Dict:
        """Serializes the StockLayer and its children to a dictionary."""
        return {
            "uid": self.uid,
            "type": "stocklayer",  # Discriminator for deserialization
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "visible": self.visible,
            "children": [child.to_dict() for child in self.children],
        }

    def add_child(self, child: T, index: Optional[int] = None) -> T:
        """
        Overrides the parent method to only allow StockItem children and
        enforce that only one StockItem can exist.
        """
        if not isinstance(child, StockItem):
            raise TypeError(
                "Only StockItem instances can be added to a StockLayer."
            )

        # Enforce the single StockItem constraint.
        if any(isinstance(c, StockItem) for c in self.children):
            raise ValueError("A StockLayer can only contain one StockItem.")

        # Call super() to handle the actual parenting logic.
        return super().add_child(child, index)

    @property
    def workflow(self) -> Optional[Workflow]:
        """
        A StockLayer does not have a workflow. This property always
        returns None.
        """
        return None

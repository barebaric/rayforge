"""
Material Test Layer - A specialized layer for material testing workpieces.
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, TYPE_CHECKING
from .layer import Layer
from .workpiece import WorkPiece

if TYPE_CHECKING:
    # For generic type hinting in add_child
    from .item import T

logger = logging.getLogger(__name__)


class MaterialTestLayer(Layer):
    """
    A specialized Layer that contains material test workpieces.
    It has a workflow like a normal layer but restricts children to
    material test workpieces only.
    """

    def __init__(self, name: str = "Material Test"):
        """
        Initializes a MaterialTestLayer instance.

        Args:
            name: The user-facing name of the layer (default: "Material Test")
        """
        # Call parent Layer's __init__ to create workflow
        super().__init__(name=name)

    def to_dict(self) -> Dict:
        """Serializes the MaterialTestLayer and its children."""
        base_dict = super().to_dict()
        # Discriminator for deserialization
        base_dict["type"] = "materialtestlayer"
        return base_dict

    def add_child(self, child: T, index: Optional[int] = None) -> T:
        """
        Overrides the parent method to only allow material test workpieces.
        Allows Workflow to be added (needed during layer initialization).

        Args:
            child: The child to add (must be a WorkPiece with material test
                   source or Workflow)
            index: Optional position to insert the child

        Returns:
            The added child

        Raises:
            TypeError: If child is not a WorkPiece or Workflow
            ValueError: If workpiece is not a material test type
        """
        from .workflow import Workflow

        # Allow Workflow to be added (needed during layer initialization)
        if isinstance(child, Workflow):
            return super().add_child(child, index)

        if not isinstance(child, WorkPiece):
            raise TypeError(
                "Only WorkPiece instances can be added to a MaterialTestLayer."
            )

        # Check if it's a material test workpiece
        # (only if source is already resolved)
        # During initial creation, the source might not be resolved yet,
        # so we allow it
        source = child.source
        if source and source.metadata.get("type") != "material_test":
            raise ValueError(
                "Only material test workpieces can be added to a "
                "MaterialTestLayer."
            )

        return super().add_child(child, index)

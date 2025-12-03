from typing import Protocol, runtime_checkable, Dict, Any


@runtime_checkable
class IAsset(Protocol):
    """
    A protocol defining the common interface for all document assets.

    This allows for structural subtyping (static duck typing), so any class
    that provides these properties will be considered an IAsset.
    """

    uid: str
    """The unique identifier of the asset instance."""

    @property
    def name(self) -> str:
        """The user-facing name of the asset instance."""
        ...

    @name.setter
    def name(self, value: str) -> None: ...

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the asset to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IAsset":
        """Deserializes the asset from a dictionary."""
        ...

    @property
    def asset_type_name(self) -> str:
        """A unique, machine-readable name like "stock" or "sketch"."""
        ...

    @property
    def display_icon_name(self) -> str:
        """The name of the icon representing the asset type."""
        ...

    @property
    def is_reorderable(self) -> bool:
        """Indicates if this asset type supports manual reordering."""
        ...

    @property
    def is_draggable_to_canvas(self) -> bool:
        """Indicates if this asset can be dragged onto the canvas."""
        ...

    @property
    def hidden(self) -> bool:
        """Indicates if this asset should be hidden from UI."""
        return False

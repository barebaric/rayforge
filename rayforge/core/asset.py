from typing import Protocol, runtime_checkable, Dict, Any, ClassVar


@runtime_checkable
class IAsset(Protocol):
    """
    A protocol defining the common interface for all document assets.

    This allows for structural subtyping (static duck typing), so any class
    that provides these properties will be considered an IAsset.
    """

    is_addable: ClassVar[bool]
    asset_type_name: ClassVar[str]
    display_icon_name: ClassVar[str]
    is_reorderable: ClassVar[bool]
    is_draggable_to_canvas: ClassVar[bool]
    type_display_name: ClassVar[str]
    can_edit: ClassVar[bool]

    @property
    def uid(self) -> str:
        """The unique identifier of the asset instance."""
        ...

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
    def hidden(self) -> bool:
        """Indicates if this asset should be hidden from UI."""
        return False

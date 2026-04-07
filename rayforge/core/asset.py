import uuid
from typing import Protocol, runtime_checkable, Dict, Any, ClassVar, Optional
from blinker import Signal
from dataclasses import dataclass, field


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
    add_action: ClassVar[Optional[str]]
    activate_action: ClassVar[Optional[str]]
    edit_item_action: ClassVar[Optional[str]]

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

    @property
    def updated(self) -> Signal:
        """Signal emitted when the asset changes."""
        ...

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

    def get_thumbnail(self, size: int) -> Optional[bytes]:
        """
        Returns PNG thumbnail bytes at the given max pixel size,
        or None if no thumbnail is available.
        """
        return None


@dataclass
class UnknownAsset(IAsset):
    """Placeholder for unknown asset types during deserialization."""

    is_addable: ClassVar[bool] = False
    asset_type_name: ClassVar[str] = "unknown"
    display_icon_name: ClassVar[str] = "question-mark-symbolic"
    is_reorderable: ClassVar[bool] = False
    is_draggable_to_canvas: ClassVar[bool] = False
    type_display_name: ClassVar[str] = "Unknown Asset"
    can_edit: ClassVar[bool] = False
    add_action: ClassVar[Optional[str]] = None
    activate_action: ClassVar[Optional[str]] = None
    edit_item_action: ClassVar[Optional[str]] = None

    _original_type: str = field(init=False)
    _data: Dict[str, Any] = field(init=False, default_factory=dict)
    _uid: str = field(init=False, default_factory=lambda: str(uuid.uuid4()))
    _name: str = field(init=False)
    _updated: Signal = field(init=False, default_factory=Signal)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnknownAsset":
        """Deserializes a dictionary into an UnknownAsset instance."""
        instance = cls.__new__(cls)
        instance._original_type = data.get("type", "unknown")
        instance._data = dict(data)
        instance._uid = instance._data.get("uid", str(uuid.uuid4()))
        default_name = f"Unknown ({instance._original_type})"
        instance._name = instance._data.get("name", default_name)
        return instance

    @property
    def uid(self) -> str:
        """The unique identifier of the asset instance."""
        return self._uid

    @property
    def updated(self) -> Signal:
        return self._updated

    @property
    def name(self) -> str:
        """The user-facing name of the asset instance."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Sets the asset name."""
        self._name = value
        self._data["name"] = value

    def to_dict(self) -> Dict[str, Any]:
        """Serializes UnknownAsset to the original dictionary."""
        return self._data

    def get_thumbnail(self, size: int) -> Optional[bytes]:
        """No thumbnail available for unknown assets."""
        return None

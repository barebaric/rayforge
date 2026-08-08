from __future__ import annotations

from abc import ABC
from typing import Any, Optional

_handle_registry: dict[str, type["BaseArtifactHandle"]] = {}


class BaseArtifactHandle(ABC):
    def __init__(
        self,
        key: str,
        handle_class_name: str,
        artifact_type_name: str,
        generation_id: int,
        array_metadata: Optional[dict[str, Any]] = None,
        **_kwargs,
    ):
        self.key = key
        self.handle_class_name = handle_class_name
        self.artifact_type_name = artifact_type_name
        self.generation_id = generation_id
        self.array_metadata = (
            array_metadata if array_metadata is not None else {}
        )
        self.refcount: int = 1
        self.holders: list[str] = []

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called whenever a class inherits from
        BaseArtifactHandle. It automatically registers the new handle type.
        """
        super().__init_subclass__(**kwargs)
        _handle_registry[cls.__name__] = cls

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the handle to a dictionary. Subclasses will be handled
        correctly.
        """
        return vars(self)

    def __eq__(self, other: object) -> bool:
        """
        Provides value-based equality comparison for handle objects.
        """
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        """
        Provides a hash based on the handle key.
        """
        return hash(self.key)

    @classmethod
    def from_dict(
        cls: type[Any], data: dict[str, Any]
    ) -> "BaseArtifactHandle":
        # This simple deserialization works for direct instantiation, but the
        # factory function below should be used for polymorphic
        # deserialization.
        return cls(**data)


def create_handle_from_dict(data: dict[str, Any]) -> "BaseArtifactHandle":
    """
    Factory function to reconstruct the correct, typed handle subclass from a
    dictionary.
    """
    class_name = data.get("handle_class_name")
    if not class_name:
        raise ValueError(
            "Cannot reconstruct handle: dictionary is missing "
            "'handle_class_name'."
        )

    handle_class = _handle_registry.get(class_name)
    if not handle_class:
        raise TypeError(
            f"Unknown handle type '{class_name}'. Was its module imported?"
        )

    return handle_class.from_dict(data)

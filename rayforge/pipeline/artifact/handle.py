from __future__ import annotations
from typing import Dict, Any, Type, Optional
from abc import ABC


# A central registry for handle classes, populated automatically.
_handle_registry: Dict[str, Type["BaseArtifactHandle"]] = {}


class BaseArtifactHandle(ABC):
    """
    A lightweight, serializable handle to artifact data stored in shared
    memory. This object is small and can be passed efficiently between
    processes.
    """

    def __init__(
        self,
        shm_name: str,
        handle_class_name: str,
        artifact_type_name: str,
        array_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initializes the base handle."""
        self.shm_name = shm_name
        self.handle_class_name = handle_class_name
        self.artifact_type_name = artifact_type_name
        self.array_metadata = (
            array_metadata if array_metadata is not None else {}
        )

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called whenever a class inherits from
        BaseArtifactHandle. It automatically registers the new handle type.
        """
        super().__init_subclass__(**kwargs)
        _handle_registry[cls.__name__] = cls

    def to_dict(self) -> Dict[str, Any]:
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

    @classmethod
    def from_dict(
        cls: Type[Any], data: Dict[str, Any]
    ) -> "BaseArtifactHandle":
        # This simple deserialization works for direct instantiation, but the
        # factory function below should be used for polymorphic
        # deserialization.
        return cls(**data)


def create_handle_from_dict(data: Dict[str, Any]) -> "BaseArtifactHandle":
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

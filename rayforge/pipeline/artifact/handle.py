from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, Optional, Type


# A central registry for handle classes, populated automatically.
_handle_registry: Dict[str, Type["BaseArtifactHandle"]] = {}


@dataclass
class BaseArtifactHandle:
    """
    A lightweight, serializable handle to artifact data stored in shared
    memory. This object is small and can be passed efficiently between
    processes.
    """

    # The unique name of the shared memory block
    shm_name: str

    # The class name of the handle itself, for reconstruction.
    handle_class_name: str

    # Metadata to reconstruct the artifact
    artifact_type_name: str  # Used by the registry to find the artifact class
    is_scalable: bool
    source_coordinate_system_name: str
    source_dimensions: Optional[Tuple[float, float]]
    time_estimate: Optional[float]

    # A dictionary describing the layout of NumPy arrays within the SHM block
    # Format: {'array_name': {'dtype': str, 'shape': tuple, 'offset': int}}
    array_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called whenever a class inherits from
        BaseArtifactHandle. It automatically registers the new handle type.
        """
        super().__init_subclass__(**kwargs)
        _handle_registry[cls.__name__] = cls

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseArtifactHandle":
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

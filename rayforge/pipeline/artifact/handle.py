from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, Optional


@dataclass
class ArtifactHandle:
    """
    A lightweight, serializable handle to artifact data stored in shared
    memory. This object is small and can be passed efficiently between
    processes.

    A dataclass is the ideal choice here as this object is a pure, immutable
    data container with no associated behavior.
    """

    # The unique name of the shared memory block
    shm_name: str

    # Metadata to reconstruct the artifact
    artifact_type: str  # 'vector', 'vertex', or 'hybrid_raster'
    is_scalable: bool
    source_coordinate_system_name: str
    source_dimensions: Optional[Tuple[float, float]]
    generation_size: Optional[Tuple[float, float]]

    # Extra metadata for artifacts containing texture data ('hybrid_raster')
    dimensions_mm: Optional[Tuple[float, float]] = None
    position_mm: Optional[Tuple[float, float]] = None

    # A dictionary describing the layout of NumPy arrays within the SHM block
    # Format: {'array_name': {'dtype': str, 'shape': tuple, 'offset': int}}
    array_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactHandle":
        return cls(**data)

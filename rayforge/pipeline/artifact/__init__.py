# flake8: noqa: F401
from .base import Artifact
from .handle import ArtifactHandle
from .vector import VectorArtifact
from .hybrid import HybridRasterArtifact
from .store import ArtifactStore


def deserialize_artifact(
    data: dict,
) -> "Artifact":
    """
    Factory function that deserializes a dictionary into the correct
    artifact type based on the 'type' field.
    """
    artifact_type = data.get("type")

    if artifact_type == "hybrid_raster":
        return HybridRasterArtifact.from_dict(data)
    elif artifact_type == "vector":
        return VectorArtifact.from_dict(data)
    else:
        raise ValueError(
            f"Unknown artifact type for deserialization: {artifact_type}"
        )


__all__ = [
    "Artifact",
    "ArtifactHandle",
    "VectorArtifact",
    "HybridRasterArtifact",
    "ArtifactStore",
    "deserialize_artifact",
]

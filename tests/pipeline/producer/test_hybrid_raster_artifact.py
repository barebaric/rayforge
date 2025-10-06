import pytest
import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.producer.base import HybridRasterArtifact


@pytest.fixture
def sample_ops() -> Ops:
    """Returns a simple Ops object with a few commands."""
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 10.0, 0.0)
    ops.move_to(5.0, 5.0, 0.0)
    ops.line_to(15.0, 15.0, 0.0)
    return ops


@pytest.fixture
def sample_texture_data() -> np.ndarray:
    """Returns a simple 2x2 numpy array of power values."""
    return np.array([[0, 128], [255, 64]], dtype=np.uint8)


@pytest.fixture
def hybrid_artifact(
    sample_ops: Ops, sample_texture_data: np.ndarray
) -> HybridRasterArtifact:
    """Returns a complete HybridRasterArtifact instance with sample data."""
    return HybridRasterArtifact(
        ops=sample_ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        power_texture_data=sample_texture_data,
        dimensions_mm=(10.0, 10.0),
        position_mm=(5.0, 5.0),
        source_dimensions=(2, 2),
        generation_size=(10.0, 10.0),
    )


def test_hybrid_raster_artifact_creation(
    hybrid_artifact: HybridRasterArtifact,
):
    """Test that a HybridRasterArtifact can be created with valid params."""
    assert isinstance(hybrid_artifact.ops, Ops)
    assert isinstance(hybrid_artifact.power_texture_data, np.ndarray)
    assert hybrid_artifact.is_scalable is False
    assert (
        hybrid_artifact.source_coordinate_system
        == CoordinateSystem.PIXEL_SPACE
    )
    assert hybrid_artifact.dimensions_mm == (10.0, 10.0)
    assert hybrid_artifact.position_mm == (5.0, 5.0)
    assert len(hybrid_artifact.ops.commands) == 4
    assert hybrid_artifact.power_texture_data.shape == (2, 2)
    assert hybrid_artifact.source_dimensions == (2, 2)
    assert hybrid_artifact.generation_size == (10.0, 10.0)


def test_hybrid_raster_artifact_to_dict(
    hybrid_artifact: HybridRasterArtifact,
):
    """Test that a HybridRasterArtifact can be serialized to a dict."""
    data = hybrid_artifact.to_dict()

    assert isinstance(data, dict)
    assert data["type"] == "hybrid_raster"
    assert data["is_scalable"] is False
    assert data["source_coordinate_system"] == "PIXEL_SPACE"
    assert data["dimensions_mm"] == (10.0, 10.0)
    assert data["position_mm"] == (5.0, 5.0)
    assert data["source_dimensions"] == (2, 2)
    assert data["generation_size"] == (10.0, 10.0)
    assert isinstance(data["ops"], dict)
    assert isinstance(data["power_texture_data"], list)
    assert len(data["power_texture_data"]) == 2
    assert len(data["power_texture_data"][0]) == 2


def test_hybrid_raster_artifact_from_dict(
    hybrid_artifact: HybridRasterArtifact,
):
    """Test that a HybridRasterArtifact can be deserialized from a dict."""
    data = hybrid_artifact.to_dict()
    restored_artifact = HybridRasterArtifact.from_dict(data)

    assert restored_artifact.dimensions_mm == hybrid_artifact.dimensions_mm
    assert restored_artifact.position_mm == hybrid_artifact.position_mm
    assert restored_artifact.is_scalable == hybrid_artifact.is_scalable
    assert (
        restored_artifact.source_coordinate_system
        == hybrid_artifact.source_coordinate_system
    )
    assert (
        restored_artifact.source_dimensions
        == hybrid_artifact.source_dimensions
    )
    assert restored_artifact.generation_size == hybrid_artifact.generation_size
    assert np.array_equal(
        restored_artifact.power_texture_data,
        hybrid_artifact.power_texture_data,
    )
    # Could also compare ops commands one-by-one if needed
    assert len(restored_artifact.ops.commands) == len(
        hybrid_artifact.ops.commands
    )


def test_hybrid_raster_artifact_serialization_roundtrip(
    hybrid_artifact: HybridRasterArtifact,
):
    """
    Test that a HybridRasterArtifact can survive a serialization roundtrip.
    """
    data = hybrid_artifact.to_dict()
    restored_artifact = HybridRasterArtifact.from_dict(data)

    assert restored_artifact.dimensions_mm == hybrid_artifact.dimensions_mm
    assert restored_artifact.position_mm == hybrid_artifact.position_mm
    assert restored_artifact.is_scalable == hybrid_artifact.is_scalable
    assert (
        restored_artifact.source_coordinate_system
        == hybrid_artifact.source_coordinate_system
    )
    assert (
        restored_artifact.source_dimensions
        == hybrid_artifact.source_dimensions
    )
    assert restored_artifact.generation_size == hybrid_artifact.generation_size
    assert np.array_equal(
        restored_artifact.power_texture_data,
        hybrid_artifact.power_texture_data,
    )


def test_hybrid_raster_artifact_with_optional_fields_as_none(
    sample_ops: Ops, sample_texture_data: np.ndarray
):
    """
    Tests that serialization works correctly when optional fields are None.
    """
    artifact = HybridRasterArtifact(
        ops=sample_ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        power_texture_data=sample_texture_data,
        dimensions_mm=(10.0, 10.0),
        position_mm=(5.0, 5.0),
        source_dimensions=None,  # Explicitly set to None
        generation_size=None,  # Explicitly set to None
    )

    data = artifact.to_dict()
    restored_artifact = HybridRasterArtifact.from_dict(data)

    assert restored_artifact.source_dimensions is None
    assert restored_artifact.generation_size is None
    assert np.array_equal(
        restored_artifact.power_texture_data, artifact.power_texture_data
    )

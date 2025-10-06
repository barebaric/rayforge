import pytest
import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline.producer.base import (
    PipelineArtifact,
    HybridRasterArtifact,
    deserialize_artifact,
    CoordinateSystem,
)


@pytest.fixture
def sample_pipeline_artifact() -> PipelineArtifact:
    """Returns a sample PipelineArtifact."""
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 10.0, 0.0)

    return PipelineArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(100.0, 100.0),
        generation_size=(100.0, 100.0),
    )


@pytest.fixture
def sample_hybrid_artifact() -> HybridRasterArtifact:
    """Returns a sample HybridRasterArtifact."""
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 10.0, 0.0)

    texture_data = np.array([[0, 128], [255, 64]], dtype=np.uint8)

    return HybridRasterArtifact(
        ops=ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        power_texture_data=texture_data,
        dimensions_mm=(10.0, 10.0),
        position_mm=(5.0, 5.0),
        source_dimensions=(2, 2),
        generation_size=(10.0, 10.0),
    )


def test_deserialize_pipeline_artifact(
    sample_pipeline_artifact: PipelineArtifact,
):
    """Test that a PipelineArtifact can be deserialized correctly."""
    data = sample_pipeline_artifact.to_dict()
    artifact = deserialize_artifact(data)

    assert isinstance(artifact, PipelineArtifact)
    # Ensure it's not the hybrid type
    assert not isinstance(artifact, HybridRasterArtifact)
    assert artifact.is_scalable == sample_pipeline_artifact.is_scalable
    assert (
        artifact.source_coordinate_system
        == sample_pipeline_artifact.source_coordinate_system
    )
    assert len(artifact.ops.commands) == len(
        sample_pipeline_artifact.ops.commands
    )


def test_deserialize_hybrid_raster_artifact(
    sample_hybrid_artifact: HybridRasterArtifact,
):
    """Test that a HybridRasterArtifact can be deserialized correctly."""
    data = sample_hybrid_artifact.to_dict()
    artifact = deserialize_artifact(data)

    assert isinstance(artifact, HybridRasterArtifact)
    assert artifact.dimensions_mm == sample_hybrid_artifact.dimensions_mm
    assert artifact.position_mm == sample_hybrid_artifact.position_mm
    assert len(artifact.ops.commands) == len(
        sample_hybrid_artifact.ops.commands
    )
    assert np.array_equal(
        artifact.power_texture_data,
        sample_hybrid_artifact.power_texture_data,
    )


def test_deserialize_without_type_field_falls_back_to_pipeline_artifact():
    """
    Test that deserialization falls back correctly when 'type' is missing.
    """
    data = {
        "ops": {"commands": [], "last_move_to": (0.0, 0.0, 0.0)},
        "is_scalable": True,
        "source_coordinate_system": "MILLIMETER_SPACE",
        "source_dimensions": (100.0, 100.0),
        "generation_size": (100.0, 100.0),
    }

    artifact = deserialize_artifact(data)
    assert isinstance(artifact, PipelineArtifact)


def test_deserialize_with_unknown_type_falls_back_to_pipeline_artifact(
    sample_pipeline_artifact: PipelineArtifact,
):
    """Test that unknown types fall back for backward compatibility."""
    data = sample_pipeline_artifact.to_dict()
    data["type"] = "some_future_unknown_type"

    artifact = deserialize_artifact(data)
    assert isinstance(artifact, PipelineArtifact)


def test_deserialize_with_none_type_falls_back_to_pipeline_artifact(
    sample_pipeline_artifact: PipelineArtifact,
):
    """Test that a None 'type' falls back for backward compatibility."""
    data = sample_pipeline_artifact.to_dict()
    data["type"] = None

    artifact = deserialize_artifact(data)
    assert isinstance(artifact, PipelineArtifact)

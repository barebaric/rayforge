from typing import cast
import numpy as np
from multiprocessing import shared_memory
import pytest
from rayforge.context import get_context
from rayforge.pipeline.artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifact,
    WorkPieceViewArtifactHandle,
)
from rayforge.pipeline.artifact import create_handle_from_dict


@pytest.fixture
def handles_to_release():
    handles = []
    yield handles
    for handle in handles:
        get_context().artifact_store.release(handle)


def test_render_context_serialization():
    """Tests that RenderContext can be serialized and deserialized."""
    original_context = RenderContext(
        pixels_per_mm=(10.5, 10.5),
        show_travel_moves=True,
        margin_px=5,
        color_set_dict={"cut": ["#ff00ff", 1.0]},
    )
    context_dict = original_context.to_dict()
    reconstructed_context = RenderContext.from_dict(context_dict)
    assert original_context == reconstructed_context


def test_handle_serialization():
    """
    Tests that WorkPieceViewArtifactHandle can be serialized and
    deserialized.
    """
    original_handle = WorkPieceViewArtifactHandle(
        shm_name="test_shm_123",
        handle_class_name="WorkPieceViewArtifactHandle",
        artifact_type_name="WorkPieceViewArtifact",
        bbox_mm=(10.0, 20.0, 100.0, 50.0),
        workpiece_size_mm=(100.0, 50.0),
        array_metadata={
            "bitmap_data": {
                "dtype": "uint8",
                "shape": (100, 200, 4),
                "offset": 0,
            }
        },
        generation_id=1,
    )
    handle_dict = original_handle.to_dict()
    reconstructed_handle = create_handle_from_dict(handle_dict)

    assert isinstance(reconstructed_handle, WorkPieceViewArtifactHandle)
    assert original_handle == reconstructed_handle


def test_artifact_store_lifecycle(handles_to_release):
    """
    Tests the full put -> get -> release lifecycle with a
    WorkPieceViewArtifact.
    """
    bitmap_data = np.arange(10 * 20 * 4, dtype=np.uint8).reshape((20, 10, 4))
    bbox_mm = (5.0, 5.0, 50.0, 100.0)
    original_artifact = WorkPieceViewArtifact(
        bitmap_data=bitmap_data,
        bbox_mm=bbox_mm,
        workpiece_size_mm=(50.0, 100.0),
        generation_id=1,
    )

    handle_base = get_context().artifact_store.put(original_artifact)
    handles_to_release.append(handle_base)

    handle = cast(WorkPieceViewArtifactHandle, handle_base)

    assert isinstance(handle, WorkPieceViewArtifactHandle)
    assert handle.bbox_mm == bbox_mm

    retrieved_base = get_context().artifact_store.get(handle)

    retrieved_artifact = cast(WorkPieceViewArtifact, retrieved_base)

    assert isinstance(retrieved_artifact, WorkPieceViewArtifact)
    assert retrieved_artifact.bbox_mm == bbox_mm
    np.testing.assert_array_equal(retrieved_artifact.bitmap_data, bitmap_data)

    get_context().artifact_store.release(handle)

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.shm_name)

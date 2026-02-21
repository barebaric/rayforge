from typing import cast
import json
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import pytest
from rayforge.context import get_context
from rayforge.core.ops import Ops
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.artifact import create_handle_from_dict
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.artifact.workpiece import (
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
)
from rayforge.pipeline.artifact.job import JobArtifact, JobArtifactHandle


@pytest.fixture
def handles_to_release():
    handles = []
    yield handles
    for handle in handles:
        get_context().artifact_store.release(handle)


def _create_sample_vertex_artifact() -> WorkPieceArtifact:
    """Helper to generate a consistent vertex artifact for tests."""
    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.line_to(10, 0, 0)
    ops.arc_to(0, 10, i=-10, j=0, clockwise=False, z=0)

    return WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(100, 100),
        generation_size=(50, 50),
        generation_id=1,
    )


def _create_sample_hybrid_artifact() -> WorkPieceArtifact:
    """Helper to generate a consistent hybrid artifact for tests."""
    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.scan_to(10, 0, 0, power_values=bytearray(range(256)))
    return WorkPieceArtifact(
        ops=ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        source_dimensions=(200, 200),
        generation_size=(50, 50),
        generation_id=1,
    )


def _create_sample_final_job_artifact() -> JobArtifact:
    """Helper to generate a final job artifact for tests."""
    machine_code_bytes = np.frombuffer(b"G1 X10 Y20", dtype=np.uint8)
    op_map = {
        "op_to_machine_code": {0: [0, 1]},
        "machine_code_to_op": {0: 0, 1: 0},
    }
    op_map_bytes = np.frombuffer(
        json.dumps(op_map).encode("utf-8"), dtype=np.uint8
    )
    return JobArtifact(
        ops=Ops(),
        distance=15.0,
        machine_code_bytes=machine_code_bytes,
        op_map_bytes=op_map_bytes,
        generation_id=1,
    )


def test_put_get_release_vertex_artifact(handles_to_release):
    """
    Tests the full put -> get -> release lifecycle with a
    vertex-based Artifact.
    """
    original_artifact = _create_sample_vertex_artifact()

    handle = get_context().artifact_store.put(original_artifact)
    handles_to_release.append(handle)
    assert isinstance(handle, WorkPieceArtifactHandle)

    retrieved_artifact = get_context().artifact_store.get(handle)

    assert isinstance(retrieved_artifact, WorkPieceArtifact)
    assert retrieved_artifact.artifact_type == "WorkPieceArtifact"
    assert len(original_artifact.ops.commands) == len(
        retrieved_artifact.ops.commands
    )
    assert (
        original_artifact.generation_size == retrieved_artifact.generation_size
    )

    get_context().artifact_store.release(handle)

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.shm_name)


def test_put_get_release_hybrid_artifact(handles_to_release):
    """
    Tests the full put -> get -> release lifecycle with a
    Hybrid-like Artifact.
    """
    original_artifact = _create_sample_hybrid_artifact()

    handle = get_context().artifact_store.put(original_artifact)
    handles_to_release.append(handle)
    assert isinstance(handle, WorkPieceArtifactHandle)

    retrieved_artifact = get_context().artifact_store.get(handle)

    assert isinstance(retrieved_artifact, WorkPieceArtifact)
    assert retrieved_artifact.artifact_type == "WorkPieceArtifact"
    assert len(original_artifact.ops.commands) == len(
        retrieved_artifact.ops.commands
    )

    get_context().artifact_store.release(handle)

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.shm_name)


def test_put_get_release_final_job_artifact(handles_to_release):
    """
    Tests the full put -> get -> release lifecycle with a
    final_job Artifact.
    """
    original_artifact = _create_sample_final_job_artifact()

    handle = get_context().artifact_store.put(original_artifact)
    handles_to_release.append(handle)
    assert isinstance(handle, JobArtifactHandle)

    retrieved_artifact = get_context().artifact_store.get(handle)

    assert isinstance(retrieved_artifact, JobArtifact)
    assert retrieved_artifact.artifact_type == "JobArtifact"
    assert retrieved_artifact.machine_code_bytes is not None
    assert retrieved_artifact.op_map_bytes is not None

    assert retrieved_artifact.machine_code_bytes is not None
    gcode_str = retrieved_artifact.machine_code_bytes.tobytes().decode("utf-8")
    op_map = retrieved_artifact.op_map
    assert op_map is not None

    assert gcode_str == "G1 X10 Y20"
    if op_map:
        assert op_map.op_to_machine_code == {0: [0, 1]}
        assert op_map.machine_code_to_op == {0: 0, 1: 0}

    get_context().artifact_store.release(handle)

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.shm_name)


def test_adopt_and_release():
    """
    Tests that `adopt` correctly takes ownership of a shared memory
    block, simulating a handover from a worker process.
    """
    original_artifact = _create_sample_vertex_artifact()
    main_store = get_context().artifact_store

    worker_store = ArtifactStore()
    handle = worker_store.put(original_artifact)

    assert handle.shm_name not in main_store._managed_shms

    main_store.adopt(handle)
    assert handle.shm_name in main_store._managed_shms

    worker_shm = worker_store._managed_shms.pop(handle.shm_name)
    worker_shm.close()

    try:
        retrieved = cast(WorkPieceArtifact, main_store.get(handle))
        assert original_artifact.generation_size == retrieved.generation_size
    except FileNotFoundError:
        pytest.fail("Shared memory block was not found after adoption.")

    main_store.release(handle)

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.shm_name)

    assert handle.shm_name not in main_store._managed_shms


def test_forget_and_adopt_across_processes():
    """
    Tests the adoption handshake between two processes:
    1. Process A (worker) creates an artifact
    2. Process B (main) adopts it
    3. Process A forgets it (closes handle without unlinking)
    4. Process B should still be able to read data
    5. Process B releases it, and data is gone
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    worker = ctx.Process(target=_worker_create_and_forget, args=(queue,))
    worker.start()

    store = ArtifactStore()

    handle_dict = queue.get()
    handle = create_handle_from_dict(handle_dict)

    store.adopt(handle)
    assert handle.shm_name in store._managed_shms

    queue.put("adopted")

    result = queue.get()
    assert result == "forgotten"

    retrieved = cast(WorkPieceArtifact, store.get(handle))
    assert (50, 50) == retrieved.generation_size

    store.release(handle)

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.shm_name)

    worker.join(timeout=5)
    assert not worker.is_alive()
    assert worker.exitcode == 0


def _worker_create_and_forget(queue: mp.Queue):
    """
    Process A: Creates an artifact and forgets it after
    signaling Process B.
    """
    store = ArtifactStore()
    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.line_to(10, 0, 0)
    ops.arc_to(0, 10, i=-10, j=0, clockwise=False, z=0)

    artifact = WorkPieceArtifact(
        ops=ops,
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        source_dimensions=(100, 100),
        generation_size=(50, 50),
        generation_id=1,
    )
    handle = store.put(artifact)

    queue.put(handle.to_dict())

    queue.get()

    store.forget(handle)

    queue.put("forgotten")

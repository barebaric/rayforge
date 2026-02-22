"""
Tests for the simplified ArtifactManager cache functionality.

The ArtifactManager is now a pure cache - it only stores handles.
State tracking is handled by DAG scheduler via ArtifactNode.
"""

from unittest.mock import Mock
import pytest

from rayforge.pipeline.artifact import (
    ArtifactKey,
    ArtifactManager,
    WorkPieceArtifactHandle,
)
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.artifact.manager import make_composite_key


def create_mock_handle(handle_class, name: str) -> Mock:
    """Creates a mock handle that behaves like a real handle for tests."""
    handle = Mock(spec=handle_class)
    handle.shm_name = f"shm_{name}"
    handle.refcount = 1
    handle.holders = []
    return handle


@pytest.fixture
def manager_with_mock_store():
    mock_store = Mock(spec=ArtifactStore)
    mock_store._handles = {}
    manager = ArtifactManager(mock_store)
    return manager, mock_store


def test_cache_handle_stores_handle(manager_with_mock_store):
    """Test cache_handle stores a handle (no retain when refcount=0)."""
    manager, mock_store = manager_with_mock_store
    key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
    handle = create_mock_handle(WorkPieceArtifactHandle, "handle")

    manager.declare_generation({key}, generation_id=1)
    manager.cache_handle(key, handle, generation_id=1)

    composite_key = make_composite_key(key, 1)
    entry = manager.get_ledger_entry(composite_key)
    assert entry is not None
    assert entry.handle is handle
    assert entry.generation_id == 1
    mock_store.retain.assert_not_called()


def test_cache_handle_replaces_old_handle(manager_with_mock_store):
    """Test cache_handle releases old handle and stores new one."""
    manager, mock_store = manager_with_mock_store
    key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
    old_handle = create_mock_handle(WorkPieceArtifactHandle, "old_handle")
    new_handle = create_mock_handle(WorkPieceArtifactHandle, "new_handle")

    manager.declare_generation({key}, generation_id=1)
    manager.cache_handle(key, old_handle, generation_id=1)
    manager.cache_handle(key, new_handle, generation_id=1)

    composite_key = make_composite_key(key, 1)
    entry = manager.get_ledger_entry(composite_key)
    assert entry is not None
    assert entry.handle is new_handle
    mock_store.retain.assert_not_called()
    mock_store.release.assert_called_once_with(old_handle)


def test_get_workpiece_handle_returns_cached_handle(manager_with_mock_store):
    """Test get_workpiece_handle returns cached handle."""
    manager, _ = manager_with_mock_store
    key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
    handle = create_mock_handle(WorkPieceArtifactHandle, "handle")

    manager.declare_generation({key}, generation_id=1)
    manager.cache_handle(key, handle, generation_id=1)

    result = manager.get_workpiece_handle(key, 1)
    assert result is handle


def test_get_workpiece_handle_returns_none_if_no_handle(
    manager_with_mock_store,
):
    """Test get_workpiece_handle returns None if no handle cached."""
    manager, _ = manager_with_mock_store
    key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")

    result = manager.get_workpiece_handle(key, 1)
    assert result is None


def test_invalidate_for_workpiece_removes_cached_handles(
    manager_with_mock_store,
):
    """Test invalidate_for_workpiece removes cached handles."""
    manager, _ = manager_with_mock_store
    key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
    handle = create_mock_handle(WorkPieceArtifactHandle, "handle")

    manager.declare_generation({key}, generation_id=1)
    manager.cache_handle(key, handle, generation_id=1)
    manager.invalidate_for_workpiece(key)

    result = manager.get_workpiece_handle(key, 1)
    assert result is None


def test_shutdown_releases_all_handles(manager_with_mock_store):
    """Test shutdown releases all cached handles."""
    manager, mock_store = manager_with_mock_store
    key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
    handle = create_mock_handle(WorkPieceArtifactHandle, "handle")

    manager.declare_generation({key}, generation_id=1)
    manager.cache_handle(key, handle, generation_id=1)
    manager.shutdown()

    mock_store.release.assert_called()


@pytest.fixture
def dependency_manager():
    mock_store = Mock(spec=ArtifactStore)
    manager = ArtifactManager(mock_store)
    return manager


def test_register_dependency(dependency_manager):
    """Test that register_dependency correctly establishes deps."""
    manager = dependency_manager
    workpiece_key = ArtifactKey.for_workpiece(
        "00000000-0000-4000-8000-000000000001"
    )
    step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")
    job_key = ArtifactKey.for_job()

    manager.register_dependency(workpiece_key, step_key)
    manager.register_dependency(step_key, job_key)

    dependents = manager.get_dependents(workpiece_key)
    assert dependents == [step_key]

    dependents = manager.get_dependents(step_key)
    assert dependents == [job_key]

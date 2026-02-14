"""
Tests for the simplified ArtifactManager cache functionality.

The ArtifactManager is now a pure cache - it only stores handles.
State tracking is handled by DAG scheduler via ArtifactNode.
"""

import unittest
from unittest.mock import Mock

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
    return handle


class TestArtifactManagerCache(unittest.TestCase):
    """Test suite for ArtifactManager as a pure cache."""

    def setUp(self):
        """Set up a fresh manager and mock store."""
        self.mock_store = Mock(spec=ArtifactStore)
        self.manager = ArtifactManager(self.mock_store)

    def test_cache_handle_stores_handle(self):
        """Test cache_handle stores a handle."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        handle = create_mock_handle(WorkPieceArtifactHandle, "handle")

        self.manager.cache_handle(key, handle, generation_id=1)

        composite_key = make_composite_key(key, 1)
        entry = self.manager.get_ledger_entry(composite_key)
        assert entry is not None
        self.assertIs(entry.handle, handle)
        self.assertEqual(entry.generation_id, 1)
        self.mock_store.retain.assert_called_once_with(handle)

    def test_cache_handle_replaces_old_handle(self):
        """Test cache_handle releases old handle and stores new one."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        old_handle = create_mock_handle(WorkPieceArtifactHandle, "old_handle")
        new_handle = create_mock_handle(WorkPieceArtifactHandle, "new_handle")

        self.manager.cache_handle(key, old_handle, generation_id=1)
        self.manager.cache_handle(key, new_handle, generation_id=1)

        composite_key = make_composite_key(key, 1)
        entry = self.manager.get_ledger_entry(composite_key)
        assert entry is not None
        self.assertIs(entry.handle, new_handle)
        self.mock_store.retain.assert_called_with(new_handle)
        self.mock_store.release.assert_called_once_with(old_handle)

    def test_get_workpiece_handle_returns_cached_handle(self):
        """Test get_workpiece_handle returns cached handle."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        handle = create_mock_handle(WorkPieceArtifactHandle, "handle")

        self.manager.cache_handle(key, handle, generation_id=1)

        result = self.manager.get_workpiece_handle(key, 1)
        self.assertIs(result, handle)

    def test_get_workpiece_handle_returns_none_if_no_handle(self):
        """Test get_workpiece_handle returns None if no handle cached."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")

        result = self.manager.get_workpiece_handle(key, 1)
        self.assertIsNone(result)

    def test_invalidate_for_workpiece_removes_cached_handles(self):
        """Test invalidate_for_workpiece removes cached handles."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        handle = create_mock_handle(WorkPieceArtifactHandle, "handle")

        self.manager.cache_handle(key, handle, generation_id=1)
        self.manager.invalidate_for_workpiece(key)

        result = self.manager.get_workpiece_handle(key, 1)
        self.assertIsNone(result)

    def test_shutdown_releases_all_handles(self):
        """Test shutdown releases all cached handles."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        handle = create_mock_handle(WorkPieceArtifactHandle, "handle")

        self.manager.cache_handle(key, handle, generation_id=1)
        self.manager.shutdown()

        self.mock_store.release.assert_called()


class TestDependencyTracking(unittest.TestCase):
    """Test dependency tracking functionality."""

    def setUp(self):
        """Set up a fresh manager and mock store."""
        self.mock_store = Mock(spec=ArtifactStore)
        self.manager = ArtifactManager(self.mock_store)

    def test_register_dependency(self):
        """Test that register_dependency correctly establishes deps."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")
        job_key = ArtifactKey.for_job()

        self.manager.register_dependency(workpiece_key, step_key)
        self.manager.register_dependency(step_key, job_key)

        dependents = self.manager.get_dependents(workpiece_key)
        self.assertEqual(dependents, [step_key])

        dependents = self.manager.get_dependents(step_key)
        self.assertEqual(dependents, [job_key])


if __name__ == "__main__":
    unittest.main()

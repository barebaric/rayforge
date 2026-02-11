from typing import Optional
import unittest
from unittest.mock import Mock

from rayforge.pipeline.artifact import (
    ArtifactKey,
    ArtifactManager,
    WorkPieceArtifactHandle,
)
from rayforge.pipeline.artifact.lifecycle import (
    ArtifactLifecycle,
    LedgerEntry,
)
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.artifact.manager import make_composite_key


def create_mock_handle(handle_class, name: str) -> Mock:
    """Creates a mock handle that behaves like a real handle for tests."""
    handle = Mock(spec=handle_class)
    handle.shm_name = f"shm_{name}"
    return handle


class TestArtifactManagerLedger(unittest.TestCase):
    """Test suite for ArtifactManager Ledger functionality."""

    def setUp(self):
        """Set up a fresh manager and mock store."""
        self.mock_store = Mock(spec=ArtifactStore)
        self.manager = ArtifactManager(self.mock_store)

    def _create_ledger_entry(
        self,
        state: ArtifactLifecycle,
        handle=None,
        generation_id: int = 0,
        error: Optional[str] = None,
    ) -> LedgerEntry:
        """Helper to create a LedgerEntry."""
        return LedgerEntry(
            state=state,
            handle=handle,
            generation_id=generation_id,
            error=error,
        )


class TestStateTransitions(TestArtifactManagerLedger):
    """Test state transition methods."""

    def test_register_intent_creates_entry(self):
        """Test register_intent creates an entry with INITIAL state."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        generation_id = 0

        self.manager.register_intent(key, generation_id)

        composite_key = make_composite_key(key, generation_id)
        entry = self.manager._get_ledger_entry(composite_key)
        assert entry is not None
        self.assertEqual(entry.state, ArtifactLifecycle.INITIAL)
        self.assertEqual(entry.generation_id, generation_id)

    def test_register_intent_duplicate_raises_assertion(self):
        """Test register_intent raises AssertionError for duplicate entry."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        generation_id = 0

        self.manager.register_intent(key, generation_id)

        with self.assertRaises(AssertionError) as cm:
            self.manager.register_intent(key, generation_id)

        self.assertIn("already exists", str(cm.exception))

    def test_mark_processing_from_stale_succeeds_temporarily(self):
        """Test mark_processing from STALE state (temporary allowance)."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        entry = self._create_ledger_entry(ArtifactLifecycle.STALE)
        composite_key = make_composite_key(key, 0)
        self.manager._set_ledger_entry(composite_key, entry)

        self.manager.mark_processing(key, generation_id=0)
        self.assertEqual(entry.state, ArtifactLifecycle.PROCESSING)

    def test_mark_processing_from_initial(self):
        """Test mark_processing works correctly from INITIAL to PROCESSING."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        entry = self._create_ledger_entry(ArtifactLifecycle.INITIAL)
        composite_key = make_composite_key(key, 0)
        self.manager._set_ledger_entry(composite_key, entry)

        self.manager.mark_processing(key, generation_id=0)

        self.assertEqual(entry.state, ArtifactLifecycle.PROCESSING)
        self.assertEqual(entry.generation_id, 0)
        self.assertIsNone(entry.error)

    def test_mark_processing_from_done_raises_assertion(self):
        """Test mark_processing raises AssertionError from DONE state."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        entry = self._create_ledger_entry(ArtifactLifecycle.DONE)
        composite_key = make_composite_key(key, 0)
        self.manager._set_ledger_entry(composite_key, entry)

        with self.assertRaises(AssertionError) as cm:
            self.manager.mark_processing(key, generation_id=0)

        self.assertIn(
            "must be INITIAL or STALE",
            str(cm.exception),
        )

    def test_mark_processing_from_error_raises_assertion(self):
        """Test mark_processing raises AssertionError from ERROR state."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        entry = self._create_ledger_entry(
            ArtifactLifecycle.ERROR, error="test error"
        )
        composite_key = make_composite_key(key, 0)
        self.manager._set_ledger_entry(composite_key, entry)

        with self.assertRaises(AssertionError) as cm:
            self.manager.mark_processing(key, generation_id=0)

        self.assertIn(
            "must be INITIAL or STALE",
            str(cm.exception),
        )

    def test_mark_processing_from_processing_raises_assertion(self):
        """Test mark_processing raises AssertionError from PROCESSING state."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        entry = self._create_ledger_entry(
            ArtifactLifecycle.PROCESSING, generation_id=0
        )
        composite_key = make_composite_key(key, 0)
        self.manager._set_ledger_entry(composite_key, entry)

        with self.assertRaises(AssertionError) as cm:
            self.manager.mark_processing(key, generation_id=0)

        self.assertIn(
            "must be INITIAL or STALE",
            str(cm.exception),
        )

    def test_mark_processing_nonexistent_key_raises_assertion(self):
        """Test mark_processing raises AssertionError for nonexistent key."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")

        with self.assertRaises(AssertionError) as cm:
            self.manager.mark_processing(key, generation_id=1)

        self.assertIn("not found in ledger", str(cm.exception))

    def test_commit_from_processing(self):
        """Test commit works correctly from PROCESSING to DONE."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        old_handle = create_mock_handle(WorkPieceArtifactHandle, "old_handle")
        new_handle = create_mock_handle(WorkPieceArtifactHandle, "new_handle")
        entry = self._create_ledger_entry(
            ArtifactLifecycle.PROCESSING,
            handle=old_handle,
            generation_id=1,
        )
        composite_key = make_composite_key(key, 1)
        self.manager._set_ledger_entry(composite_key, entry)

        self.manager.commit_artifact(key, new_handle, generation_id=1)

        self.assertEqual(entry.state, ArtifactLifecycle.DONE)
        self.assertIs(entry.handle, new_handle)
        self.assertEqual(entry.generation_id, 1)
        self.assertIsNone(entry.error)
        self.mock_store.adopt.assert_called_once_with(new_handle)
        self.mock_store.release.assert_called_once_with(old_handle)

    def test_commit_from_initial(self):
        """Test commit works correctly from INITIAL to DONE."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        new_handle = create_mock_handle(WorkPieceArtifactHandle, "new_handle")
        entry = self._create_ledger_entry(
            ArtifactLifecycle.INITIAL,
            handle=None,
            generation_id=1,
        )
        composite_key = make_composite_key(key, 1)
        self.manager._set_ledger_entry(composite_key, entry)

        self.manager.commit_artifact(key, new_handle, generation_id=1)

        self.assertEqual(entry.state, ArtifactLifecycle.DONE)
        self.assertIs(entry.handle, new_handle)
        self.assertEqual(entry.generation_id, 1)
        self.assertIsNone(entry.error)
        self.mock_store.adopt.assert_called_once_with(new_handle)

    def test_commit_from_stale_raises_assertion(self):
        """
        Test commit raises AssertionError from STALE state.
        """
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        handle = create_mock_handle(WorkPieceArtifactHandle, "handle")
        entry = self._create_ledger_entry(ArtifactLifecycle.STALE)
        composite_key = make_composite_key(key, 0)
        self.manager._set_ledger_entry(composite_key, entry)

        with self.assertRaises(AssertionError) as cm:
            self.manager.commit_artifact(key, handle, generation_id=0)

        self.assertIn("must be INITIAL or PROCESSING", str(cm.exception))

    def test_commit_generation_id_mismatch_creates_new_entry(self):
        """Test commit creates a new entry on generation_id mismatch."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        handle = create_mock_handle(WorkPieceArtifactHandle, "handle")
        old_handle = create_mock_handle(WorkPieceArtifactHandle, "old_handle")
        entry_g0 = self._create_ledger_entry(
            ArtifactLifecycle.PROCESSING, handle=old_handle, generation_id=0
        )
        composite_key_g0 = make_composite_key(key, 0)
        self.manager._set_ledger_entry(composite_key_g0, entry_g0)

        # This should not raise an error, but create a new entry for gen 1
        self.manager.commit_artifact(key, handle, generation_id=1)

        composite_key_g1 = make_composite_key(key, 1)
        entry_g1 = self.manager._get_ledger_entry(composite_key_g1)
        assert entry_g1 is not None
        self.assertEqual(entry_g1.state, ArtifactLifecycle.DONE)
        self.assertEqual(entry_g1.generation_id, 1)

    def test_commit_nonexistent_key_creates_new_entry(self):
        """Test commit creates a new entry for a nonexistent key."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        handle = create_mock_handle(WorkPieceArtifactHandle, "handle")

        # This should not raise an error, but create the entry
        self.manager.commit_artifact(key, handle, generation_id=1)

        composite_key = make_composite_key(key, 1)
        entry = self.manager._get_ledger_entry(composite_key)
        assert entry is not None
        self.assertEqual(entry.state, ArtifactLifecycle.DONE)

    def test_commit_without_old_handle(self):
        """Test commit works when entry has no previous handle."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        new_handle = create_mock_handle(WorkPieceArtifactHandle, "new_handle")
        entry = self._create_ledger_entry(
            ArtifactLifecycle.PROCESSING,
            handle=None,
            generation_id=1,
        )
        composite_key = make_composite_key(key, 1)
        self.manager._set_ledger_entry(composite_key, entry)

        self.manager.commit_artifact(key, new_handle, generation_id=1)

        self.assertEqual(entry.state, ArtifactLifecycle.DONE)
        self.assertIs(entry.handle, new_handle)
        self.mock_store.adopt.assert_called_once_with(new_handle)
        self.mock_store.release.assert_not_called()

    def test_fail_generation_from_processing(self):
        """Test fail_generation works correctly from PROCESSING to ERROR."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        entry = self._create_ledger_entry(
            ArtifactLifecycle.PROCESSING, generation_id=1
        )
        composite_key = make_composite_key(key, 1)
        self.manager._set_ledger_entry(composite_key, entry)

        self.manager.fail_generation(
            key, "Test error message", generation_id=1
        )

        self.assertEqual(entry.state, ArtifactLifecycle.ERROR)
        self.assertEqual(entry.error, "Test error message")
        self.assertEqual(entry.generation_id, 1)

    def test_fail_generation_from_done_raises_assertion(self):
        """Test fail_generation raises AssertionError when not PROCESSING."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        entry = self._create_ledger_entry(ArtifactLifecycle.DONE)
        composite_key = make_composite_key(key, 0)
        self.manager._set_ledger_entry(composite_key, entry)

        with self.assertRaises(AssertionError) as cm:
            self.manager.fail_generation(key, "Test error", generation_id=0)

        self.assertIn("must be PROCESSING", str(cm.exception))

    def test_fail_generation_generation_id_mismatch_raises_assertion(self):
        """Test fail_generation raises AssertionError on gen_id mismatch."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        entry = self._create_ledger_entry(
            ArtifactLifecycle.PROCESSING, generation_id=0
        )
        composite_key = make_composite_key(key, 0)
        self.manager._set_ledger_entry(composite_key, entry)

        with self.assertRaises(AssertionError) as cm:
            self.manager.fail_generation(key, "Test error", generation_id=1)
        self.assertIn("not found in ledger", str(cm.exception))

    def test_fail_generation_nonexistent_key_raises_assertion(self):
        """Test fail_generation raises AssertionError for nonexistent key."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")

        with self.assertRaises(AssertionError) as cm:
            self.manager.fail_generation(key, "Test error", generation_id=1)

        self.assertIn("not found in ledger", str(cm.exception))


class TestDependencyGraphAndInvalidation(TestArtifactManagerLedger):
    """Test dependency graph and invalidation methods."""

    def test_register_dependency(self):
        """Test that _register_dependency correctly establishes deps."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")
        job_key = ArtifactKey.for_job()

        self.manager._register_dependency(workpiece_key, step_key)
        self.manager._register_dependency(step_key, job_key)

        deps = self.manager._get_dependencies(job_key, 0)
        self.assertEqual(deps, [(step_key, 0)])

        deps = self.manager._get_dependencies(step_key, 0)
        self.assertEqual(deps, [(workpiece_key, 0)])

    def test_register_dependency_idempotent(self):
        """Test that _register_dependency doesn't duplicate dependencies."""
        child_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        parent_key = ArtifactKey.for_step(
            "00000000-0000-4000-8000-000000000003"
        )

        self.manager._register_dependency(child_key, parent_key)
        self.manager._register_dependency(child_key, parent_key)

        deps = self.manager._get_dependencies(parent_key, 0)
        self.assertEqual(deps, [(child_key, 0)])
        self.assertEqual(len(deps), 1)

    def test_get_dependents_returns_parents(self):
        """Test that _get_dependents returns correct parents."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")
        job_key = ArtifactKey.for_job()

        self.manager._register_dependency(workpiece_key, step_key)
        self.manager._register_dependency(step_key, job_key)

        dependents = self.manager._get_dependents(workpiece_key)
        self.assertEqual(dependents, [step_key])

        dependents = self.manager._get_dependents(step_key)
        self.assertEqual(dependents, [job_key])

        dependents = self.manager._get_dependents(job_key)
        self.assertEqual(dependents, [])

    def test_get_dependencies_returns_children(self):
        """Test that _get_dependencies returns correct children."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")
        job_key = ArtifactKey.for_job()

        self.manager._register_dependency(workpiece_key, step_key)
        self.manager._register_dependency(step_key, job_key)

        deps = self.manager._get_dependencies(workpiece_key, 0)
        self.assertEqual(deps, [])

        deps = self.manager._get_dependencies(step_key, 0)
        self.assertEqual(deps, [(workpiece_key, 0)])

        deps = self.manager._get_dependencies(job_key, 0)
        self.assertEqual(deps, [(step_key, 0)])

    def test_invalidate_cascades_to_dependents(self):
        """Test invalidate() on Workpiece transitions Step and Job to STALE."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")
        job_key = ArtifactKey.for_job()

        wp_entry = self._create_ledger_entry(ArtifactLifecycle.DONE)
        step_entry = self._create_ledger_entry(ArtifactLifecycle.DONE)
        job_entry = self._create_ledger_entry(ArtifactLifecycle.DONE)

        wp_composite = make_composite_key(workpiece_key, 0)
        step_composite = make_composite_key(step_key, 0)
        job_composite = make_composite_key(job_key, 0)
        self.manager._set_ledger_entry(wp_composite, wp_entry)
        self.manager._set_ledger_entry(step_composite, step_entry)
        self.manager._set_ledger_entry(job_composite, job_entry)

        self.manager._register_dependency(workpiece_key, step_key)
        self.manager._register_dependency(step_key, job_key)

        self.manager.invalidate(workpiece_key)

        self.assertEqual(wp_entry.state, ArtifactLifecycle.STALE)
        self.assertEqual(step_entry.state, ArtifactLifecycle.STALE)
        self.assertEqual(job_entry.state, ArtifactLifecycle.STALE)

    def test_invalidate_with_no_dependents_is_noop(self):
        """Test that invalidating a key with no dependents is a no-op."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        entry = self._create_ledger_entry(ArtifactLifecycle.DONE)
        composite_key = make_composite_key(key, 0)
        self.manager._set_ledger_entry(composite_key, entry)

        self.manager.invalidate(key)

        self.assertEqual(entry.state, ArtifactLifecycle.STALE)

    def test_invalidate_nonexistent_key_is_noop(self):
        """Test that invalidating a nonexistent key does nothing."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")

        self.manager.invalidate(key)

        self.assertEqual(len(self.manager._ledger), 0)

    def test_invalidate_with_multiple_dependents(self):
        """Test that invalidate cascades to all dependents."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step1_key = ArtifactKey.for_step(
            "00000000-0000-4000-8000-000000000003"
        )
        step2_key = ArtifactKey.for_step(
            "00000000-0000-4000-8000-000000000004"
        )

        wp_entry = self._create_ledger_entry(ArtifactLifecycle.DONE)
        step1_entry = self._create_ledger_entry(ArtifactLifecycle.DONE)
        step2_entry = self._create_ledger_entry(ArtifactLifecycle.DONE)

        wp_composite = make_composite_key(workpiece_key, 0)
        step1_composite = make_composite_key(step1_key, 0)
        step2_composite = make_composite_key(step2_key, 0)
        self.manager._set_ledger_entry(wp_composite, wp_entry)
        self.manager._set_ledger_entry(step1_composite, step1_entry)
        self.manager._set_ledger_entry(step2_composite, step2_entry)

        self.manager._register_dependency(workpiece_key, step1_key)
        self.manager._register_dependency(workpiece_key, step2_key)

        self.manager.invalidate(workpiece_key)

        self.assertEqual(wp_entry.state, ArtifactLifecycle.STALE)
        self.assertEqual(step1_entry.state, ArtifactLifecycle.STALE)
        self.assertEqual(step2_entry.state, ArtifactLifecycle.STALE)


class TestQueryWorkForStage(TestArtifactManagerLedger):
    """Test query_work_for_stage method."""

    def test_query_work_returns_missing_keys_with_ready_deps(self):
        """Test query_work_for_stage returns INITIAL keys with DONE deps."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")

        wp_entry = self._create_ledger_entry(ArtifactLifecycle.DONE)
        step_entry = self._create_ledger_entry(ArtifactLifecycle.INITIAL)

        wp_composite = make_composite_key(workpiece_key, 0)
        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(wp_composite, wp_entry)
        self.manager._set_ledger_entry(step_composite, step_entry)

        self.manager._register_dependency(workpiece_key, step_key)

        work = self.manager.query_work_for_stage("step", 0)

        self.assertEqual(work, [step_key])

    def test_query_work_empty_when_dependency_not_ready(self):
        """Test query_work_for_stage returns empty when deps not DONE."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")

        wp_entry = self._create_ledger_entry(ArtifactLifecycle.PROCESSING)
        step_entry = self._create_ledger_entry(ArtifactLifecycle.STALE)

        wp_composite = make_composite_key(workpiece_key, 0)
        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(wp_composite, wp_entry)
        self.manager._set_ledger_entry(step_composite, step_entry)

        self.manager._register_dependency(workpiece_key, step_key)

        work = self.manager.query_work_for_stage("step", 0)

        self.assertEqual(work, [])

    def test_query_work_empty_when_dependency_stale(self):
        """Test query_work_for_stage returns empty when deps are STALE."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")

        wp_entry = self._create_ledger_entry(ArtifactLifecycle.STALE)
        step_entry = self._create_ledger_entry(ArtifactLifecycle.STALE)

        wp_composite = make_composite_key(workpiece_key, 0)
        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(wp_composite, wp_entry)
        self.manager._set_ledger_entry(step_composite, step_entry)

        self.manager._register_dependency(workpiece_key, step_key)

        work = self.manager.query_work_for_stage("step", 0)

        self.assertEqual(work, [])

    def test_query_work_filters_by_stage_type(self):
        """Test that query_work_for_stage filters by stage_type correctly."""
        wp1_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        wp2_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000002"
        )
        step1_entry = self._create_ledger_entry(ArtifactLifecycle.INITIAL)
        step2_entry = self._create_ledger_entry(ArtifactLifecycle.INITIAL)

        wp1_composite = make_composite_key(wp1_key, 0)
        wp2_composite = make_composite_key(wp2_key, 0)
        self.manager._set_ledger_entry(wp1_composite, step1_entry)
        self.manager._set_ledger_entry(wp2_composite, step2_entry)

        work = self.manager.query_work_for_stage("workpiece", 0)

        self.assertEqual(work, [wp1_key, wp2_key])

    def test_query_work_returns_multiple_keys(self):
        """Test query_work_for_stage returns multiple matching keys."""
        wp1_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        wp2_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000002"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")

        wp1_entry = self._create_ledger_entry(ArtifactLifecycle.DONE)
        wp2_entry = self._create_ledger_entry(ArtifactLifecycle.DONE)
        step_entry = self._create_ledger_entry(ArtifactLifecycle.INITIAL)

        wp1_composite = make_composite_key(wp1_key, 0)
        wp2_composite = make_composite_key(wp2_key, 0)
        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(wp1_composite, wp1_entry)
        self.manager._set_ledger_entry(wp2_composite, wp2_entry)
        self.manager._set_ledger_entry(step_composite, step_entry)

        work = self.manager.query_work_for_stage("step", 0)

        self.assertEqual(work, [step_key])

    def test_query_work_excludes_ready_keys(self):
        """Test query_work_for_stage excludes DONE keys."""
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")
        entry = self._create_ledger_entry(ArtifactLifecycle.DONE)
        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(step_composite, entry)

        work = self.manager.query_work_for_stage("step", 0)

        self.assertEqual(work, [])

    def test_query_work_excludes_error_keys(self):
        """Test query_work_for_stage excludes ERROR keys."""
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")
        entry = self._create_ledger_entry(
            ArtifactLifecycle.ERROR, error="test error"
        )
        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(step_composite, entry)

        work = self.manager.query_work_for_stage("step", 0)

        self.assertEqual(work, [])

    def test_query_work_excludes_processing_keys(self):
        """Test query_work_for_stage excludes PROCESSING keys."""
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")
        entry = self._create_ledger_entry(ArtifactLifecycle.PROCESSING)
        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(step_composite, entry)

        work = self.manager.query_work_for_stage("step", 0)

        self.assertEqual(work, [])


class TestHandleLifecycle(TestArtifactManagerLedger):
    """Test handle lifecycle in commit method."""

    def test_commit_calls_release_on_old_handle(self):
        """Test that commit calls release on the old handle exactly once."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        old_handle = create_mock_handle(WorkPieceArtifactHandle, "old_handle")
        new_handle = create_mock_handle(WorkPieceArtifactHandle, "new_handle")
        entry = self._create_ledger_entry(
            ArtifactLifecycle.PROCESSING,
            handle=old_handle,
            generation_id=1,
        )
        composite_key = make_composite_key(key, 1)
        self.manager._set_ledger_entry(composite_key, entry)

        self.manager.commit_artifact(key, new_handle, generation_id=1)

        self.mock_store.release.assert_called_once_with(old_handle)

    def test_commit_calls_adopt_on_new_handle(self):
        """Test that commit calls adopt on the new handle exactly once."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        new_handle = create_mock_handle(WorkPieceArtifactHandle, "new_handle")
        entry = self._create_ledger_entry(
            ArtifactLifecycle.PROCESSING,
            generation_id=1,
        )
        composite_key = make_composite_key(key, 1)
        self.manager._set_ledger_entry(composite_key, entry)

        self.manager.commit_artifact(key, new_handle, generation_id=1)

        self.mock_store.adopt.assert_called_once_with(new_handle)

    def test_commit_without_old_handle_does_not_call_release(self):
        """Test that committing without old handle doesn't call release."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        new_handle = create_mock_handle(WorkPieceArtifactHandle, "new_handle")
        entry = self._create_ledger_entry(
            ArtifactLifecycle.PROCESSING,
            handle=None,
            generation_id=1,
        )
        composite_key = make_composite_key(key, 1)
        self.manager._set_ledger_entry(composite_key, entry)

        self.manager.commit_artifact(key, new_handle, generation_id=1)

        self.mock_store.release.assert_not_called()

    def test_commit_replaces_handle_in_entry(self):
        """Test that commit replaces the handle in the ledger entry."""
        key = ArtifactKey.for_workpiece("00000000-0000-4000-8000-000000000001")
        old_handle = create_mock_handle(WorkPieceArtifactHandle, "old_handle")
        new_handle = create_mock_handle(WorkPieceArtifactHandle, "new_handle")
        entry = self._create_ledger_entry(
            ArtifactLifecycle.PROCESSING,
            handle=old_handle,
            generation_id=1,
        )
        composite_key = make_composite_key(key, 1)
        self.manager._set_ledger_entry(composite_key, entry)

        self.manager.commit_artifact(key, new_handle, generation_id=1)

        self.assertIs(entry.handle, new_handle)
        self.assertIsNot(entry.handle, old_handle)


class TestCheckoutDependencies(TestArtifactManagerLedger):
    """Test checkout_dependencies method."""

    def test_checkout_dependencies_returns_ready_handles(self):
        """Test that checkout_dependencies returns handles when deps DONE."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")
        wp_handle = create_mock_handle(WorkPieceArtifactHandle, "wp_handle")

        wp_entry = self._create_ledger_entry(
            ArtifactLifecycle.DONE, handle=wp_handle
        )
        step_entry = self._create_ledger_entry(ArtifactLifecycle.STALE)

        wp_composite = make_composite_key(workpiece_key, 0)
        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(wp_composite, wp_entry)
        self.manager._set_ledger_entry(step_composite, step_entry)
        self.manager._register_dependency(workpiece_key, step_key)

        deps = self.manager.checkout_dependencies(step_key, 0)

        self.assertEqual(deps, {workpiece_key: wp_handle})

    def test_checkout_dependencies_raises_on_not_ready(self):
        """Test that checkout_dependencies raises when dep not DONE."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")

        wp_entry = self._create_ledger_entry(ArtifactLifecycle.STALE)
        step_entry = self._create_ledger_entry(ArtifactLifecycle.STALE)

        wp_composite = make_composite_key(workpiece_key, 0)
        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(wp_composite, wp_entry)
        self.manager._set_ledger_entry(step_composite, step_entry)
        self.manager._register_dependency(workpiece_key, step_key)

        with self.assertRaises(AssertionError) as cm:
            self.manager.checkout_dependencies(step_key, 0)

        self.assertIn("is not DONE", str(cm.exception))

    def test_checkout_dependencies_raises_on_no_handle(self):
        """Test that checkout_dependencies raises when dep has no handle."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")

        wp_entry = self._create_ledger_entry(
            ArtifactLifecycle.DONE, handle=None
        )
        step_entry = self._create_ledger_entry(ArtifactLifecycle.STALE)

        wp_composite = make_composite_key(workpiece_key, 0)
        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(wp_composite, wp_entry)
        self.manager._set_ledger_entry(step_composite, step_entry)
        self.manager._register_dependency(workpiece_key, step_key)

        with self.assertRaises(AssertionError) as cm:
            self.manager.checkout_dependencies(step_key, 0)

        self.assertIn("has no handle", str(cm.exception))

    def test_checkout_dependencies_raises_on_missing_dep(self):
        """Test that checkout_dependencies raises when dep missing."""
        workpiece_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")

        step_entry = self._create_ledger_entry(ArtifactLifecycle.STALE)

        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(step_composite, step_entry)
        self.manager._register_dependency(workpiece_key, step_key)

        with self.assertRaises(AssertionError) as cm:
            self.manager.checkout_dependencies(step_key, 0)

        self.assertIn("not in ledger", str(cm.exception))

    def test_checkout_dependencies_returns_multiple_deps(self):
        """Test that checkout_dependencies returns all ready deps."""
        wp1_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000001"
        )
        wp2_key = ArtifactKey.for_workpiece(
            "00000000-0000-4000-8000-000000000002"
        )
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")
        wp1_handle = create_mock_handle(WorkPieceArtifactHandle, "wp1_handle")
        wp2_handle = create_mock_handle(WorkPieceArtifactHandle, "wp2_handle")

        wp1_entry = self._create_ledger_entry(
            ArtifactLifecycle.DONE, handle=wp1_handle
        )
        wp2_entry = self._create_ledger_entry(
            ArtifactLifecycle.DONE, handle=wp2_handle
        )
        step_entry = self._create_ledger_entry(ArtifactLifecycle.STALE)

        wp1_composite = make_composite_key(wp1_key, 0)
        wp2_composite = make_composite_key(wp2_key, 0)
        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(wp1_composite, wp1_entry)
        self.manager._set_ledger_entry(wp2_composite, wp2_entry)
        self.manager._set_ledger_entry(step_composite, step_entry)
        self.manager._register_dependency(wp1_key, step_key)
        self.manager._register_dependency(wp2_key, step_key)

        deps = self.manager.checkout_dependencies(step_key, 0)

        self.assertEqual(len(deps), 2)
        self.assertIs(deps[wp1_key], wp1_handle)
        self.assertIs(deps[wp2_key], wp2_handle)

    def test_checkout_dependencies_with_no_deps(self):
        """Test that checkout_dependencies returns empty dict when no deps."""
        step_key = ArtifactKey.for_step("00000000-0000-4000-8000-000000000003")
        step_entry = self._create_ledger_entry(ArtifactLifecycle.STALE)

        step_composite = make_composite_key(step_key, 0)
        self.manager._set_ledger_entry(step_composite, step_entry)

        deps = self.manager.checkout_dependencies(step_key, 0)

        self.assertEqual(deps, {})


if __name__ == "__main__":
    unittest.main()

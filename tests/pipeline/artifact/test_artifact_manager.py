"""
Tests for ArtifactManager as a pure cache.

The ArtifactManager is now a pure cache - it only stores handles.
State tracking is handled by DAG scheduler via ArtifactNode.
"""

import unittest
import uuid
from unittest.mock import Mock
from rayforge.pipeline.artifact import (
    ArtifactManager,
    ArtifactKey,
    WorkPieceArtifactHandle,
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    JobArtifactHandle,
)
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.artifact.manager import make_composite_key


STEP1_UID = str(uuid.uuid4())
STEP2_UID = str(uuid.uuid4())
WP1_UID = str(uuid.uuid4())
WP2_UID = str(uuid.uuid4())
WP3_UID = str(uuid.uuid4())
JOB_UID = str(uuid.uuid4())


def create_mock_handle(handle_class, name: str) -> Mock:
    """Creates a mock handle that behaves like a real handle for tests."""
    handle = Mock(spec=handle_class)
    handle.shm_name = f"shm_{name}"
    return handle


class TestArtifactManager(unittest.TestCase):
    """Test suite for ArtifactManager as a pure cache."""

    def setUp(self):
        """Set up a fresh manager and mock store."""
        self.mock_store = Mock(spec=ArtifactStore)
        self.mock_release = self.mock_store.release
        self.manager = ArtifactManager(self.mock_store)

    def test_put_and_get_step_handles(self):
        """Tests storage and retrieval of step handle types."""
        render_handle = create_mock_handle(
            StepRenderArtifactHandle, "step1_render"
        )
        ops_handle = create_mock_handle(StepOpsArtifactHandle, "step1_ops")

        self.manager.put_step_render_handle(STEP1_UID, render_handle)
        self.manager.put_step_ops_handle(
            ArtifactKey.for_step(STEP1_UID), ops_handle, 0
        )

        retrieved_render = self.manager.get_step_render_handle(STEP1_UID)
        retrieved_ops = self.manager.get_step_ops_handle(
            ArtifactKey.for_step(STEP1_UID), 0
        )

        self.assertIs(retrieved_render, render_handle)
        self.assertIs(retrieved_ops, ops_handle)
        self.mock_release.assert_not_called()

    def test_put_and_get_job(self):
        """Tests basic storage and retrieval of a job handle."""
        handle = create_mock_handle(JobArtifactHandle, "job")
        job_key = ArtifactKey(id=JOB_UID, group="job")
        self.manager.cache_handle(job_key, handle, 0)

        retrieved = self.manager.get_job_handle(job_key, 0)
        self.assertIs(retrieved, handle)
        self.mock_release.assert_not_called()

    def test_invalidate_workpiece_cascades_correctly(self):
        """Tests that invalidating a workpiece removes only that workpiece."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")

        self.manager.cache_handle(ArtifactKey.for_workpiece(WP1_UID), wp_h, 0)
        self.manager.put_step_render_handle(WP1_UID, render_h)
        self.manager.put_step_ops_handle(
            ArtifactKey.for_step(WP1_UID), ops_h, 0
        )

        self.manager.invalidate_for_workpiece(
            ArtifactKey.for_workpiece(WP1_UID)
        )

        self.assertIsNone(
            self.manager.get_workpiece_handle(
                ArtifactKey.for_workpiece(WP1_UID), 0
            )
        )
        self.assertIs(
            self.manager.get_step_ops_handle(ArtifactKey.for_step(WP1_UID), 0),
            ops_h,
        )
        self.assertIs(self.manager.get_step_render_handle(WP1_UID), render_h)
        self.mock_release.assert_any_call(wp_h)

    def test_invalidate_step_cascades_correctly(self):
        """Tests that invalidating a step cascades to all dependent."""
        wp1_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp2_h = create_mock_handle(WorkPieceArtifactHandle, "wp2")
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")

        self.manager.cache_handle(
            ArtifactKey.for_workpiece(STEP1_UID), wp1_h, 0
        )
        self.manager.cache_handle(
            ArtifactKey.for_workpiece(STEP1_UID), wp2_h, 1
        )
        self.manager.put_step_render_handle(STEP1_UID, render_h)
        self.manager.put_step_ops_handle(
            ArtifactKey.for_step(STEP1_UID), ops_h, 0
        )

        self.manager.invalidate_for_step(ArtifactKey.for_step(STEP1_UID))

        self.assertIsNone(
            self.manager.get_workpiece_handle(
                ArtifactKey.for_workpiece(STEP1_UID), 0
            )
        )
        self.assertIsNone(
            self.manager.get_step_ops_handle(
                ArtifactKey.for_step(STEP1_UID), 0
            )
        )
        self.assertIsNone(self.manager.get_step_render_handle(STEP1_UID))
        self.mock_release.assert_any_call(wp1_h)
        self.mock_release.assert_any_call(wp2_h)
        self.mock_release.assert_any_call(render_h)
        self.mock_release.assert_any_call(ops_h)

    def test_put_job_replaces_old_handle(self):
        """Tests that putting a job handle releases old handle."""
        old_job_h = create_mock_handle(JobArtifactHandle, "job_old")
        new_job_h = create_mock_handle(JobArtifactHandle, "job_new")
        job_key = ArtifactKey(id=JOB_UID, group="job")

        self.manager.cache_handle(job_key, old_job_h, 0)
        self.manager.cache_handle(job_key, new_job_h, 0)

        self.assertIs(self.manager.get_job_handle(job_key, 0), new_job_h)
        self.mock_release.assert_called_with(old_job_h)

    def test_shutdown_releases_all_artifacts(self):
        """Tests that shutdown releases all cached artifacts."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        self.manager.cache_handle(ArtifactKey.for_workpiece(WP1_UID), wp_h, 0)
        self.manager.put_step_render_handle(STEP1_UID, render_h)
        self.manager.put_step_ops_handle(
            ArtifactKey.for_step(STEP1_UID), ops_h, 0
        )
        self.manager.cache_handle(
            ArtifactKey(id=JOB_UID, group="job"), job_h, 0
        )

        self.manager.shutdown()

        self.mock_release.assert_any_call(wp_h)
        self.mock_release.assert_any_call(render_h)
        self.mock_release.assert_any_call(ops_h)
        self.mock_release.assert_any_call(job_h)
        self.assertEqual(len(self.manager._ledger), 0)
        self.assertEqual(len(self.manager._step_render_handles), 0)

    def test_get_all_workpiece_keys(self):
        """Tests getting all workpiece keys."""
        self.manager.cache_handle(
            ArtifactKey.for_workpiece(WP1_UID),
            create_mock_handle(WorkPieceArtifactHandle, "wp1"),
            0,
        )
        self.manager.cache_handle(
            ArtifactKey.for_workpiece(WP2_UID),
            create_mock_handle(WorkPieceArtifactHandle, "wp2"),
            0,
        )
        self.manager.cache_handle(
            ArtifactKey.for_workpiece(WP3_UID),
            create_mock_handle(WorkPieceArtifactHandle, "wp3"),
            0,
        )

        keys = self.manager.get_all_workpiece_keys()
        self.assertEqual(len(keys), 3)
        self.assertIn(ArtifactKey.for_workpiece(WP1_UID), keys)
        self.assertIn(ArtifactKey.for_workpiece(WP2_UID), keys)
        self.assertIn(ArtifactKey.for_workpiece(WP3_UID), keys)

    def test_get_all_step_render_uids(self):
        """Tests getting all step render UIDs."""
        self.manager._step_render_handles[STEP1_UID] = Mock()
        self.manager._step_render_handles[STEP2_UID] = Mock()

        uids = self.manager.get_all_step_render_uids()
        self.assertEqual(len(uids), 2)
        self.assertIn(STEP1_UID, uids)
        self.assertIn(STEP2_UID, uids)

    def test_has_step_render_handle(self):
        """Tests checking if a step render handle exists."""
        self.manager._step_render_handles[STEP1_UID] = Mock()
        self.assertTrue(self.manager.has_step_render_handle(STEP1_UID))
        self.assertFalse(self.manager.has_step_render_handle(STEP2_UID))

    def test_pop_step_render_handle(self):
        """Tests popping a step render handle."""
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        self.manager._step_render_handles[STEP1_UID] = render_h

        popped = self.manager.pop_step_render_handle(STEP1_UID)
        self.assertIs(popped, render_h)
        self.assertIsNone(self.manager.get_step_render_handle(STEP1_UID))

    def test_checkout_step_render_handle(self):
        """Tests checking out a step ops handle."""
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        self.manager.put_step_ops_handle(
            ArtifactKey.for_step(STEP1_UID), ops_h, 0
        )

        with self.manager.checkout(
            ArtifactKey.for_step(STEP1_UID), 0
        ) as handle:
            self.assertIs(handle, ops_h)
            self.assertEqual(
                self.manager._ref_counts[ArtifactKey.for_step(STEP1_UID)], 1
            )
            self.mock_store.retain.assert_called_once_with(ops_h)

        self.mock_store.release.assert_called_once_with(ops_h)
        self.assertNotIn(
            ArtifactKey.for_step(STEP1_UID), self.manager._ref_counts
        )

    def test_checkout_job_handle(self):
        """Tests checking out job handle."""
        job_h = create_mock_handle(JobArtifactHandle, "job")
        job_key = ArtifactKey(id=JOB_UID, group="job")
        self.manager.cache_handle(job_key, job_h, 0)

        with self.manager.checkout(job_key, 0) as handle:
            self.assertIs(handle, job_h)
            self.assertEqual(self.manager._ref_counts[job_key], 1)
            self.mock_store.retain.assert_called_with(job_h)

        self.mock_store.release.assert_called_with(job_h)
        self.assertNotIn(job_key, self.manager._ref_counts)

    def test_get_workpiece_handle_from_ledger(self):
        """Tests getting workpiece handle from ledger."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        self.manager.cache_handle(ArtifactKey.for_workpiece(WP1_UID), wp_h, 0)

        retrieved = self.manager.get_workpiece_handle(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        self.assertIs(retrieved, wp_h)

    def test_get_workpiece_handle_returns_none_when_not_found(self):
        """Tests getting workpiece handle returns None when not found."""
        retrieved = self.manager.get_workpiece_handle(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        self.assertIsNone(retrieved)

    def test_prune_removes_obsolete_data_generation(self):
        """Tests pruning removes ledger entries from non-active data gens."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        self.manager.cache_handle(ArtifactKey.for_workpiece(WP1_UID), wp_h, 0)

        self.manager.prune(active_data_gen_ids={1})

        wp_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        self.assertNotIn(wp_composite, self.manager._ledger)

    def test_prune_keeps_active_data_generation(self):
        """Tests pruning keeps artifacts from active data generations."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        self.manager.cache_handle(ArtifactKey.for_workpiece(WP1_UID), wp_h, 0)

        self.manager.prune(active_data_gen_ids={0})

        wp_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        self.assertIn(wp_composite, self.manager._ledger)
        self.mock_release.assert_not_called()

    def test_prune_preserves_step_for_processing_data_gen(self):
        """Tests pruning preserves step entries for processing data gens."""
        step_h0 = create_mock_handle(StepOpsArtifactHandle, "step0")
        step_h1 = create_mock_handle(StepOpsArtifactHandle, "step1")
        step_key0 = ArtifactKey.for_step(STEP1_UID)
        step_key1 = ArtifactKey.for_step(STEP2_UID)
        self.manager.cache_handle(step_key0, step_h0, 0)
        self.manager.cache_handle(step_key1, step_h1, 1)

        self.manager.prune(
            active_data_gen_ids={1},
            processing_data_gen_ids={1, 0},
        )

        step_composite_0 = make_composite_key(step_key0, 0)
        step_composite_1 = make_composite_key(step_key1, 1)
        self.assertIn(step_composite_0, self.manager._ledger)
        self.assertIn(step_composite_1, self.manager._ledger)
        self.mock_release.assert_not_called()

    def test_is_generation_current_returns_true_for_matching(self):
        """Test is_generation_current returns True for matching gen ID."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        self.manager.cache_handle(wp_key, wp_h, 1)

        result = self.manager.is_generation_current(wp_key, 1)
        self.assertTrue(result)

    def test_is_generation_current_returns_false_for_mismatch(self):
        """Test is_generation_current returns False for gen ID mismatch."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        self.manager.cache_handle(wp_key, wp_h, 1)

        result = self.manager.is_generation_current(wp_key, 2)
        self.assertFalse(result)

    def test_is_generation_current_returns_false_for_missing(self):
        """Test is_generation_current returns False for missing entry."""
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        result = self.manager.is_generation_current(wp_key, 0)
        self.assertFalse(result)

    def test_get_artifact_retrieves_from_store(self):
        """Test get_artifact retrieves artifact from store."""
        handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        artifact = Mock()
        self.mock_store.get.return_value = artifact

        result = self.manager.get_artifact(handle)

        self.assertIs(result, artifact)
        self.mock_store.get.assert_called_once_with(handle)

    def test_put_step_render_handle_replaces_old(self):
        """
        Test put_step_render_handle replaces old handle without releasing.
        """
        old_h = create_mock_handle(StepRenderArtifactHandle, "old")
        new_h = create_mock_handle(StepRenderArtifactHandle, "new")
        self.manager._step_render_handles[STEP1_UID] = old_h

        self.manager.put_step_render_handle(STEP1_UID, new_h)

        self.assertIs(self.manager._step_render_handles[STEP1_UID], new_h)
        self.mock_release.assert_not_called()

    def test_put_step_render_handle_without_old(self):
        """Test put_step_render_handle stores without old handle."""
        new_h = create_mock_handle(StepRenderArtifactHandle, "new")

        self.manager.put_step_render_handle(STEP1_UID, new_h)

        self.assertIs(self.manager._step_render_handles[STEP1_UID], new_h)
        self.mock_release.assert_not_called()

    def test_put_step_ops_handle_creates_entry(self):
        """Test put_step_ops_handle creates entry."""
        handle = create_mock_handle(StepOpsArtifactHandle, "ops")
        step_key = ArtifactKey.for_step(STEP1_UID)

        self.manager.put_step_ops_handle(step_key, handle, 1)

        composite_key = make_composite_key(step_key, 1)
        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertIs(entry.handle, handle)
        self.assertEqual(entry.generation_id, 1)

    def test_retain_handle_calls_store_retain(self):
        """Test retain_handle delegates to store.retain."""
        handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")

        self.manager.retain_handle(handle)

        self.mock_store.retain.assert_called_once_with(handle)

    def test_get_all_workpiece_keys_for_generation(self):
        """Test getting workpiece keys for specific generation."""
        self.manager.cache_handle(
            ArtifactKey.for_workpiece(WP1_UID),
            create_mock_handle(WorkPieceArtifactHandle, "wp1"),
            0,
        )
        self.manager.cache_handle(
            ArtifactKey.for_workpiece(WP2_UID),
            create_mock_handle(WorkPieceArtifactHandle, "wp2"),
            1,
        )
        self.manager.cache_handle(
            ArtifactKey.for_workpiece(WP3_UID),
            create_mock_handle(WorkPieceArtifactHandle, "wp3"),
            0,
        )

        keys = self.manager.get_all_workpiece_keys_for_generation(0)

        self.assertEqual(len(keys), 2)
        self.assertIn(ArtifactKey.for_workpiece(WP1_UID), keys)
        self.assertIn(ArtifactKey.for_workpiece(WP3_UID), keys)
        self.assertNotIn(ArtifactKey.for_workpiece(WP2_UID), keys)

    def test_invalidate_for_job_releases_and_removes(self):
        """Test invalidate_for_job releases handle and removes entry."""
        job_h = create_mock_handle(JobArtifactHandle, "job")
        job_key = ArtifactKey(id=JOB_UID, group="job")
        self.manager.cache_handle(job_key, job_h, 0)

        self.manager.invalidate_for_job(job_key)

        job_composite = make_composite_key(job_key, 0)
        self.assertNotIn(job_composite, self.manager._ledger)
        self.mock_release.assert_called_once_with(job_h)

    def test_checkout_handle_with_none(self):
        """Test checkout_handle yields None for None handle."""
        with self.manager.checkout_handle(None) as artifact:
            self.assertIsNone(artifact)

    def test_checkout_handle_retains_and_releases(self):
        """Test checkout_handle retains on entry and releases on exit."""
        handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        artifact = Mock()
        self.mock_store.get.return_value = artifact

        with self.manager.checkout_handle(handle):
            self.mock_store.retain.assert_called_once_with(handle)

        self.mock_store.release.assert_called_once_with(handle)

    def test_checkout_handle_releases_on_exception(self):
        """Test checkout_handle releases even if exception occurs."""
        handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        artifact = Mock()
        self.mock_store.get.return_value = artifact

        with self.assertRaises(ValueError):
            with self.manager.checkout_handle(handle):
                self.mock_store.retain.assert_called_once_with(handle)
                raise ValueError("test")

        self.mock_store.release.assert_called_once_with(handle)

    def test_mark_done_without_handle(self):
        """Test mark_done works when entry has no handle."""
        key = ArtifactKey.for_workpiece(WP1_UID)
        self.manager.declare_generation({key}, 0)

        self.manager.mark_done(key, 0)

        composite_key = make_composite_key(key, 0)
        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertIsNone(entry.handle)
        self.mock_release.assert_not_called()

    def test_mark_done_nonexistent_skips(self):
        """Test mark_done skips when entry does not exist."""
        key = ArtifactKey.for_workpiece(WP1_UID)

        self.manager.mark_done(key, 0)

        self.assertEqual(len(self.manager._ledger), 0)

    def test_complete_generation_marks_existing_entry_done(self):
        """Test complete_generation marks existing entry as done."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        key = ArtifactKey.for_workpiece(WP1_UID)
        self.manager.cache_handle(key, wp_h, 0)

        self.manager.complete_generation(key, 0)

        composite_key = make_composite_key(key, 0)
        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertIs(entry.handle, wp_h)

    def test_complete_generation_creates_new_entry_with_handle(self):
        """Test complete_generation creates new entry with handle."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        key = ArtifactKey.for_workpiece(WP1_UID)

        self.manager.complete_generation(key, 0, wp_h)

        composite_key = make_composite_key(key, 0)
        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertIs(entry.handle, wp_h)

    def test_complete_generation_replaces_handle(self):
        """Test complete_generation replaces existing handle."""
        old_h = create_mock_handle(WorkPieceArtifactHandle, "old")
        new_h = create_mock_handle(WorkPieceArtifactHandle, "new")
        key = ArtifactKey.for_workpiece(WP1_UID)
        self.manager.cache_handle(key, old_h, 0)

        self.manager.complete_generation(key, 0, new_h)

        composite_key = make_composite_key(key, 0)
        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertIs(entry.handle, new_h)
        self.mock_release.assert_called_once_with(old_h)

    def test_complete_generation_nonexistent_without_handle_creates_entry(
        self,
    ):
        """Test complete_generation creates done entry when no handle."""
        key = ArtifactKey.for_workpiece(WP1_UID)

        self.manager.complete_generation(key, 0)

        composite_key = make_composite_key(key, 0)
        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertIsNone(entry.handle)

    def test_declare_generation_creates_placeholder_entries(self):
        """Test declare_generation creates placeholder entries for new keys."""
        wp1_key = ArtifactKey.for_workpiece(WP1_UID)
        wp2_key = ArtifactKey.for_workpiece(WP2_UID)

        self.manager.declare_generation({wp1_key, wp2_key}, 0)

        wp1_composite = make_composite_key(wp1_key, 0)
        wp2_composite = make_composite_key(wp2_key, 0)
        entry1 = self.manager._ledger.get(wp1_composite)
        entry2 = self.manager._ledger.get(wp2_composite)
        assert entry1 is not None
        assert entry2 is not None
        self.assertIsNone(entry1.handle)
        self.assertIsNone(entry2.handle)

    def test_declare_generation_skips_existing_entries(self):
        """Test declare_generation skips existing entries."""
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        self.manager.cache_handle(wp_key, wp_h, 0)

        self.manager.declare_generation({wp_key}, 0)

        wp_composite = make_composite_key(wp_key, 0)
        entry = self.manager._ledger.get(wp_composite)
        assert entry is not None
        self.assertIs(entry.handle, wp_h)

    def test_declare_generation_does_not_modify_previous_generations(self):
        """Test declare_generation does not modify previous generations."""
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        self.manager.cache_handle(wp_key, wp_h, 0)

        self.manager.declare_generation({wp_key}, 1)

        wp_composite_g0 = make_composite_key(wp_key, 0)
        wp_composite_g1 = make_composite_key(wp_key, 1)
        entry_g0 = self.manager._ledger.get(wp_composite_g0)
        entry_g1 = self.manager._ledger.get(wp_composite_g1)
        assert entry_g0 is not None
        assert entry_g1 is not None
        self.assertIs(entry_g0.handle, wp_h)
        self.assertIsNone(entry_g1.handle)

    def test_declare_generation_copies_step_handle_from_previous(self):
        """Test declare_generation copies step handles from previous gen."""
        step_key = ArtifactKey.for_step(STEP1_UID)
        step_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        self.manager.cache_handle(step_key, step_h, 0)

        self.manager.declare_generation({step_key}, 1)

        step_composite_g0 = make_composite_key(step_key, 0)
        step_composite_g1 = make_composite_key(step_key, 1)
        entry_g0 = self.manager._ledger.get(step_composite_g0)
        entry_g1 = self.manager._ledger.get(step_composite_g1)
        assert entry_g0 is not None
        assert entry_g1 is not None
        self.assertIs(entry_g0.handle, step_h)
        self.assertIs(entry_g1.handle, step_h)
        self.mock_store.retain.assert_called()


class TestArtifactManagerCommitRetains(unittest.TestCase):
    """Tests for Manager's Claim on cache_handle."""

    def setUp(self):
        """Set up a fresh manager and mock store."""
        self.mock_store = Mock(spec=ArtifactStore)
        self.mock_retain = self.mock_store.retain
        self.manager = ArtifactManager(self.mock_store)

    def test_cache_handle_retains_handle(self):
        """Test that cache_handle calls retain on handle."""
        handle = create_mock_handle(JobArtifactHandle, "job")
        job_key = ArtifactKey(id=JOB_UID, group="job")

        self.manager.cache_handle(job_key, handle, 0)

        self.mock_retain.assert_called_once_with(handle)

    def test_cache_handle_retains_new_handle_releases_old(self):
        """Test that replacing a handle retains new and releases old."""
        old_handle = create_mock_handle(JobArtifactHandle, "job_old")
        new_handle = create_mock_handle(JobArtifactHandle, "job_new")
        job_key = ArtifactKey(id=JOB_UID, group="job")

        self.manager.cache_handle(job_key, old_handle, 0)
        self.mock_retain.reset_mock()

        self.manager.cache_handle(job_key, new_handle, 0)

        self.mock_retain.assert_called_once_with(new_handle)
        self.mock_store.release.assert_called_once_with(old_handle)

    def test_cache_handle_retains_multiple_commits(self):
        """Test that each cache_handle call retains handle."""
        handle1 = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        handle2 = create_mock_handle(WorkPieceArtifactHandle, "wp2")

        wp1_key = ArtifactKey.for_workpiece(WP1_UID)
        wp2_key = ArtifactKey.for_workpiece(WP2_UID)

        self.manager.cache_handle(wp1_key, handle1, 0)
        self.manager.cache_handle(wp2_key, handle2, 0)

        retain_calls = self.mock_retain.call_args_list
        self.assertEqual(len(retain_calls), 2)
        self.assertIn(handle1, [call[0][0] for call in retain_calls])
        self.assertIn(handle2, [call[0][0] for call in retain_calls])


if __name__ == "__main__":
    unittest.main()

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
from rayforge.pipeline.artifact.lifecycle import ArtifactLifecycle
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
    """Test suite for ArtifactManager."""

    def setUp(self):
        """Set up a fresh manager and mock store."""
        self.mock_store = Mock(spec=ArtifactStore)
        self.mock_release = self.mock_store.release
        self.manager = ArtifactManager(self.mock_store)

    def test_put_and_get_step_handles(self):
        """Tests storage and retrieval of new step handle types."""
        render_handle = create_mock_handle(
            StepRenderArtifactHandle, "step1_render"
        )
        ops_handle = create_mock_handle(StepOpsArtifactHandle, "step1_ops")

        self.manager.put_step_render_handle(STEP1_UID, render_handle)
        ledger_key = ArtifactKey.for_step(STEP1_UID)
        composite_key = make_composite_key(ledger_key, 0)
        self.manager._ledger[composite_key] = Mock(
            state=ArtifactLifecycle.DONE, handle=ops_handle
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
        job_composite = make_composite_key(job_key, 0)
        self.manager._ledger[job_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=handle
        )
        retrieved = self.manager.get_job_handle(job_key, 0)
        self.assertIs(retrieved, handle)
        self.mock_release.assert_not_called()

    def test_invalidate_workpiece_cascades_correctly(self):
        """
        Tests that invalidating a workpiece cascades to step ops artifact,
        leaving the render artifact intact.
        """
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")

        wp_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        self.manager._ledger[wp_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=wp_h
        )
        self.manager._step_render_handles[WP1_UID] = render_h
        step_composite = make_composite_key(ArtifactKey.for_step(WP1_UID), 0)
        self.manager._ledger[step_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=ops_h
        )

        self.manager.invalidate_for_workpiece(
            ArtifactKey.for_workpiece(WP1_UID)
        )

        # Assert correct artifacts were removed
        self.assertIsNone(
            self.manager.get_workpiece_handle(
                ArtifactKey.for_workpiece(WP1_UID), 0
            )
        )
        self.assertIsNone(
            self.manager.get_step_ops_handle(ArtifactKey.for_step(WP1_UID), 0)
        )
        # Assert render handle remains for UI stability
        self.assertIs(self.manager.get_step_render_handle(WP1_UID), render_h)
        # Assert only workpiece and ops were released
        self.mock_release.assert_any_call(wp_h)
        self.mock_release.assert_any_call(ops_h)
        self.assertEqual(self.mock_release.call_count, 2)

    def test_invalidate_step_cascades_correctly(self):
        """
        Tests that invalidating a step cascades to all dependent
        artifacts (workpieces and step artifacts).
        """
        wp1_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp2_h = create_mock_handle(WorkPieceArtifactHandle, "wp2")
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")

        wp1_composite = make_composite_key(
            ArtifactKey.for_workpiece(STEP1_UID), 0
        )
        wp2_composite = make_composite_key(
            ArtifactKey.for_workpiece(STEP1_UID), 1
        )
        self.manager._ledger[wp1_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=wp1_h
        )
        self.manager._ledger[wp2_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=wp2_h
        )
        self.manager._step_render_handles[STEP1_UID] = render_h
        step_composite = make_composite_key(ArtifactKey.for_step(STEP1_UID), 0)
        self.manager._ledger[step_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=ops_h
        )

        self.manager.invalidate_for_step(ArtifactKey.for_step(STEP1_UID))

        # Assert all step-related artifacts were removed
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
        # Assert all artifacts were released
        self.mock_release.assert_any_call(wp1_h)
        self.mock_release.assert_any_call(wp2_h)
        self.mock_release.assert_any_call(render_h)
        self.mock_release.assert_any_call(ops_h)
        self.assertEqual(self.mock_release.call_count, 4)

    def test_put_job_replaces_old_handle(self):
        """Tests that putting a job handle releases of old handle."""
        old_job_h = create_mock_handle(JobArtifactHandle, "job_old")
        new_job_h = create_mock_handle(JobArtifactHandle, "job_new")
        job_key = ArtifactKey(id=JOB_UID, group="job")
        job_composite = make_composite_key(job_key, 0)
        self.manager._ledger[job_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=old_job_h
        )

        self.manager._ledger[job_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=new_job_h
        )
        self.manager.release_handle(old_job_h)

        self.assertIs(self.manager.get_job_handle(job_key, 0), new_job_h)
        self.mock_release.assert_called_once_with(old_job_h)

    def test_shutdown_releases_all_artifacts(self):
        """Tests that shutdown releases all cached artifacts."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        wp_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        self.manager._ledger[wp_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=wp_h
        )
        self.manager._step_render_handles[STEP1_UID] = render_h
        step_composite = make_composite_key(ArtifactKey.for_step(STEP1_UID), 0)
        self.manager._ledger[step_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=ops_h
        )
        job_composite = make_composite_key(
            ArtifactKey(id=JOB_UID, group="job"), 0
        )
        self.manager._ledger[job_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=job_h
        )

        self.manager.shutdown()

        # Assert all handles were released
        self.mock_release.assert_any_call(wp_h)
        self.mock_release.assert_any_call(render_h)
        self.mock_release.assert_any_call(ops_h)
        self.mock_release.assert_any_call(job_h)
        self.assertEqual(self.mock_release.call_count, 4)
        # Assert all caches are cleared
        self.assertEqual(len(self.manager._ledger), 0)
        self.assertEqual(len(self.manager._step_render_handles), 0)

    def test_get_all_workpiece_keys(self):
        """Tests getting all workpiece keys."""
        wp1_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        wp2_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP2_UID), 0
        )
        wp3_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP3_UID), 0
        )
        self.manager._ledger[wp1_composite] = Mock()
        self.manager._ledger[wp2_composite] = Mock()
        self.manager._ledger[wp3_composite] = Mock()

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
        step_composite = make_composite_key(ArtifactKey.for_step(STEP1_UID), 0)
        self.manager._ledger[step_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=ops_h
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
        job_composite = make_composite_key(job_key, 0)
        self.manager._ledger[job_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=job_h
        )

        with self.manager.checkout(job_key, 0) as handle:
            self.assertIs(handle, job_h)
            self.assertEqual(self.manager._ref_counts[job_key], 1)
            self.mock_store.retain.assert_called_once_with(job_h)

        self.mock_store.release.assert_called_once_with(job_h)
        self.assertNotIn(job_key, self.manager._ref_counts)

    def test_get_workpiece_handle_from_ledger(self):
        """Tests getting workpiece handle from ledger."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        ledger_key = ArtifactKey.for_workpiece(WP1_UID)
        composite_key = make_composite_key(ledger_key, 0)
        self.manager._ledger[composite_key] = Mock(
            state=ArtifactLifecycle.DONE, handle=wp_h
        )

        retrieved = self.manager.get_workpiece_handle(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        self.assertIs(retrieved, wp_h)

    def test_get_workpiece_handle_returns_none_when_not_ready(self):
        """Tests getting workpiece handle returns None when not DONE."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        ledger_key = ArtifactKey.for_workpiece(WP1_UID)
        composite_key = make_composite_key(ledger_key, 0)
        self.manager._ledger[composite_key] = Mock(
            state=ArtifactLifecycle.PROCESSING, handle=wp_h
        )

        retrieved = self.manager.get_workpiece_handle(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        self.assertIsNone(retrieved)

    def test_get_workpiece_handle_returns_none_when_not_found(self):
        """Tests getting workpiece handle returns None when not found."""
        retrieved = self.manager.get_workpiece_handle(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        self.assertIsNone(retrieved)

    def test_prune_removes_obsolete_data_generation(self):
        """Tests pruning removes artifacts from non-active data generations."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        self.manager._ledger[wp_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=wp_h
        )

        self.manager.prune(active_data_gen_ids={1}, active_view_gen_ids=set())

        self.assertNotIn(wp_composite, self.manager._ledger)
        self.mock_release.assert_called_once_with(wp_h)

    def test_prune_keeps_active_data_generation(self):
        """Tests pruning keeps artifacts from active data generations."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        self.manager._ledger[wp_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=wp_h
        )

        self.manager.prune(active_data_gen_ids={0}, active_view_gen_ids=set())

        self.assertIn(wp_composite, self.manager._ledger)
        self.mock_release.assert_not_called()

    def test_prune_removes_obsolete_view_generation(self):
        """Tests pruning removes artifacts from non-active view generations."""
        from rayforge.pipeline.artifact import WorkPieceViewArtifactHandle

        view_h = create_mock_handle(WorkPieceViewArtifactHandle, "view1")
        view_composite = make_composite_key(ArtifactKey.for_view(WP1_UID), 0)
        self.manager._ledger[view_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=view_h
        )

        self.manager.prune(active_data_gen_ids=set(), active_view_gen_ids={1})

        self.assertNotIn(view_composite, self.manager._ledger)
        self.mock_release.assert_called_once_with(view_h)

    def test_prune_keeps_active_view_generation(self):
        """Tests pruning keeps artifacts from active view generations."""
        from rayforge.pipeline.artifact import WorkPieceViewArtifactHandle

        view_h = create_mock_handle(WorkPieceViewArtifactHandle, "view1")
        view_composite = make_composite_key(ArtifactKey.for_view(WP1_UID), 0)
        self.manager._ledger[view_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=view_h
        )

        self.manager.prune(active_data_gen_ids=set(), active_view_gen_ids={0})

        self.assertIn(view_composite, self.manager._ledger)
        self.mock_release.assert_not_called()

    def test_prune_preserves_processing_entries(self):
        """Tests pruning does not remove entries in PROCESSING state."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        self.manager._ledger[wp_composite] = Mock(
            state=ArtifactLifecycle.PROCESSING, handle=wp_h
        )

        self.manager.prune(active_data_gen_ids={1}, active_view_gen_ids=set())

        self.assertIn(wp_composite, self.manager._ledger)
        self.mock_release.assert_not_called()

    def test_prune_mixed_generations(self):
        """Tests pruning with multiple generations of mixed types."""
        wp_h0 = create_mock_handle(WorkPieceArtifactHandle, "wp0")
        wp_h1 = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        from rayforge.pipeline.artifact import WorkPieceViewArtifactHandle

        view_h0 = create_mock_handle(WorkPieceViewArtifactHandle, "view0")
        view_h1 = create_mock_handle(WorkPieceViewArtifactHandle, "view1")

        wp_composite_0 = make_composite_key(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        wp_composite_1 = make_composite_key(
            ArtifactKey.for_workpiece(WP2_UID), 1
        )
        view_composite_0 = make_composite_key(ArtifactKey.for_view(WP1_UID), 0)
        view_composite_1 = make_composite_key(ArtifactKey.for_view(WP2_UID), 1)

        self.manager._ledger[wp_composite_0] = Mock(
            state=ArtifactLifecycle.DONE, handle=wp_h0
        )
        self.manager._ledger[wp_composite_1] = Mock(
            state=ArtifactLifecycle.DONE, handle=wp_h1
        )
        self.manager._ledger[view_composite_0] = Mock(
            state=ArtifactLifecycle.DONE, handle=view_h0
        )
        self.manager._ledger[view_composite_1] = Mock(
            state=ArtifactLifecycle.DONE, handle=view_h1
        )

        self.manager.prune(active_data_gen_ids={1}, active_view_gen_ids={0})

        self.assertNotIn(wp_composite_0, self.manager._ledger)
        self.assertIn(wp_composite_1, self.manager._ledger)
        self.assertIn(view_composite_0, self.manager._ledger)
        self.assertNotIn(view_composite_1, self.manager._ledger)

        self.mock_release.assert_any_call(wp_h0)
        self.mock_release.assert_any_call(view_h1)
        self.assertEqual(self.mock_release.call_count, 2)

    def test_prune_ledger_size_stays_constant(self):
        """Tests that ledger size stays constant after cycling generations."""
        initial_size = 0
        for gen_id in range(3):
            wp_h = create_mock_handle(WorkPieceArtifactHandle, f"wp{gen_id}")
            wp_composite = make_composite_key(
                ArtifactKey.for_workpiece(WP1_UID), gen_id
            )
            self.manager._ledger[wp_composite] = Mock(
                state=ArtifactLifecycle.DONE, handle=wp_h
            )
            initial_size += 1

            self.manager.prune(
                active_data_gen_ids={gen_id}, active_view_gen_ids=set()
            )

            self.assertEqual(len(self.manager._ledger), 1)

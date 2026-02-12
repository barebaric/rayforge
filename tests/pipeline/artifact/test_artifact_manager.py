import unittest
import unittest.mock
import uuid
from unittest.mock import Mock
from rayforge.pipeline.artifact import (
    ArtifactManager,
    ArtifactKey,
    WorkPieceArtifactHandle,
    WorkPieceViewArtifactHandle,
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    JobArtifactHandle,
)
from rayforge.pipeline.artifact.lifecycle import ArtifactLifecycle
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.artifact.manager import make_composite_key
from rayforge.pipeline.artifact.workpiece_view import RenderContext


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

    def test_is_generation_current_returns_true_for_matching(self):
        """Test is_generation_current returns True for matching gen ID."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        wp_composite = make_composite_key(wp_key, 1)
        self.manager._ledger[wp_composite] = Mock(
            state=ArtifactLifecycle.DONE,
            handle=wp_h,
            generation_id=1,
        )

        result = self.manager.is_generation_current(wp_key, 1)

        self.assertTrue(result)

    def test_is_generation_current_returns_false_for_mismatch(self):
        """Test is_generation_current returns False for gen ID mismatch."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        wp_composite = make_composite_key(wp_key, 1)
        self.manager._ledger[wp_composite] = Mock(
            state=ArtifactLifecycle.DONE,
            handle=wp_h,
            generation_id=1,
        )

        result = self.manager.is_generation_current(wp_key, 2)

        self.assertFalse(result)

    def test_is_generation_current_returns_false_for_missing(self):
        """Test is_generation_current returns False for missing entry."""
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        result = self.manager.is_generation_current(wp_key, 0)

        self.assertFalse(result)

    def test_get_workpiece_view_handle_returns_handle(self):
        """Test get_workpiece_view_handle returns the handle when DONE."""
        view_h = create_mock_handle(WorkPieceViewArtifactHandle, "view1")
        view_composite = make_composite_key(ArtifactKey.for_view(WP1_UID), 0)
        self.manager._ledger[view_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=view_h
        )

        retrieved = self.manager.get_workpiece_view_handle(
            ArtifactKey.for_view(WP1_UID), 0
        )

        self.assertIs(retrieved, view_h)

    def test_get_workpiece_view_handle_returns_handle_when_processing(self):
        """Test get_workpiece_view_handle returns handle when PROCESSING."""
        view_h = create_mock_handle(WorkPieceViewArtifactHandle, "view1")
        view_composite = make_composite_key(ArtifactKey.for_view(WP1_UID), 0)
        self.manager._ledger[view_composite] = Mock(
            state=ArtifactLifecycle.PROCESSING, handle=view_h
        )

        retrieved = self.manager.get_workpiece_view_handle(
            ArtifactKey.for_view(WP1_UID), 0
        )

        self.assertIs(retrieved, view_h)

    def test_get_workpiece_view_handle_returns_none_when_initial(self):
        """Test get_workpiece_view_handle returns None when QUEUED."""
        view_composite = make_composite_key(ArtifactKey.for_view(WP1_UID), 0)
        self.manager._ledger[view_composite] = Mock(
            state=ArtifactLifecycle.QUEUED, handle=None
        )

        retrieved = self.manager.get_workpiece_view_handle(
            ArtifactKey.for_view(WP1_UID), 0
        )

        self.assertIsNone(retrieved)

    def test_get_workpiece_view_handle_returns_none_when_missing(self):
        """Test get_workpiece_view_handle returns None when not found."""
        retrieved = self.manager.get_workpiece_view_handle(
            ArtifactKey.for_view(WP1_UID), 0
        )

        self.assertIsNone(retrieved)

    def test_is_view_stale_returns_true_for_missing_entry(self):
        """Test is_view_stale returns True when entry is missing."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp_h.is_scalable = True
        wp_h.source_coordinate_system_name = "wcs"
        wp_h.generation_size = 100
        wp_h.source_dimensions = (10, 10)

        view_key = ArtifactKey.for_view(WP1_UID)
        result = self.manager.is_view_stale(view_key, None, wp_h, 0)

        self.assertTrue(result)

    def test_is_view_stale_returns_true_for_not_ready_state(self):
        """Test is_view_stale returns True when entry is QUEUED."""
        view_composite = make_composite_key(ArtifactKey.for_view(WP1_UID), 0)
        self.manager._ledger[view_composite] = Mock(
            state=ArtifactLifecycle.QUEUED,
            handle=None,
            metadata={},
        )

        result = self.manager.is_view_stale(
            ArtifactKey.for_view(WP1_UID), None, None, 0
        )

        self.assertTrue(result)

    def test_is_view_stale_returns_true_for_context_mismatch(self):
        """Test is_view_stale returns True when render context changes."""
        view_h = create_mock_handle(WorkPieceViewArtifactHandle, "view1")
        new_context = RenderContext(
            pixels_per_mm=(1.0, 1.0),
            show_travel_moves=True,
            margin_px=10,
            color_set_dict={"default": {}},
        )
        stored_context = RenderContext(
            pixels_per_mm=(0.5, 0.5),
            show_travel_moves=False,
            margin_px=5,
            color_set_dict={"default": {}},
        )
        view_key = ArtifactKey.for_view(WP1_UID)
        view_composite = make_composite_key(ArtifactKey.for_view(WP1_UID), 0)
        self.manager._ledger[view_composite] = Mock(
            state=ArtifactLifecycle.DONE,
            handle=view_h,
            metadata={"render_context": stored_context},
        )

        result = self.manager.is_view_stale(view_key, new_context, None, 0)

        self.assertTrue(result)

    def test_is_view_stale_returns_true_for_property_mismatch(self):
        """Test is_view_stale returns True when source properties change."""
        view_h = create_mock_handle(WorkPieceViewArtifactHandle, "view1")
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp_h.is_scalable = True
        wp_h.source_coordinate_system_name = "wcs"
        wp_h.generation_size = 100
        wp_h.source_dimensions = (10, 10)

        stored_props = {
            "is_scalable": False,
            "source_coordinate_system_name": "old",
            "generation_size": 50,
            "source_dimensions": (5, 5),
        }
        view_composite = make_composite_key(ArtifactKey.for_view(WP1_UID), 0)
        self.manager._ledger[view_composite] = Mock(
            state=ArtifactLifecycle.DONE,
            handle=view_h,
            metadata={"source_properties": stored_props},
        )

        view_key = ArtifactKey.for_view(WP1_UID)
        result = self.manager.is_view_stale(view_key, None, wp_h, 0)

        self.assertTrue(result)

    def test_is_view_stale_returns_false_for_valid_view(self):
        """Test is_view_stale returns False when view is valid."""
        view_h = create_mock_handle(WorkPieceViewArtifactHandle, "view1")
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp_h.is_scalable = True
        wp_h.source_coordinate_system_name = "wcs"
        wp_h.generation_size = 100
        wp_h.source_dimensions = (10, 10)

        context = RenderContext(
            pixels_per_mm=(1.0, 1.0),
            show_travel_moves=True,
            margin_px=10,
            color_set_dict={"default": {}},
        )
        stored_props = {
            "is_scalable": True,
            "source_coordinate_system_name": "wcs",
            "generation_size": 100,
            "source_dimensions": (10, 10),
        }
        view_composite = make_composite_key(ArtifactKey.for_view(WP1_UID), 0)
        self.manager._ledger[view_composite] = Mock(
            state=ArtifactLifecycle.DONE,
            handle=view_h,
            metadata={
                "render_context": context,
                "source_properties": stored_props,
            },
        )

        result = self.manager.is_view_stale(
            ArtifactKey.for_view(WP1_UID), context, wp_h, 0
        )

        self.assertFalse(result)

    def test_get_artifact_retrieves_from_store(self):
        """Test get_artifact retrieves artifact from store."""
        handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        artifact = Mock()
        self.mock_store.get.return_value = artifact

        result = self.manager.get_artifact(handle)

        self.assertIs(result, artifact)
        self.mock_store.get.assert_called_once_with(handle)

    def test_put_step_render_handle_stores_and_releases_old(self):
        """Test put_step_render_handle stores and releases old handle."""
        old_h = create_mock_handle(StepRenderArtifactHandle, "old")
        new_h = create_mock_handle(StepRenderArtifactHandle, "new")
        self.manager._step_render_handles[STEP1_UID] = old_h

        self.manager.put_step_render_handle(STEP1_UID, new_h)

        self.assertIs(self.manager._step_render_handles[STEP1_UID], new_h)
        self.mock_release.assert_called_once_with(old_h)

    def test_put_step_render_handle_without_old(self):
        """Test put_step_render_handle stores without old handle."""
        new_h = create_mock_handle(StepRenderArtifactHandle, "new")

        self.manager.put_step_render_handle(STEP1_UID, new_h)

        self.assertIs(self.manager._step_render_handles[STEP1_UID], new_h)
        self.mock_release.assert_not_called()

    def test_put_step_ops_handle_creates_processing_entry(self):
        """Test put_step_ops_handle creates PROCESSING entry."""
        handle = create_mock_handle(StepOpsArtifactHandle, "ops")
        step_key = ArtifactKey.for_step(STEP1_UID)

        self.manager.put_step_ops_handle(step_key, handle, 1)

        composite_key = make_composite_key(step_key, 1)
        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertEqual(entry.state, ArtifactLifecycle.PROCESSING)
        self.assertIs(entry.handle, handle)
        self.assertEqual(entry.generation_id, 1)

    def test_put_step_ops_handle_replaces_old_handle(self):
        """Test put_step_ops_handle releases old handle."""
        old_h = create_mock_handle(StepOpsArtifactHandle, "old")
        new_h = create_mock_handle(StepOpsArtifactHandle, "new")
        step_key = ArtifactKey.for_step(STEP1_UID)
        composite_key = make_composite_key(step_key, 1)
        self.manager._ledger[composite_key] = Mock(
            state=ArtifactLifecycle.DONE, handle=old_h
        )

        self.manager.put_step_ops_handle(step_key, new_h, 1)

        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertIs(entry.handle, new_h)
        self.mock_release.assert_called_once_with(old_h)

    def test_put_workpiece_view_handle_creates_processing_entry(self):
        """Test put_workpiece_view_handle creates PROCESSING entry."""
        handle = create_mock_handle(WorkPieceViewArtifactHandle, "view")
        view_key = ArtifactKey.for_view(WP1_UID)

        self.manager.put_workpiece_view_handle(view_key, handle, 0)

        composite_key = make_composite_key(view_key, 0)
        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertEqual(entry.state, ArtifactLifecycle.PROCESSING)
        self.assertIs(entry.handle, handle)
        self.assertEqual(entry.generation_id, 0)

    def test_put_workpiece_view_handle_replaces_old_handle(self):
        """Test put_workpiece_view_handle releases old handle."""
        old_h = create_mock_handle(WorkPieceViewArtifactHandle, "old")
        new_h = create_mock_handle(WorkPieceViewArtifactHandle, "new")
        view_key = ArtifactKey.for_view(WP1_UID)
        composite_key = make_composite_key(view_key, 0)
        self.manager._ledger[composite_key] = Mock(
            state=ArtifactLifecycle.DONE, handle=old_h
        )

        self.manager.put_workpiece_view_handle(view_key, new_h, 0)

        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertIs(entry.handle, new_h)
        self.mock_release.assert_called_once_with(old_h)

    def test_adopt_artifact_creates_and_adopts_handle(self):
        """Test adopt_artifact deserializes and adopts handle."""
        handle_dict = {
            "type": "workpiece",
            "shm_name": "test_shm",
            "other_field": "value",
        }
        created_handle = create_mock_handle(WorkPieceArtifactHandle, "adopted")

        with unittest.mock.patch(
            "rayforge.pipeline.artifact.manager.create_handle_from_dict",
            return_value=created_handle,
        ):
            result = self.manager.adopt_artifact(
                ArtifactKey.for_workpiece(WP1_UID), handle_dict
            )

        self.assertIs(result, created_handle)
        self.mock_store.adopt.assert_called_once_with(created_handle)

    def test_retain_handle_calls_store_retain(self):
        """Test retain_handle delegates to store.retain."""
        handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")

        self.manager.retain_handle(handle)

        self.mock_store.retain.assert_called_once_with(handle)

    def test_get_all_workpiece_keys_for_generation(self):
        """Test getting workpiece keys for specific generation."""
        wp1_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP1_UID), 0
        )
        wp2_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP2_UID), 1
        )
        wp3_composite = make_composite_key(
            ArtifactKey.for_workpiece(WP3_UID), 0
        )
        self.manager._ledger[wp1_composite] = Mock()
        self.manager._ledger[wp2_composite] = Mock()
        self.manager._ledger[wp3_composite] = Mock()

        keys = self.manager.get_all_workpiece_keys_for_generation(0)

        self.assertEqual(len(keys), 2)
        self.assertIn(ArtifactKey.for_workpiece(WP1_UID), keys)
        self.assertIn(ArtifactKey.for_workpiece(WP3_UID), keys)
        self.assertNotIn(ArtifactKey.for_workpiece(WP2_UID), keys)

    def test_invalidate_for_job_releases_and_invalidates(self):
        """Test invalidate_for_job releases handle and invalidates."""
        job_h = create_mock_handle(JobArtifactHandle, "job")
        job_key = ArtifactKey(id=JOB_UID, group="job")
        job_composite = make_composite_key(job_key, 0)
        self.manager._ledger[job_composite] = Mock(
            state=ArtifactLifecycle.DONE, handle=job_h
        )

        self.manager.invalidate_for_job(job_key)

        entry = self.manager._ledger.get(job_composite)
        assert entry is not None
        self.assertEqual(entry.state, ArtifactLifecycle.STALE)
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

    def test_mark_done_transitions_from_initial(self):
        """Test mark_done transitions from QUEUED to DONE."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        key = ArtifactKey.for_workpiece(WP1_UID)
        composite_key = make_composite_key(key, 0)
        self.manager._ledger[composite_key] = Mock(
            state=ArtifactLifecycle.QUEUED, handle=wp_h
        )

        self.manager.mark_done(key, 0)

        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertEqual(entry.state, ArtifactLifecycle.DONE)
        self.assertIsNone(entry.handle)
        self.mock_release.assert_called_once_with(wp_h)

    def test_mark_done_without_handle(self):
        """Test mark_done works when entry has no handle."""
        key = ArtifactKey.for_workpiece(WP1_UID)
        composite_key = make_composite_key(key, 0)
        self.manager._ledger[composite_key] = Mock(
            state=ArtifactLifecycle.QUEUED, handle=None
        )

        self.manager.mark_done(key, 0)

        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertEqual(entry.state, ArtifactLifecycle.DONE)
        self.mock_release.assert_not_called()

    def test_mark_done_nonexistent_skips(self):
        """Test mark_done skips when entry does not exist."""
        key = ArtifactKey.for_workpiece(WP1_UID)

        self.manager.mark_done(key, 0)

        self.assertEqual(len(self.manager._ledger), 0)

    def test_mark_done_from_non_initial_raises_assertion(self):
        """Test mark_done raises AssertionError from non-QUEUED state."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        key = ArtifactKey.for_workpiece(WP1_UID)
        composite_key = make_composite_key(key, 0)
        self.manager._ledger[composite_key] = Mock(
            state=ArtifactLifecycle.PROCESSING, handle=wp_h
        )

        with self.assertRaises(AssertionError) as cm:
            self.manager.mark_done(key, 0)

        self.assertIn("must be QUEUED", str(cm.exception))

    def test_complete_generation_marks_existing_entry_done(self):
        """Test complete_generation marks existing entry as DONE."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        key = ArtifactKey.for_workpiece(WP1_UID)
        composite_key = make_composite_key(key, 0)
        self.manager._ledger[composite_key] = Mock(
            state=ArtifactLifecycle.PROCESSING, handle=wp_h
        )

        self.manager.complete_generation(key, 0)

        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertEqual(entry.state, ArtifactLifecycle.DONE)
        self.assertIs(entry.handle, wp_h)

    def test_complete_generation_creates_new_entry_with_handle(self):
        """Test complete_generation creates new entry with handle."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        key = ArtifactKey.for_workpiece(WP1_UID)

        self.manager.complete_generation(key, 0, wp_h)

        composite_key = make_composite_key(key, 0)
        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertEqual(entry.state, ArtifactLifecycle.DONE)
        self.assertIs(entry.handle, wp_h)

    def test_complete_generation_replaces_handle(self):
        """Test complete_generation replaces existing handle."""
        old_h = create_mock_handle(WorkPieceArtifactHandle, "old")
        new_h = create_mock_handle(WorkPieceArtifactHandle, "new")
        key = ArtifactKey.for_workpiece(WP1_UID)
        composite_key = make_composite_key(key, 0)
        self.manager._ledger[composite_key] = Mock(
            state=ArtifactLifecycle.PROCESSING, handle=old_h
        )

        self.manager.complete_generation(key, 0, new_h)

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
        self.assertEqual(entry.state, ArtifactLifecycle.DONE)
        self.assertIsNone(entry.handle)

    def test_complete_generation_entry_without_handle_marks_done(self):
        """Test complete_generation marks done when entry has no handle."""
        key = ArtifactKey.for_workpiece(WP1_UID)
        composite_key = make_composite_key(key, 0)
        self.manager._ledger[composite_key] = Mock(
            state=ArtifactLifecycle.PROCESSING, handle=None
        )

        self.manager.complete_generation(key, 0)

        entry = self.manager._ledger.get(composite_key)
        assert entry is not None
        self.assertEqual(entry.state, ArtifactLifecycle.DONE)

    def test_declare_generation_creates_initial_entries(self):
        """Test declare_generation creates QUEUED entries for new keys."""
        wp1_key = ArtifactKey.for_workpiece(WP1_UID)
        wp2_key = ArtifactKey.for_workpiece(WP2_UID)

        self.manager.declare_generation({wp1_key, wp2_key}, 0)

        wp1_composite = make_composite_key(wp1_key, 0)
        wp2_composite = make_composite_key(wp2_key, 0)
        entry1 = self.manager._ledger.get(wp1_composite)
        entry2 = self.manager._ledger.get(wp2_composite)
        assert entry1 is not None
        assert entry2 is not None
        self.assertEqual(entry1.state, ArtifactLifecycle.QUEUED)
        self.assertEqual(entry2.state, ArtifactLifecycle.QUEUED)

    def test_declare_generation_skips_existing_entries(self):
        """Test declare_generation skips existing entries."""
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        wp_composite = make_composite_key(wp_key, 0)
        self.manager._ledger[wp_composite] = Mock(
            state=ArtifactLifecycle.PROCESSING
        )

        self.manager.declare_generation({wp_key}, 0)

        entry = self.manager._ledger.get(wp_composite)
        assert entry is not None
        self.assertEqual(entry.state, ArtifactLifecycle.PROCESSING)

    def test_declare_generation_does_not_modify_previous_generations(self):
        """Test declare_generation does not modify previous generations."""
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        wp_composite_g0 = make_composite_key(wp_key, 0)
        self.manager._ledger[wp_composite_g0] = Mock(
            state=ArtifactLifecycle.DONE
        )

        self.manager.declare_generation({wp_key}, 1)

        wp_composite_g1 = make_composite_key(wp_key, 1)
        entry_g0 = self.manager._ledger.get(wp_composite_g0)
        entry_g1 = self.manager._ledger.get(wp_composite_g1)
        assert entry_g0 is not None
        assert entry_g1 is not None
        self.assertEqual(entry_g0.state, ArtifactLifecycle.DONE)
        self.assertEqual(entry_g1.state, ArtifactLifecycle.QUEUED)

    def test_is_finished_returns_true_for_empty_ledger(self):
        """Test is_finished returns True when ledger is empty."""
        self.assertTrue(self.manager.is_finished())

    def test_is_finished_returns_true_for_all_done(self):
        """Test is_finished returns True when all entries are DONE."""
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        wp_composite = make_composite_key(wp_key, 0)
        self.manager._ledger[wp_composite] = Mock(state=ArtifactLifecycle.DONE)

        self.assertTrue(self.manager.is_finished())

    def test_is_finished_returns_false_for_initial(self):
        """Test is_finished returns False when any entry is QUEUED."""
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        wp_composite = make_composite_key(wp_key, 0)
        self.manager._ledger[wp_composite] = Mock(
            state=ArtifactLifecycle.QUEUED
        )

        self.assertFalse(self.manager.is_finished())

    def test_is_finished_returns_false_for_processing(self):
        """Test is_finished returns False when any entry is PROCESSING."""
        wp_key = ArtifactKey.for_workpiece(WP1_UID)
        wp_composite = make_composite_key(wp_key, 0)
        self.manager._ledger[wp_composite] = Mock(
            state=ArtifactLifecycle.PROCESSING
        )

        self.assertFalse(self.manager.is_finished())

    def test_is_finished_returns_true_for_stale_or_error(self):
        """Test is_finished returns True for STALE or ERROR states."""
        wp1_key = ArtifactKey.for_workpiece(WP1_UID)
        wp2_key = ArtifactKey.for_workpiece(WP2_UID)
        wp1_composite = make_composite_key(wp1_key, 0)
        wp2_composite = make_composite_key(wp2_key, 0)
        self.manager._ledger[wp1_composite] = Mock(
            state=ArtifactLifecycle.STALE
        )
        self.manager._ledger[wp2_composite] = Mock(
            state=ArtifactLifecycle.ERROR
        )

        self.assertTrue(self.manager.is_finished())

import unittest
from unittest.mock import Mock
from rayforge.pipeline.artifact import (
    ArtifactManager,
    WorkPieceArtifactHandle,
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    JobArtifactHandle,
)
from rayforge.pipeline.artifact.lifecycle import ArtifactLifecycle
from rayforge.pipeline.artifact.store import ArtifactStore


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

        self.manager.put_step_render_handle("step1", render_handle)
        ledger_key = ("step", "step1")
        self.manager._ledger[ledger_key] = Mock(
            state=ArtifactLifecycle.READY, handle=ops_handle
        )

        retrieved_render = self.manager.get_step_render_handle("step1")
        retrieved_ops = self.manager.get_step_ops_handle("step1")

        self.assertIs(retrieved_render, render_handle)
        self.assertIs(retrieved_ops, ops_handle)
        self.mock_release.assert_not_called()

    def test_put_and_get_job(self):
        """Tests basic storage and retrieval of a job handle."""
        handle = create_mock_handle(JobArtifactHandle, "job")
        self.manager._ledger[self.manager.JOB_KEY] = Mock(
            state=ArtifactLifecycle.READY, handle=handle
        )
        retrieved = self.manager.get_job_handle()
        self.assertIs(retrieved, handle)
        self.mock_release.assert_not_called()

    def test_invalidate_workpiece_cascades_correctly(self):
        """
        Tests that invalidating a workpiece only cascades up to the ops
        artifact and job, leaving the render artifact intact.
        """
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        ledger_key = ("workpiece", "step1", "wp1")
        self.manager._ledger[ledger_key] = Mock(
            state=ArtifactLifecycle.READY, handle=wp_h
        )
        self.manager._step_render_handles["step1"] = render_h
        ledger_key = ("step", "step1")
        self.manager._ledger[ledger_key] = Mock(
            state=ArtifactLifecycle.READY, handle=ops_h
        )
        self.manager._ledger[self.manager.JOB_KEY] = Mock(
            state=ArtifactLifecycle.READY, handle=job_h
        )

        self.manager.invalidate_for_workpiece("step1", "wp1")

        # Assert correct artifacts were removed
        self.assertIsNone(self.manager.get_workpiece_handle("step1", "wp1"))
        self.assertIsNone(self.manager.get_step_ops_handle("step1"))
        self.assertIsNone(self.manager.get_job_handle())
        # Assert render handle remains for UI stability
        self.assertIs(self.manager.get_step_render_handle("step1"), render_h)
        # Assert only workpiece and ops were released
        self.mock_release.assert_any_call(wp_h)
        self.mock_release.assert_any_call(ops_h)
        self.mock_release.assert_any_call(job_h)
        self.assertEqual(self.mock_release.call_count, 3)

    def test_invalidate_step_cascades_correctly(self):
        """
        Tests that invalidating a step cascades to all dependent
        artifacts (workpieces, step artifacts, and job).
        """
        wp1_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        wp2_h = create_mock_handle(WorkPieceArtifactHandle, "wp2")
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        self.manager._ledger[("workpiece", "step1", "wp1")] = Mock(
            state=ArtifactLifecycle.READY, handle=wp1_h
        )
        self.manager._ledger[("workpiece", "step1", "wp2")] = Mock(
            state=ArtifactLifecycle.READY, handle=wp2_h
        )
        self.manager._step_render_handles["step1"] = render_h
        ledger_key = ("step", "step1")
        self.manager._ledger[ledger_key] = Mock(
            state=ArtifactLifecycle.READY, handle=ops_h
        )
        self.manager._ledger[self.manager.JOB_KEY] = Mock(
            state=ArtifactLifecycle.READY, handle=job_h
        )

        self.manager.invalidate_for_step("step1")

        # Assert all step-related artifacts were removed
        self.assertIsNone(self.manager.get_workpiece_handle("step1", "wp1"))
        self.assertIsNone(self.manager.get_workpiece_handle("step1", "wp2"))
        self.assertIsNone(self.manager.get_step_render_handle("step1"))
        self.assertIsNone(self.manager.get_step_ops_handle("step1"))
        self.assertIsNone(self.manager.get_job_handle())
        # Assert all artifacts were released
        self.mock_release.assert_any_call(wp1_h)
        self.mock_release.assert_any_call(wp2_h)
        self.mock_release.assert_any_call(render_h)
        self.mock_release.assert_any_call(ops_h)
        self.mock_release.assert_any_call(job_h)
        self.assertEqual(self.mock_release.call_count, 5)

    def test_put_job_replaces_old_handle(self):
        """Tests that putting a job handle releases of old handle."""
        old_job_h = create_mock_handle(JobArtifactHandle, "job_old")
        new_job_h = create_mock_handle(JobArtifactHandle, "job_new")

        self.manager._ledger[self.manager.JOB_KEY] = Mock(
            state=ArtifactLifecycle.READY, handle=old_job_h
        )
        self.manager._ledger[self.manager.JOB_KEY] = Mock(
            state=ArtifactLifecycle.READY, handle=new_job_h
        )
        self.manager.release_handle(old_job_h)

        self.assertIs(self.manager.get_job_handle(), new_job_h)
        self.mock_release.assert_called_once_with(old_job_h)

    def test_shutdown_releases_all_artifacts(self):
        """Tests that shutdown releases all cached artifacts."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        ledger_key = ("workpiece", "step1", "wp1")
        self.manager._ledger[ledger_key] = Mock(
            state=ArtifactLifecycle.READY, handle=wp_h
        )
        self.manager._step_render_handles["step1"] = render_h
        ledger_key = ("step", "step1")
        self.manager._ledger[ledger_key] = Mock(
            state=ArtifactLifecycle.READY, handle=ops_h
        )
        self.manager._ledger[self.manager.JOB_KEY] = Mock(
            state=ArtifactLifecycle.READY, handle=job_h
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
        self.manager._ledger[("workpiece", "step1", "wp1")] = Mock()
        self.manager._ledger[("workpiece", "step1", "wp2")] = Mock()
        self.manager._ledger[("workpiece", "step2", "wp1")] = Mock()

        keys = self.manager.get_all_workpiece_keys()
        self.assertEqual(len(keys), 3)
        self.assertIn(("step1", "wp1"), keys)
        self.assertIn(("step1", "wp2"), keys)
        self.assertIn(("step2", "wp1"), keys)

    def test_get_all_step_render_uids(self):
        """Tests getting all step render UIDs."""
        self.manager._step_render_handles["step1"] = Mock()
        self.manager._step_render_handles["step2"] = Mock()

        uids = self.manager.get_all_step_render_uids()
        self.assertEqual(len(uids), 2)
        self.assertIn("step1", uids)
        self.assertIn("step2", uids)

    def test_has_step_render_handle(self):
        """Tests checking if a step render handle exists."""
        self.manager._step_render_handles["step1"] = Mock()
        self.assertTrue(self.manager.has_step_render_handle("step1"))
        self.assertFalse(self.manager.has_step_render_handle("step2"))

    def test_pop_step_render_handle(self):
        """Tests popping a step render handle."""
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        self.manager._step_render_handles["step1"] = render_h

        popped = self.manager.pop_step_render_handle("step1")
        self.assertIs(popped, render_h)
        self.assertIsNone(self.manager.get_step_render_handle("step1"))

    def test_checkout_step_render_handle(self):
        """Tests checking out a step render handle."""
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        self.manager._step_render_handles["step1"] = render_h

        with self.manager.checkout("step1") as handle:
            self.assertIs(handle, render_h)
            self.assertEqual(self.manager._ref_counts["step1"], 1)
            self.mock_store.retain.assert_called_once_with(render_h)

        self.mock_store.release.assert_called_once_with(render_h)
        self.assertNotIn("step1", self.manager._ref_counts)

    def test_checkout_job_handle(self):
        """Tests checking out job handle."""
        job_h = create_mock_handle(JobArtifactHandle, "job")
        self.manager._ledger[ArtifactManager.JOB_KEY] = Mock(
            state=ArtifactLifecycle.READY, handle=job_h
        )

        with self.manager.checkout(ArtifactManager.JOB_KEY) as handle:
            self.assertIs(handle, job_h)
            self.assertEqual(
                self.manager._ref_counts[ArtifactManager.JOB_KEY], 1
            )
            self.mock_store.retain.assert_called_once_with(job_h)

        self.mock_store.release.assert_called_once_with(job_h)
        self.assertNotIn(ArtifactManager.JOB_KEY, self.manager._ref_counts)

    def test_get_workpiece_handle_from_ledger(self):
        """Tests getting workpiece handle from ledger."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        ledger_key = ("workpiece", "step1", "wp1")
        self.manager._ledger[ledger_key] = Mock(
            state=ArtifactLifecycle.READY, handle=wp_h
        )

        retrieved = self.manager.get_workpiece_handle("step1", "wp1")
        self.assertIs(retrieved, wp_h)

    def test_get_workpiece_handle_returns_none_when_not_ready(self):
        """Tests getting workpiece handle returns None when not READY."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        ledger_key = ("workpiece", "step1", "wp1")
        self.manager._ledger[ledger_key] = Mock(
            state=ArtifactLifecycle.PENDING, handle=wp_h
        )

        retrieved = self.manager.get_workpiece_handle("step1", "wp1")
        self.assertIsNone(retrieved)

    def test_get_workpiece_handle_returns_none_when_not_found(self):
        """Tests getting workpiece handle returns None when not found."""
        retrieved = self.manager.get_workpiece_handle("step1", "wp1")
        self.assertIsNone(retrieved)

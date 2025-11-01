import unittest
from unittest.mock import Mock, patch
from rayforge.pipeline.artifact import (
    ArtifactCache,
    WorkPieceArtifactHandle,
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    JobArtifactHandle,
)


def create_mock_handle(handle_class, name: str) -> Mock:
    """Creates a mock handle that behaves like a real handle for tests."""
    handle = Mock(spec=handle_class)
    handle.shm_name = f"shm_{name}"
    return handle


class TestArtifactCache(unittest.TestCase):
    """Test suite for the ArtifactCache."""

    def setUp(self):
        """Set up a fresh cache and mock the context-based dependency."""
        # Patch the `get_context` function in the module where it is used.
        self.mock_get_context_patch = patch(
            "rayforge.pipeline.artifact.cache.get_context"
        )
        mock_get_context = self.mock_get_context_patch.start()

        # Configure the mock context to return a mock artifact_store.
        mock_context = Mock()
        mock_artifact_store = Mock()
        self.mock_release = mock_artifact_store.release
        mock_context.artifact_store = mock_artifact_store
        mock_get_context.return_value = mock_context

        self.cache = ArtifactCache()

    def tearDown(self):
        """Stop the patcher after each test."""
        self.mock_get_context_patch.stop()

    def test_put_and_get_workpiece(self):
        """Tests basic storage and retrieval of a workpiece handle."""
        handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        self.cache.put_workpiece_handle("step1", "wp1", handle)
        retrieved = self.cache.get_workpiece_handle("step1", "wp1")
        self.assertIs(retrieved, handle)
        self.mock_release.assert_not_called()

    def test_put_and_get_step_handles(self):
        """Tests storage and retrieval of step handle types."""
        render_handle = create_mock_handle(
            StepRenderArtifactHandle, "step1_render"
        )
        ops_handle = create_mock_handle(StepOpsArtifactHandle, "step1_ops")

        self.cache.put_step_render_handle("step1", render_handle)
        self.cache.put_step_ops_handle("step1", ops_handle)

        retrieved_render = self.cache.get_step_render_handle("step1")
        retrieved_ops = self.cache.get_step_ops_handle("step1")

        self.assertIs(retrieved_render, render_handle)
        self.assertIs(retrieved_ops, ops_handle)
        self.mock_release.assert_not_called()

    def test_put_and_get_job(self):
        """Tests basic storage and retrieval of a job handle."""
        handle = create_mock_handle(JobArtifactHandle, "job")
        self.cache.put_job_handle(handle)
        retrieved = self.cache.get_job_handle()
        self.assertIs(retrieved, handle)
        self.mock_release.assert_not_called()

    def test_get_non_existent_handle_returns_none(self):
        """Tests that getting a non-existent handle safely returns None."""
        self.assertIsNone(self.cache.get_workpiece_handle("s1", "wp1"))
        self.assertIsNone(self.cache.get_step_render_handle("s1"))
        self.assertIsNone(self.cache.get_step_ops_handle("s1"))
        self.assertIsNone(self.cache.get_job_handle())

    def test_put_replaces_and_releases_old_handle(self):
        """
        Tests that putting a handle for an existing key replaces the old
        handle and releases its resources.
        """
        old_wp = create_mock_handle(WorkPieceArtifactHandle, "wp_old")
        new_wp = create_mock_handle(WorkPieceArtifactHandle, "wp_new")
        self.cache.put_workpiece_handle("s1", "wp1", old_wp)
        self.cache.put_workpiece_handle("s1", "wp1", new_wp)
        self.mock_release.assert_any_call(old_wp)
        self.assertIs(self.cache.get_workpiece_handle("s1", "wp1"), new_wp)

        old_render = create_mock_handle(StepRenderArtifactHandle, "render_old")
        new_render = create_mock_handle(StepRenderArtifactHandle, "render_new")
        self.cache.put_step_render_handle("s1", old_render)
        self.cache.put_step_render_handle("s1", new_render)
        self.mock_release.assert_any_call(old_render)
        self.assertIs(self.cache.get_step_render_handle("s1"), new_render)

        old_job = create_mock_handle(JobArtifactHandle, "job_old")
        new_job = create_mock_handle(JobArtifactHandle, "job_new")
        self.cache.put_job_handle(old_job)
        self.cache.put_job_handle(new_job)
        self.mock_release.assert_any_call(old_job)
        self.assertIs(self.cache.get_job_handle(), new_job)

    def test_invalidate_for_workpiece_is_isolated(self):
        """
        Tests that invalidating a workpiece directly only removes itself
        and does NOT cascade.
        """
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        # Manually populate cache to bypass put methods' invalidation logic
        self.cache._workpiece_handles[("step1", "wp1")] = wp_h
        self.cache._step_render_handles["step1"] = render_h
        self.cache._step_ops_handles["step1"] = ops_h
        self.cache._job_handle = job_h

        self.cache.invalidate_for_workpiece("step1", "wp1")

        # Assert only the workpiece was removed
        self.assertIsNone(self.cache.get_workpiece_handle("step1", "wp1"))
        self.assertIsNotNone(self.cache.get_step_render_handle("step1"))
        self.assertIsNotNone(self.cache.get_step_ops_handle("step1"))
        self.assertIsNotNone(self.cache.get_job_handle())

        # Assert release was called only for the workpiece handle
        self.mock_release.assert_called_once_with(wp_h)

    def test_invalidate_for_step_cascades_correctly(self):
        """Tests that invalidating a step removes all its related artifacts."""
        wp1_h = create_mock_handle(WorkPieceArtifactHandle, "s1_wp1")
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        # Populate cache
        self.cache._workpiece_handles[("step1", "wp1")] = wp1_h
        self.cache._step_render_handles["step1"] = render_h
        self.cache._step_ops_handles["step1"] = ops_h
        self.cache._job_handle = job_h

        self.cache.invalidate_for_step("step1")

        # Assert all items were removed
        self.assertIsNone(self.cache.get_workpiece_handle("step1", "wp1"))
        self.assertIsNone(self.cache.get_step_render_handle("step1"))
        self.assertIsNone(self.cache.get_step_ops_handle("step1"))
        self.assertIsNone(self.cache.get_job_handle())

        # 1 workpiece + 2 step handles + 1 job = 4 releases
        self.assertEqual(self.mock_release.call_count, 4)
        self.mock_release.assert_any_call(wp1_h)
        self.mock_release.assert_any_call(render_h)
        self.mock_release.assert_any_call(ops_h)
        self.mock_release.assert_any_call(job_h)

    def test_invalidation_on_non_existent_keys_is_safe(self):
        """Tests that invalidating non-existent keys does not raise errors."""
        try:
            self.cache.invalidate_for_workpiece("s_bad", "wp_bad")
            self.cache.invalidate_for_step("s_bad")
            self.cache.invalidate_for_job()
        except Exception as e:
            self.fail(f"Invalidation on non-existent keys raised {e}")

        # No handles were ever added, so release should not be called
        self.mock_release.assert_not_called()

    def test_put_workpiece_invalidates_ops_and_job(self):
        """Putting a new workpiece handle should invalidate ops/job."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        # Pre-populate step and job
        self.cache._step_ops_handles["step1"] = ops_h
        self.cache._job_handle = job_h

        self.cache.put_workpiece_handle("step1", "wp1", wp_h)

        # Ops and job should be gone
        self.assertIsNone(self.cache.get_step_ops_handle("step1"))
        self.assertIsNone(self.cache.get_job_handle())

        # The new workpiece handle should exist
        self.assertIsNotNone(self.cache.get_workpiece_handle("step1", "wp1"))

        # Assert old handles were released
        self.assertEqual(self.mock_release.call_count, 2)
        self.mock_release.assert_any_call(ops_h)
        self.mock_release.assert_any_call(job_h)

    def test_put_step_ops_invalidates_job(self):
        """Putting a new step ops handle should invalidate the job."""
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        # Pre-populate the job
        self.cache._job_handle = job_h

        self.cache.put_step_ops_handle("step1", ops_h)

        # Job should be gone
        self.assertIsNone(self.cache.get_job_handle())
        # The new ops handle should exist
        self.assertIsNotNone(self.cache.get_step_ops_handle("step1"))
        # The old job handle should be released
        self.mock_release.assert_called_once_with(job_h)

    def test_shutdown_clears_all_and_releases(self):
        """Tests that shutdown releases all stored handles."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        render_h = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        ops_h = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        self.cache._workpiece_handles[("step1", "wp1")] = wp_h
        self.cache._step_render_handles["step1"] = render_h
        self.cache._step_ops_handles["step1"] = ops_h
        self.cache._job_handle = job_h

        self.cache.shutdown()

        self.assertIsNone(self.cache.get_workpiece_handle("step1", "wp1"))
        self.assertIsNone(self.cache.get_step_render_handle("step1"))
        self.assertIsNone(self.cache.get_step_ops_handle("step1"))
        self.assertIsNone(self.cache.get_job_handle())
        self.assertEqual(len(self.cache._workpiece_handles), 0)
        self.assertEqual(len(self.cache._step_render_handles), 0)
        self.assertEqual(len(self.cache._step_ops_handles), 0)

        self.assertEqual(self.mock_release.call_count, 4)
        self.mock_release.assert_any_call(wp_h)
        self.mock_release.assert_any_call(render_h)
        self.mock_release.assert_any_call(ops_h)
        self.mock_release.assert_any_call(job_h)

    def test_has_step_render_handle(self):
        """Tests the has_step_render_handle method."""
        self.assertFalse(self.cache.has_step_render_handle("step1"))
        handle = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        self.cache.put_step_render_handle("step1", handle)
        self.assertTrue(self.cache.has_step_render_handle("step1"))
        self.cache.invalidate_for_step("step1")
        self.assertFalse(self.cache.has_step_render_handle("step1"))

    def test_get_all_step_render_uids(self):
        """Tests getting all step UIDs with render handles."""
        self.assertEqual(self.cache.get_all_step_render_uids(), set())
        h1 = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        h2 = create_mock_handle(StepRenderArtifactHandle, "step2_render")
        self.cache.put_step_render_handle("step1", h1)
        self.cache.put_step_render_handle("step2", h2)
        self.assertEqual(
            self.cache.get_all_step_render_uids(), {"step1", "step2"}
        )

    def test_get_all_workpiece_keys(self):
        """Tests getting all workpiece keys."""
        self.assertEqual(self.cache.get_all_workpiece_keys(), set())
        h1 = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        h2 = create_mock_handle(WorkPieceArtifactHandle, "wp2")
        self.cache.put_workpiece_handle("step1", "wp1", h1)
        self.cache.put_workpiece_handle("step2", "wp2", h2)
        expected_keys = {("step1", "wp1"), ("step2", "wp2")}
        self.assertEqual(self.cache.get_all_workpiece_keys(), expected_keys)

    def test_pop_step_ops_handle(self):
        """Tests popping a step ops handle from the cache."""
        handle = create_mock_handle(StepOpsArtifactHandle, "step1_ops")
        self.cache.put_step_ops_handle("step1", handle)

        # Pop should return the handle and remove it
        popped_handle = self.cache.pop_step_ops_handle("step1")
        self.assertIs(popped_handle, handle)
        self.assertIsNone(self.cache.get_step_ops_handle("step1"))

        # Popping again should return None
        self.assertIsNone(self.cache.pop_step_ops_handle("step1"))

        # Pop should not trigger a release
        self.mock_release.assert_not_called()

    def test_pop_step_render_handle(self):
        """Tests popping a step render handle from the cache."""
        handle = create_mock_handle(StepRenderArtifactHandle, "step1_render")
        self.cache.put_step_render_handle("step1", handle)

        # Pop should return the handle and remove it
        popped_handle = self.cache.pop_step_render_handle("step1")
        self.assertIs(popped_handle, handle)
        self.assertIsNone(self.cache.get_step_render_handle("step1"))

        # Popping again should return None
        self.assertIsNone(self.cache.pop_step_render_handle("step1"))

        # Pop should not trigger a release
        self.mock_release.assert_not_called()


if __name__ == "__main__":
    unittest.main()

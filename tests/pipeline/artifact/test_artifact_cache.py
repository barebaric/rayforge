import unittest
from unittest.mock import Mock, patch
from rayforge.pipeline.artifact import ArtifactCache
from rayforge.pipeline.artifact import (
    WorkPieceArtifactHandle,
    StepArtifactHandle,
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
        """Set up a fresh cache and mock for ArtifactStore for each test."""
        # The patch creates a mock for the entire test method's duration
        self.mock_release_patch = patch(
            "rayforge.pipeline.artifact.cache.ArtifactStore.release"
        )
        self.mock_release = self.mock_release_patch.start()
        self.cache = ArtifactCache()

    def tearDown(self):
        """Stop the patcher after each test."""
        self.mock_release_patch.stop()

    def test_put_and_get_workpiece(self):
        """Tests basic storage and retrieval of a workpiece handle."""
        handle = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        self.cache.put_workpiece_handle("step1", "wp1", handle)
        retrieved = self.cache.get_workpiece_handle("step1", "wp1")
        self.assertIs(retrieved, handle)
        self.mock_release.assert_not_called()

    def test_put_and_get_step(self):
        """Tests basic storage and retrieval of a step handle."""
        handle = create_mock_handle(StepArtifactHandle, "step1")
        self.cache.put_step_handle("step1", handle)
        retrieved = self.cache.get_step_handle("step1")
        self.assertIs(retrieved, handle)
        self.mock_release.assert_not_called()

    def test_put_and_get_job(self):
        """Tests basic storage and retrieval of a job handle."""
        handle = create_mock_handle(JobArtifactHandle, "job")
        self.cache.put_job_handle(handle)
        retrieved = self.cache.get_job_handle()
        self.assertIs(retrieved, handle)
        self.mock_release.assert_not_called()

    def test_invalidate_workpiece_cascades_correctly(self):
        """Tests that invalidating a workpiece cascades up the chain."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        step_h = create_mock_handle(StepArtifactHandle, "step1")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        # Manually populate cache to bypass put methods' invalidation
        self.cache._workpiece_handles[("step1", "wp1")] = wp_h
        self.cache._step_handles["step1"] = step_h
        self.cache._job_handle = job_h

        self.cache.invalidate_for_workpiece("step1", "wp1")

        # Assert that all dependent artifacts were removed
        self.assertIsNone(self.cache.get_workpiece_handle("step1", "wp1"))
        self.assertIsNone(self.cache.get_step_handle("step1"))
        self.assertIsNone(self.cache.get_job_handle())

        # Assert that release was called for all three handles
        self.assertEqual(self.mock_release.call_count, 3)
        self.mock_release.assert_any_call(wp_h)
        self.mock_release.assert_any_call(step_h)
        self.mock_release.assert_any_call(job_h)

    def test_invalidate_step_cascades_correctly(self):
        """Tests that invalidating a step removes its children and parent."""
        wp1_h = create_mock_handle(WorkPieceArtifactHandle, "s1_wp1")
        wp2_h = create_mock_handle(WorkPieceArtifactHandle, "s1_wp2")
        other_wp_h = create_mock_handle(WorkPieceArtifactHandle, "s2_wp1")
        step1_h = create_mock_handle(StepArtifactHandle, "step1")
        step2_h = create_mock_handle(StepArtifactHandle, "step2")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        # Populate cache
        self.cache._workpiece_handles[("step1", "wp1")] = wp1_h
        self.cache._workpiece_handles[("step1", "wp2")] = wp2_h
        self.cache._workpiece_handles[("step2", "wp1")] = other_wp_h
        self.cache._step_handles["step1"] = step1_h
        self.cache._step_handles["step2"] = step2_h
        self.cache._job_handle = job_h

        self.cache.invalidate_for_step("step1")

        # Assert correct items were removed
        self.assertIsNone(self.cache.get_workpiece_handle("step1", "wp1"))
        self.assertIsNone(self.cache.get_workpiece_handle("step1", "wp2"))
        self.assertIsNone(self.cache.get_step_handle("step1"))
        self.assertIsNone(self.cache.get_job_handle())

        # Assert other items remain untouched
        self.assertIsNotNone(self.cache.get_workpiece_handle("step2", "wp1"))
        self.assertIsNotNone(self.cache.get_step_handle("step2"))

        # 2 workpieces + 1 step + 1 job = 4 releases
        self.assertEqual(self.mock_release.call_count, 4)
        self.mock_release.assert_any_call(wp1_h)
        self.mock_release.assert_any_call(wp2_h)
        self.mock_release.assert_any_call(step1_h)
        self.mock_release.assert_any_call(job_h)

    def test_put_workpiece_invalidates_step_and_job(self):
        """Putting a new workpiece handle should invalidate the step/job."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        step_h = create_mock_handle(StepArtifactHandle, "step1")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        # Pre-populate step and job
        self.cache._step_handles["step1"] = step_h
        self.cache._job_handle = job_h

        self.cache.put_workpiece_handle("step1", "wp1", wp_h)

        # Step and job should be gone
        self.assertIsNone(self.cache.get_step_handle("step1"))
        self.assertIsNone(self.cache.get_job_handle())

        # The new workpiece handle should exist
        self.assertIsNotNone(self.cache.get_workpiece_handle("step1", "wp1"))

        # Assert old handles were released
        self.assertEqual(self.mock_release.call_count, 2)
        self.mock_release.assert_any_call(step_h)
        self.mock_release.assert_any_call(job_h)

    def test_shutdown_clears_all_and_releases(self):
        """Tests that shutdown releases all stored handles."""
        wp_h = create_mock_handle(WorkPieceArtifactHandle, "wp1")
        step_h = create_mock_handle(StepArtifactHandle, "step1")
        job_h = create_mock_handle(JobArtifactHandle, "job")

        self.cache._workpiece_handles[("step1", "wp1")] = wp_h
        self.cache._step_handles["step1"] = step_h
        self.cache._job_handle = job_h

        self.cache.shutdown()

        self.assertIsNone(self.cache.get_workpiece_handle("step1", "wp1"))
        self.assertIsNone(self.cache.get_step_handle("step1"))
        self.assertIsNone(self.cache.get_job_handle())
        self.assertEqual(len(self.cache._workpiece_handles), 0)
        self.assertEqual(len(self.cache._step_handles), 0)

        self.assertEqual(self.mock_release.call_count, 3)
        self.mock_release.assert_any_call(wp_h)
        self.mock_release.assert_any_call(step_h)
        self.mock_release.assert_any_call(job_h)


if __name__ == "__main__":
    unittest.main()

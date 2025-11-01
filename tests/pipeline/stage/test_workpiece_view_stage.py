import unittest
from unittest.mock import MagicMock, patch

from rayforge.pipeline.stage.workpiece_view_stage import (
    WorkPieceViewPipelineStage,
)
from rayforge.pipeline.artifact import (
    RenderContext,
    WorkPieceArtifactHandle,
    WorkPieceViewArtifactHandle,
)


@patch("rayforge.pipeline.stage.workpiece_view_stage.get_context")
class TestWorkPieceViewStage(unittest.TestCase):
    """Test suite for the WorkPieceViewPipelineStage."""

    def setUp(self):
        self.mock_artifact_cache = MagicMock()
        self.mock_task_manager = MagicMock()
        self.stage = WorkPieceViewPipelineStage(
            self.mock_task_manager, self.mock_artifact_cache
        )

    def test_stage_requests_render_and_passes_arguments(
        self, mock_get_context
    ):
        """
        Tests that the stage correctly requests a render and passes the
        correct arguments to the worker process.
        """
        step_uid, wp_uid = "s1", "w1"
        source_handle = WorkPieceArtifactHandle(
            shm_name="source_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )

        # Assume this is the final source for this workpiece
        self.mock_artifact_cache.get_workpiece_handle.return_value = (
            source_handle
        )

        context = RenderContext(
            pixels_per_mm=(10.0, 10.0),
            show_travel_moves=False,
            margin_px=0,
            color_set_dict={},
        )

        # Act
        self.stage.request_view_render(
            step_uid, wp_uid, context, source_handle
        )

        # Assert the task manager call
        self.mock_task_manager.run_process.assert_called_once()
        call_args = self.mock_task_manager.run_process.call_args
        kwargs = call_args.kwargs

        self.assertEqual(
            kwargs["workpiece_artifact_handle_dict"], source_handle.to_dict()
        )
        self.assertEqual(kwargs["render_context_dict"], context.to_dict())
        self.assertEqual(kwargs["step_uid"], step_uid)
        self.assertEqual(kwargs["workpiece_uid"], wp_uid)
        self.assertTrue(kwargs["is_final_source"])
        # Check that the key is the source handle's shm_name
        self.assertEqual(kwargs["key"], source_handle.shm_name)

    def test_stage_handles_events_and_cleans_up_source_handle(
        self, mock_get_context
    ):
        """
        Tests that the stage correctly handles events from the runner,
        emits its own signals, and cleans up the source artifact handle
        on completion.
        """
        step_uid, wp_uid = "s1", "w1"

        # Arrange Mocks
        source_handle = WorkPieceArtifactHandle(
            shm_name="source_shm",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
        )
        mock_store = mock_get_context.return_value.artifact_store
        self.mock_artifact_cache.get_workpiece_handle.return_value = (
            source_handle
        )
        context = RenderContext((10.0, 10.0), False, 0, {})
        # End Arrange

        self.stage.request_view_render(
            step_uid, wp_uid, context, source_handle
        )

        # Get the callbacks and key from the task manager call
        self.mock_task_manager.run_process.assert_called_once()
        call_kwargs = self.mock_task_manager.run_process.call_args.kwargs
        when_event_cb = call_kwargs["when_event"]
        when_done_cb = call_kwargs["when_done"]
        task_key = call_kwargs["key"]

        mock_task = MagicMock()
        mock_task.key = task_key
        mock_task.get_status.return_value = "completed"

        # Mock signal handler for the new signal
        changed_handler = MagicMock()
        self.stage.view_artifacts_changed.connect(changed_handler)

        # Act 1: Simulate "created" event from runner
        view_handle_dict = {
            "shm_name": "final_view_shm",
            "bbox_mm": (0, 0, 1, 1),
            "handle_class_name": "WorkPieceViewArtifactHandle",
            "artifact_type_name": "WorkPieceViewArtifact",
        }
        event_payload = {
            "handle_dict": view_handle_dict,
            "is_final_source": True,
            "step_uid": step_uid,
            "workpiece_uid": wp_uid,
        }
        when_event_cb(mock_task, "view_artifact_created", event_payload)

        # Assert 1: `adopt` is called and `view_artifacts_changed` signal fires
        mock_store.adopt.assert_called_once()
        # Check that the adopted handle is of the correct type
        adopted_handle = mock_store.adopt.call_args.args[0]
        self.assertIsInstance(adopted_handle, WorkPieceViewArtifactHandle)
        self.assertEqual(adopted_handle.shm_name, "final_view_shm")
        # Check that the signal was emitted correctly
        changed_handler.assert_called_once_with(
            self.stage, step_uid=step_uid, workpiece_uid=wp_uid
        )

        # Act 2: Simulate task completion
        when_done_cb(mock_task)

        # Assert 2: Original source artifact is released
        mock_store.release.assert_called_once_with(source_handle)

    def test_shutdown_cleans_up_source_handles(self, mock_get_context):
        """
        Tests that the shutdown method cancels tasks. The release of the
        source handle is managed by the task's `when_done` callback, which
        should be triggered by the task manager upon cancellation.
        """
        # Arrange: Simulate a render request in flight
        step_uid, wp_uid = "s1", "w1"
        source_handle = WorkPieceArtifactHandle(
            shm_name="source_shm",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            generation_size=(10, 10),
            source_dimensions=(10, 10),
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
        )

        mock_store = mock_get_context.return_value.artifact_store
        self.mock_artifact_cache.get_workpiece_handle.return_value = (
            source_handle
        )
        context = RenderContext((10.0, 10.0), False, 0, {})

        self.stage.request_view_render(
            step_uid, wp_uid, context, source_handle
        )
        self.assertTrue(self.stage.is_busy)

        # Act
        self.stage.shutdown()

        # Assert
        self.mock_task_manager.cancel_task.assert_called_once()
        # The shutdown method itself does not release the handle; it relies on
        # the task's when_done callback, which the TaskManager should invoke
        # even for a cancelled task. We don't test for release here as it's
        # an interaction with the TaskManager, not a direct action of shutdown.
        mock_store.release.assert_not_called()
        self.assertFalse(self.stage.is_busy)


if __name__ == "__main__":
    unittest.main()

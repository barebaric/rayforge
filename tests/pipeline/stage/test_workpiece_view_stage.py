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


@patch("rayforge.pipeline.stage.workpiece_view.get_context")
class TestWorkPieceViewStage(unittest.TestCase):
    """Test suite for the WorkPieceViewPipelineStage."""

    def setUp(self):
        self.mock_artifact_cache = MagicMock()
        self.mock_task_manager = MagicMock()
        self.stage = WorkPieceViewPipelineStage(
            self.mock_task_manager, self.mock_artifact_cache
        )

    def test_stage_requests_render_and_passes_source_handle(
        self, mock_get_context
    ):
        """
        Tests that the stage correctly requests a render and passes the
        original source handle to the worker.
        """
        step_uid, wp_uid = "s1", "w1"
        generation_id = 1
        source_handle = WorkPieceArtifactHandle(
            shm_name="source_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )

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
            step_uid, wp_uid, context, source_handle, generation_id
        )

        # Assert the task manager call
        self.mock_task_manager.run_process.assert_called_once()
        call_args = self.mock_task_manager.run_process.call_args
        # The worker receives the handle to the ORIGINAL source artifact
        self.assertEqual(
            call_args.kwargs["workpiece_artifact_handle_dict"],
            source_handle.to_dict(),
        )
        self.assertEqual(
            call_args.kwargs["render_context_dict"], context.to_dict()
        )
        self.assertEqual(call_args.kwargs["generation_id"], generation_id)
        # Check that the key is constructed correctly
        expected_key = (
            step_uid,
            wp_uid,
            source_handle.shm_name,
            generation_id,
        )
        self.assertEqual(call_args.kwargs["key"], expected_key)

    def test_stage_handles_events_and_cleans_up_source_handle(
        self, mock_get_context
    ):
        """
        Tests that the stage correctly handles events from the runner,
        emits its own signals, and cleans up the source artifact handle
        on completion.
        """
        step_uid, wp_uid = "s1", "w1"
        generation_id = 1

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
            step_uid, wp_uid, context, source_handle, generation_id
        )

        # Get the callbacks and key from the task manager call
        self.mock_task_manager.run_process.assert_called_once()
        call_kwargs = self.mock_task_manager.run_process.call_args.kwargs
        when_event_cb = call_kwargs["when_event"]
        when_done_cb = call_kwargs["when_done"]
        task_key = call_kwargs["key"]

        mock_task = MagicMock()
        mock_task.key = task_key  # Use the correct, 4-element key
        mock_task.get_status.return_value = "completed"

        # Mock signal handlers
        created_handler = MagicMock()
        updated_handler = MagicMock()
        ready_handler = MagicMock()
        finished_handler = MagicMock()
        self.stage.view_artifact_created.connect(created_handler)
        self.stage.view_artifact_updated.connect(updated_handler)
        self.stage.view_artifact_ready.connect(ready_handler)
        self.stage.generation_finished.connect(finished_handler)

        # Act 1: Simulate "created" event
        view_handle_dict = {
            "shm_name": "final_view_shm",
            "bbox_mm": (0, 0, 1, 1),
            "handle_class_name": "WorkPieceViewArtifactHandle",
            "artifact_type_name": "WorkPieceViewArtifact",
        }
        when_event_cb(
            mock_task,
            "view_artifact_created",
            {"handle_dict": view_handle_dict, "generation_id": generation_id},
        )

        # Assert 1: `adopt` is called and `created` signal fires
        mock_store.adopt.assert_called_once()
        created_handler.assert_called_once()
        ready_handler.assert_not_called()
        self.assertIsInstance(
            created_handler.call_args.kwargs["handle"],
            WorkPieceViewArtifactHandle,
        )

        # Act 2: Simulate "updated" event
        when_event_cb(mock_task, "view_artifact_updated", {})
        updated_handler.assert_called_once()

        # Act 3: Simulate task completion
        when_done_cb(mock_task)

        # Assert 3: Final signals fire and original source artifact is released
        ready_handler.assert_called_once()
        finished_handler.assert_called_once()
        self.assertEqual(
            finished_handler.call_args.kwargs["key"], (step_uid, wp_uid)
        )
        mock_store.release.assert_called_once_with(source_handle)

    def test_shutdown_cleans_up_source_handles(self, mock_get_context):
        """
        Tests that the shutdown method cancels tasks and cleans up any
        in-flight source artifact handles it has taken ownership of.
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
            step_uid, wp_uid, context, source_handle, 1
        )
        self.assertTrue(self.stage.is_busy)

        # Act
        self.stage.shutdown()

        # Assert
        self.mock_task_manager.cancel_task.assert_called_once()
        mock_store.release.assert_called_once_with(source_handle)
        self.assertFalse(self.stage.is_busy)


if __name__ == "__main__":
    unittest.main()

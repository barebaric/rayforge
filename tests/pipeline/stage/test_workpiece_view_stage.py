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


class TestWorkPieceViewStage(unittest.TestCase):
    """Test suite for the WorkPieceViewPipelineStage."""

    def setUp(self):
        self.mock_artifact_cache = MagicMock()
        self.mock_task_manager = MagicMock()
        self.stage = WorkPieceViewPipelineStage(
            self.mock_task_manager, self.mock_artifact_cache
        )

    def test_stage_requests_vector_render(self):
        """
        Tests that the stage correctly calls the task manager for a
        vector artifact.
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
        self.mock_artifact_cache.get_workpiece_handle.return_value = (
            source_handle
        )
        context = RenderContext(
            pixels_per_mm=(10.0, 10.0),
            show_travel_moves=False,
            margin_px=0,
            color_set_dict={},
        )

        self.stage.request_view_render(step_uid, wp_uid, context)

        self.mock_task_manager.run_process.assert_called_once()
        call_args = self.mock_task_manager.run_process.call_args
        self.assertEqual(
            call_args.kwargs["workpiece_artifact_handle_dict"],
            source_handle.to_dict(),
        )
        self.assertEqual(
            call_args.kwargs["render_context_dict"], context.to_dict()
        )

    @patch("rayforge.pipeline.stage.workpiece_view_stage.get_context")
    def test_stage_handles_events_and_completion(self, mock_get_context):
        """
        Tests that the stage correctly handles events from the runner and
        emits its own signals.
        """
        step_uid, wp_uid = "s1", "w1"
        key = (step_uid, wp_uid)
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
        context = RenderContext((10.0, 10.0), False, 0, {})

        self.stage.request_view_render(step_uid, wp_uid, context)

        # Get the callbacks from the task manager call
        self.mock_task_manager.run_process.assert_called_once()
        call_kwargs = self.mock_task_manager.run_process.call_args.kwargs
        when_event_cb = call_kwargs["when_event"]
        when_done_cb = call_kwargs["when_done"]

        mock_task = MagicMock()
        mock_task.key = key
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
        handle_dict = {
            "shm_name": "test",
            "bbox_mm": (0, 0, 1, 1),
            "handle_class_name": "WorkPieceViewArtifactHandle",
            "artifact_type_name": "WorkPieceViewArtifact",
        }
        when_event_cb(
            mock_task, "view_artifact_created", {"handle_dict": handle_dict}
        )

        # Assert 1: `adopt` is called and signals fire
        mock_get_context.return_value.artifact_store.adopt.assert_called_once()
        created_handler.assert_called_once()
        ready_handler.assert_called_once()
        self.assertIsInstance(
            created_handler.call_args.kwargs["handle"],
            WorkPieceViewArtifactHandle,
        )
        self.assertIsInstance(
            ready_handler.call_args.kwargs["handle"],
            WorkPieceViewArtifactHandle,
        )

        # Act 2: Simulate "updated" event
        when_event_cb(mock_task, "view_artifact_updated", {})

        # Assert 2
        updated_handler.assert_called_once()
        self.assertEqual(
            updated_handler.call_args.kwargs["step_uid"], step_uid
        )

        # Act 3: Simulate task completion
        when_done_cb(mock_task)

        # Assert 3
        finished_handler.assert_called_once()
        self.assertEqual(finished_handler.call_args.kwargs["key"], key)
        # `ready` handler should NOT be called again on completion
        ready_handler.assert_called_once()


if __name__ == "__main__":
    unittest.main()

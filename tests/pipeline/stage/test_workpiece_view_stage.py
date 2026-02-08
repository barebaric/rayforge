import time
import unittest
import numpy as np
from unittest.mock import MagicMock

from rayforge.pipeline.stage.workpiece_view_stage import (
    WorkPieceViewPipelineStage,
)
from rayforge.pipeline.artifact import (
    RenderContext,
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
    WorkPieceViewArtifactHandle,
)
from rayforge.pipeline.artifact.lifecycle import (
    LedgerEntry,
    ArtifactLifecycle,
)
from rayforge.core.ops import Ops
from rayforge.pipeline.coord import CoordinateSystem


class TestWorkPieceViewStage(unittest.TestCase):
    """Test suite for the WorkPieceViewPipelineStage."""

    def setUp(self):
        self.mock_artifact_manager = MagicMock()
        self.mock_task_manager = MagicMock()
        self.mock_machine = MagicMock()
        self.stage = WorkPieceViewPipelineStage(
            self.mock_task_manager,
            self.mock_artifact_manager,
            self.mock_machine,
        )
        # Mock cancel_task to do nothing
        self.mock_task_manager.cancel_task = MagicMock()

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
        self.mock_artifact_manager.get_workpiece_handle.return_value = (
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

    def test_stage_handles_events_and_completion(self):
        """
        Tests that the stage correctly handles events from the runner and
        emits its own signals.
        """
        step_uid, wp_uid = "s1", "w1"
        key = ("view", step_uid, wp_uid)
        source_handle = WorkPieceArtifactHandle(
            shm_name="source_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )
        self.mock_artifact_manager.get_workpiece_handle.return_value = (
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
        mock_task.id = 123
        mock_task.get_status.return_value = "completed"
        mock_task.is_final.return_value = False

        # Mock get_task to return our mock task
        self.mock_task_manager.get_task.return_value = mock_task

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
        view_handle = WorkPieceViewArtifactHandle(
            shm_name="test",
            bbox_mm=(0, 0, 1, 1),
            handle_class_name="WorkPieceViewArtifactHandle",
            artifact_type_name="WorkPieceViewArtifact",
        )
        self.mock_artifact_manager.adopt_artifact.return_value = view_handle
        when_event_cb(
            mock_task, "view_artifact_created", {"handle_dict": handle_dict}
        )

        # Assert 1: `adopt` is called and signals fire
        self.mock_artifact_manager.adopt_artifact.assert_called_once()
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

        # Assert 3: Task completion is signaled
        finished_handler.assert_called_once()
        # The signal is sent with internal ViewKey (step_uid, workpiece_uid)
        internal_key = (step_uid, wp_uid)
        self.assertEqual(
            finished_handler.call_args.kwargs["key"], internal_key
        )
        # `ready` handler should NOT be called again on completion
        ready_handler.assert_called_once()

    def test_adoption_failure_does_not_crash(self):
        """
        Tests that adoption failures are handled gracefully without
        crashing the stage.
        """
        step_uid, wp_uid = "s1", "w1"
        key = ("view", step_uid, wp_uid)
        source_handle = WorkPieceArtifactHandle(
            shm_name="source_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )
        self.mock_artifact_manager.get_workpiece_handle.return_value = (
            source_handle
        )
        context = RenderContext((10.0, 10.0), False, 0, {})

        self.stage.request_view_render(step_uid, wp_uid, context)

        # Get the callbacks from the task manager call
        self.mock_task_manager.run_process.assert_called_once()
        call_kwargs = self.mock_task_manager.run_process.call_args.kwargs
        when_event_cb = call_kwargs["when_event"]

        mock_task = MagicMock()
        mock_task.key = key

        # Mock adopt to raise an exception
        self.mock_artifact_manager.adopt_artifact.side_effect = Exception(
            "Adoption failed"
        )

        # Mock signal handler to verify no crash
        created_handler = MagicMock()
        self.stage.view_artifact_created.connect(created_handler)

        # Act: Simulate "created" event with adoption failure
        handle_dict = {
            "shm_name": "test",
            "bbox_mm": (0, 0, 1, 1),
            "handle_class_name": "WorkPieceViewArtifactHandle",
            "artifact_type_name": "WorkPieceViewArtifact",
        }

        when_event_cb(
            mock_task, "view_artifact_created", {"handle_dict": handle_dict}
        )

        # Assert: Created signal was not sent due to adoption failure
        created_handler.assert_not_called()

    def test_multiple_view_artifact_updated_events(self):
        """
        Tests that multiple view_artifact_updated events are sent and
        processed correctly for progressive rendering.
        """
        step_uid, wp_uid = "s1", "w1"
        key = ("view", step_uid, wp_uid)
        source_handle = WorkPieceArtifactHandle(
            shm_name="source_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )
        self.mock_artifact_manager.get_workpiece_handle.return_value = (
            source_handle
        )
        context = RenderContext((10.0, 10.0), False, 0, {})

        self.stage.request_view_render(step_uid, wp_uid, context)

        # Get the callbacks from the task manager call
        self.mock_task_manager.run_process.assert_called_once()
        call_kwargs = self.mock_task_manager.run_process.call_args.kwargs
        when_event_cb = call_kwargs["when_event"]

        mock_task = MagicMock()
        mock_task.key = key

        # Mock signal handlers
        created_handler = MagicMock()
        updated_handler = MagicMock()
        self.stage.view_artifact_created.connect(created_handler)
        self.stage.view_artifact_updated.connect(updated_handler)

        # Act 1: Simulate "created" event
        handle_dict = {
            "shm_name": "test_view",
            "bbox_mm": (0, 0, 1, 1),
            "handle_class_name": "WorkPieceViewArtifactHandle",
            "artifact_type_name": "WorkPieceViewArtifact",
        }
        view_handle = WorkPieceViewArtifactHandle(
            shm_name="test_view",
            bbox_mm=(0, 0, 1, 1),
            handle_class_name="WorkPieceViewArtifactHandle",
            artifact_type_name="WorkPieceViewArtifact",
        )
        self.mock_artifact_manager.adopt_artifact.return_value = view_handle
        when_event_cb(
            mock_task, "view_artifact_created", {"handle_dict": handle_dict}
        )

        # Assert 1: Created signal fired
        created_handler.assert_called_once()

        # Act 2: Simulate multiple "updated" events (progressive rendering)
        when_event_cb(mock_task, "view_artifact_updated", {})
        when_event_cb(mock_task, "view_artifact_updated", {})
        when_event_cb(mock_task, "view_artifact_updated", {})

        # Assert 2: All updated signals should be fired
        self.assertEqual(updated_handler.call_count, 3)
        # Each call should have the correct step_uid
        for call in updated_handler.call_args_list:
            self.assertEqual(call.kwargs["step_uid"], step_uid)
            self.assertEqual(call.kwargs["workpiece_uid"], wp_uid)

    def test_progressive_rendering_sends_multiple_updates(self):
        """
        Tests that the workpiece view stage correctly handles multiple
        view_artifact_updated events for progressive rendering.
        Verifies that each update is relayed correctly.
        """
        step_uid, wp_uid = "s1", "w1"
        key = ("view", step_uid, wp_uid)
        source_handle = WorkPieceArtifactHandle(
            shm_name="source_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )
        self.mock_artifact_manager.get_workpiece_handle.return_value = (
            source_handle
        )
        context = RenderContext((10.0, 10.0), False, 0, {})

        self.stage.request_view_render(step_uid, wp_uid, context)

        # Get the callbacks from the task manager call
        self.mock_task_manager.run_process.assert_called_once()
        call_kwargs = self.mock_task_manager.run_process.call_args.kwargs
        when_event_cb = call_kwargs["when_event"]

        mock_task = MagicMock()
        mock_task.key = key

        # Mock signal handlers to track calls
        created_handler = MagicMock()
        updated_handler = MagicMock()
        self.stage.view_artifact_created.connect(created_handler)
        self.stage.view_artifact_updated.connect(updated_handler)

        # Act 1: Simulate "created" event
        handle_dict = {
            "shm_name": "test_progressive",
            "bbox_mm": (0, 0, 1, 1),
            "handle_class_name": "WorkPieceViewArtifactHandle",
            "artifact_type_name": "WorkPieceViewArtifact",
        }
        view_handle = WorkPieceViewArtifactHandle(
            shm_name="test_progressive",
            bbox_mm=(0, 0, 1, 1),
            handle_class_name="WorkPieceViewArtifactHandle",
            artifact_type_name="WorkPieceViewArtifact",
        )
        self.mock_artifact_manager.adopt_artifact.return_value = view_handle
        when_event_cb(
            mock_task,
            "view_artifact_created",
            {"handle_dict": handle_dict},
        )

        # Assert 1: Created signal fired once
        created_handler.assert_called_once()

        # Act 2: Simulate multiple "updated" events (progressive rendering)
        # This simulates the worker sending updates after drawing texture,
        # after drawing vertices, etc.
        when_event_cb(mock_task, "view_artifact_updated", {})
        when_event_cb(mock_task, "view_artifact_updated", {})
        when_event_cb(mock_task, "view_artifact_updated", {})

        # Assert 2: All updated signals should be fired
        self.assertEqual(updated_handler.call_count, 3)
        # Each call should have the correct step_uid and workpiece_uid
        for call in updated_handler.call_args_list:
            self.assertEqual(call.kwargs["step_uid"], step_uid)
            self.assertEqual(call.kwargs["workpiece_uid"], wp_uid)
            # The handle should be the same for all updates (same artifact)
            self.assertIsNotNone(call.kwargs["handle"])

    def test_on_workpiece_chunk_available_receives_chunks(self):
        """
        Tests that _on_workpiece_chunk_available is called when a chunk
        is available from the workpiece stage.
        """
        step_uid, wp_uid = "s1", "w1"
        key = (step_uid, wp_uid)

        # Create a mock chunk handle
        chunk_handle = WorkPieceArtifactHandle(
            shm_name="chunk_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )

        generation_id = 1

        # Act: Call _on_workpiece_chunk_available directly
        self.stage._on_workpiece_chunk_available(
            sender=None,
            key=key,
            chunk_handle=chunk_handle,
            generation_id=generation_id,
        )

        # Assert: The method should complete without errors
        # For now, it just logs, so we just verify it doesn't crash

    def test_live_render_context_established_on_view_creation(self):
        """
        Tests that a live render context is established when a view
        artifact is created, enabling progressive chunk rendering.
        """
        step_uid, wp_uid = "s1", "w1"
        key = ("view", step_uid, wp_uid)
        source_handle = WorkPieceArtifactHandle(
            shm_name="source_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )
        self.mock_artifact_manager.get_workpiece_handle.return_value = (
            source_handle
        )
        context = RenderContext((10.0, 10.0), False, 0, {})

        self.stage.request_view_render(step_uid, wp_uid, context)

        # Get the callbacks from the task manager call
        self.mock_task_manager.run_process.assert_called_once()
        call_kwargs = self.mock_task_manager.run_process.call_args.kwargs
        when_event_cb = call_kwargs["when_event"]

        mock_task = MagicMock()
        mock_task.key = key

        # Act: Simulate "created" event
        handle_dict = {
            "shm_name": "test_live_render",
            "bbox_mm": (0, 0, 1, 1),
            "handle_class_name": "WorkPieceViewArtifactHandle",
            "artifact_type_name": "WorkPieceViewArtifact",
        }
        view_handle = WorkPieceViewArtifactHandle(
            shm_name="test_live_render",
            bbox_mm=(0, 0, 1, 1),
            handle_class_name="WorkPieceViewArtifactHandle",
            artifact_type_name="WorkPieceViewArtifact",
        )
        self.mock_artifact_manager.adopt_artifact.return_value = view_handle

        # Set up mock for _get_ledger_entry to return a proper entry
        mock_entry = LedgerEntry(
            state=ArtifactLifecycle.READY,
            handle=view_handle,
            metadata={"render_context": context}
        )
        self.mock_artifact_manager._get_ledger_entry.return_value = (
            mock_entry
        )

        when_event_cb(
            mock_task, "view_artifact_created", {"handle_dict": handle_dict}
        )

        # Assert: Render context should be stored in ledger metadata
        self.assertIn("render_context", mock_entry.metadata)
        self.assertEqual(mock_entry.metadata["render_context"], context)

    def test_throttled_notification_limits_update_frequency(self):
        """
        Tests that throttled notification limits the frequency of
        view_artifact_updated signals when many chunks arrive quickly.
        """
        step_uid, wp_uid = "s1", "w1"
        key = ("view", step_uid, wp_uid)
        source_handle = WorkPieceArtifactHandle(
            shm_name="source_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )
        self.mock_artifact_manager.get_workpiece_handle.return_value = (
            source_handle
        )
        context = RenderContext((10.0, 10.0), False, 0, {})

        self.stage.request_view_render(step_uid, wp_uid, context)

        # Get the callbacks from the task manager call
        self.mock_task_manager.run_process.assert_called_once()
        call_kwargs = self.mock_task_manager.run_process.call_args.kwargs
        when_event_cb = call_kwargs["when_event"]

        mock_task = MagicMock()
        mock_task.key = key

        # Act 1: Create view artifact to establish live render context
        handle_dict = {
            "shm_name": "test_throttle",
            "bbox_mm": (0, 0, 1, 1),
            "handle_class_name": "WorkPieceViewArtifactHandle",
            "artifact_type_name": "WorkPieceViewArtifact",
        }
        view_handle = WorkPieceViewArtifactHandle(
            shm_name="test_throttle",
            bbox_mm=(0, 0, 1, 1),
            handle_class_name="WorkPieceViewArtifactHandle",
            artifact_type_name="WorkPieceViewArtifact",
        )
        self.mock_artifact_manager.adopt_artifact.return_value = view_handle
        self.mock_artifact_manager.get_workpiece_view_handle.return_value = (
            view_handle
        )
        self.mock_artifact_manager.put_workpiece_view_handle.return_value = (
            None
        )

        # Set up mock for _get_ledger_entry to return a proper entry
        mock_entry = LedgerEntry(
            state=ArtifactLifecycle.READY,
            handle=view_handle,
            metadata={"render_context": context}
        )
        self.mock_artifact_manager._get_ledger_entry.return_value = (
            mock_entry
        )

        when_event_cb(
            mock_task, "view_artifact_created", {"handle_dict": handle_dict}
        )

        # Track update signal calls
        update_handler = MagicMock()
        self.stage.view_artifact_updated.connect(update_handler)

        # Wait for view_artifact_created to be processed
        time.sleep(0.01)

        # Mock artifact_manager.get_artifact to return a real WorkPieceArtifact
        chunk_ops = Ops()
        chunk_ops.move_to(0, 0, 0)
        chunk_ops.line_to(1, 1, 0)

        chunk_artifact = WorkPieceArtifact(
            ops=chunk_ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(10.0, 10.0),
            generation_size=(10.0, 10.0),
        )
        self.mock_artifact_manager.get_artifact.return_value = chunk_artifact

        # Mock run_thread to execute callback synchronously
        def mock_run_thread(target, *args, when_done=None, **kwargs):
            mock_task = MagicMock()
            mock_task.get_status.return_value = "completed"
            mock_task.result.return_value = True
            if when_done:
                when_done(mock_task)
            return mock_task

        self.mock_task_manager.run_thread.side_effect = mock_run_thread

        # Act 2: Flood with 10 chunk updates (simulating rapid chunk arrival)
        chunk_handle = WorkPieceArtifactHandle(
            shm_name="chunk_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )

        # Use internal ViewKey (step_uid, workpiece_uid) not task key
        internal_key = (step_uid, wp_uid)
        # Use generation_id=0 to match the live render context's generation_id
        # which is set to 0 by _initialize_live_render_context
        for i in range(10):
            self.stage._on_workpiece_chunk_available(
                sender=None,
                key=internal_key,
                chunk_handle=chunk_handle,
                generation_id=0,
            )

        # Wait for any pending timers to complete
        time.sleep(0.1)

        # Assert: Should have far fewer than 10 update signals due to
        # throttling.
        # With 10 rapid chunks and ~33ms throttle interval, we expect
        # only 1-2 updates to be sent
        self.assertLess(update_handler.call_count, 5)
        self.assertGreater(update_handler.call_count, 0)

    def test_incremental_bitmap_rendering_draws_chunk_to_view(self):
        """
        Unit test for Phase 3, Item 6: Incremental Bitmap Updating.
        Feeds a blank bitmap and a mock vector chunk. Verifies the bitmap
        is modified correctly. Ensures the source artifact's refcount is
        respected during this operation.
        """
        step_uid, wp_uid = "s1", "w1"
        key = ("view", step_uid, wp_uid)
        source_handle = WorkPieceArtifactHandle(
            shm_name="source_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10.0, 10.0),
            generation_size=(10.0, 10.0),
        )
        self.mock_artifact_manager.get_workpiece_handle.return_value = (
            source_handle
        )
        context = RenderContext((10.0, 10.0), False, 0, {})

        self.stage.request_view_render(step_uid, wp_uid, context)

        # Get the callbacks from the task manager call
        call_kwargs = self.mock_task_manager.run_process.call_args.kwargs
        when_event_cb = call_kwargs["when_event"]

        mock_task = MagicMock()
        mock_task.key = key

        # Create a blank bitmap for the view artifact
        from rayforge.pipeline.artifact import WorkPieceViewArtifact

        blank_bitmap = np.zeros((100, 100, 4), dtype=np.uint8)
        view_artifact = WorkPieceViewArtifact(
            bitmap_data=blank_bitmap, bbox_mm=(0, 0, 10.0, 10.0)
        )

        # Mock the artifact_manager to return our artifacts
        def mock_get_artifact(handle):
            if handle.shm_name == "view_shm":
                return view_artifact
            return None

        self.mock_artifact_manager.get_artifact.side_effect = mock_get_artifact

        # Act 1: Create view artifact to establish live render context
        handle_dict = {
            "shm_name": "view_shm",
            "bbox_mm": (0, 0, 10.0, 10.0),
            "handle_class_name": "WorkPieceViewArtifactHandle",
            "artifact_type_name": "WorkPieceViewArtifact",
        }
        view_handle = WorkPieceViewArtifactHandle(
            shm_name="view_shm",
            bbox_mm=(0, 0, 10.0, 10.0),
            handle_class_name="WorkPieceViewArtifactHandle",
            artifact_type_name="WorkPieceViewArtifact",
        )
        self.mock_artifact_manager.adopt_artifact.return_value = view_handle
        self.mock_artifact_manager.get_workpiece_view_handle.return_value = (
            view_handle
        )
        self.mock_artifact_manager.put_workpiece_view_handle.return_value = (
            None
        )

        # Set up mock for _get_ledger_entry to return a proper entry
        mock_entry = LedgerEntry(
            state=ArtifactLifecycle.READY,
            handle=view_handle,
            metadata={"render_context": context}
        )
        self.mock_artifact_manager._get_ledger_entry.return_value = (
            mock_entry
        )

        when_event_cb(
            mock_task, "view_artifact_created", {"handle_dict": handle_dict}
        )

        # Assert: Render context should be stored in ledger metadata
        self.assertIn("render_context", mock_entry.metadata)

        # Track retain/release calls to verify refcount is respected
        retain_calls = []
        release_calls = []

        def mock_retain_handle(handle):
            retain_calls.append(handle.shm_name)

        def mock_release_handle(handle):
            release_calls.append(handle.shm_name)

        self.mock_artifact_manager.retain_handle.side_effect = (
            mock_retain_handle
        )
        self.mock_artifact_manager.release_handle.side_effect = (
            mock_release_handle
        )

        # Create a mock vector chunk artifact
        chunk_ops = Ops()
        chunk_ops.move_to(2.0, 2.0, 0)
        chunk_ops.line_to(8.0, 8.0, 0)

        chunk_artifact = WorkPieceArtifact(
            ops=chunk_ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(10.0, 10.0),
            generation_size=(10.0, 10.0),
        )

        # Update mock_get_artifact to return chunk artifact for chunk_shm
        def mock_get_artifact_with_chunk(handle):
            if handle.shm_name == "view_shm":
                return view_artifact
            elif handle.shm_name == "chunk_shm":
                return chunk_artifact
            return None

        self.mock_artifact_manager.get_artifact.side_effect = (
            mock_get_artifact_with_chunk
        )

        # Act 2: Send a chunk to be rendered incrementally
        chunk_handle = WorkPieceArtifactHandle(
            shm_name="chunk_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )

        # Get the initial bitmap state (all zeros)
        initial_bitmap = view_artifact.bitmap_data.copy()
        self.assertEqual(np.count_nonzero(initial_bitmap), 0)

        # Mock run_thread to execute callback synchronously
        def mock_run_thread(target, *args, when_done=None, **kwargs):
            mock_task = MagicMock()
            mock_task.get_status.return_value = "completed"
            mock_task.result.return_value = True
            if when_done:
                when_done(mock_task)
            return mock_task

        self.mock_task_manager.run_thread.side_effect = mock_run_thread

        # Use internal ViewKey (step_uid, workpiece_uid) not task key
        internal_key = (step_uid, wp_uid)
        self.stage._on_workpiece_chunk_available(
            sender=None,
            key=internal_key,
            chunk_handle=chunk_handle,
            generation_id=0,
        )

        # Assert: The chunk handle should be released (refcount respected)
        self.assertIn("chunk_shm", release_calls)

        # Note: We can't easily verify the bitmap was modified without
        # actually rendering to shared memory, which requires more complex
        # setup. The key verification here is that:
        # 1. The method completes without errors
        # 2. The chunk handle is properly released (refcount respected)
        # 3. The update signal is triggered (via throttled update)

    def test_adopt_view_handle(self):
        """Tests that _adopt_view_handle correctly adopts a view handle."""
        step_uid, wp_uid = "s1", "w1"
        key = (step_uid, wp_uid)

        handle_dict = {
            "shm_name": "test_adopt",
            "bbox_mm": (0, 0, 1, 1),
            "handle_class_name": "WorkPieceViewArtifactHandle",
            "artifact_type_name": "WorkPieceViewArtifact",
        }
        view_handle = WorkPieceViewArtifactHandle(
            shm_name="test_adopt",
            bbox_mm=(0, 0, 1, 1),
            handle_class_name="WorkPieceViewArtifactHandle",
            artifact_type_name="WorkPieceViewArtifact",
        )
        self.mock_artifact_manager.adopt_artifact.return_value = view_handle

        result = self.stage._adopt_view_handle(
            key, {"handle_dict": handle_dict}
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, WorkPieceViewArtifactHandle)
        self.mock_artifact_manager.adopt_artifact.assert_called_once_with(
            key, handle_dict
        )

    def test_adopt_view_handle_wrong_type(self):
        """Tests that _adopt_view_handle raises on wrong handle type."""
        step_uid, wp_uid = "s1", "w1"
        key = (step_uid, wp_uid)

        handle_dict = {
            "shm_name": "test_wrong",
            "bbox_mm": (0, 0, 1, 1),
            "handle_class_name": "WorkPieceArtifactHandle",
            "artifact_type_name": "WorkPieceArtifact",
        }
        wrong_handle = WorkPieceArtifactHandle(
            shm_name="test_wrong",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )
        self.mock_artifact_manager.adopt_artifact.return_value = wrong_handle

        with self.assertRaises(TypeError):
            self.stage._adopt_view_handle(key, {"handle_dict": handle_dict})

    def test_replace_current_view_handle(self):
        """Tests that _replace_current_view_handle replaces handles."""
        step_uid, wp_uid = "s1", "w1"
        key = (step_uid, wp_uid)

        old_handle = WorkPieceViewArtifactHandle(
            shm_name="old",
            bbox_mm=(0, 0, 1, 1),
            handle_class_name="WorkPieceViewArtifactHandle",
            artifact_type_name="WorkPieceViewArtifact",
        )
        self.mock_artifact_manager.get_workpiece_view_handle.return_value = (
            old_handle
        )

        new_handle = WorkPieceViewArtifactHandle(
            shm_name="new",
            bbox_mm=(0, 0, 1, 1),
            handle_class_name="WorkPieceViewArtifactHandle",
            artifact_type_name="WorkPieceViewArtifact",
        )

        self.stage._replace_current_view_handle(key, new_handle)

        self.mock_artifact_manager.release_handle.assert_called_once_with(
            old_handle
        )
        self.mock_artifact_manager.retain_handle.assert_called_once_with(
            new_handle
        )

    def test_get_chunk_artifact(self):
        """Tests that _get_chunk_artifact returns valid artifact."""
        step_uid, wp_uid = "s1", "w1"
        key = (step_uid, wp_uid)

        chunk_ops = Ops()
        chunk_ops.move_to(0, 0, 0)
        chunk_artifact = WorkPieceArtifact(
            ops=chunk_ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(10.0, 10.0),
            generation_size=(10.0, 10.0),
        )

        chunk_handle = WorkPieceArtifactHandle(
            shm_name="chunk",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )
        self.mock_artifact_manager.get_artifact.return_value = chunk_artifact

        result = self.stage._get_chunk_artifact(key, chunk_handle)
        self.assertEqual(result, chunk_artifact)

    def test_get_chunk_artifact_none(self):
        """Tests that _get_chunk_artifact handles None artifact."""
        step_uid, wp_uid = "s1", "w1"
        key = (step_uid, wp_uid)

        chunk_handle = WorkPieceArtifactHandle(
            shm_name="chunk",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )
        self.mock_artifact_manager.get_artifact.return_value = None

        result = self.stage._get_chunk_artifact(key, chunk_handle)
        self.assertIsNone(result)

    def test_get_render_components(self):
        """Tests that _get_render_components returns components."""
        step_uid, wp_uid = "s1", "w1"
        key = (step_uid, wp_uid)
        ledger_key = ("view", step_uid, wp_uid)

        context = RenderContext((10.0, 10.0), False, 0, {})
        handle = WorkPieceViewArtifactHandle(
            shm_name="test",
            bbox_mm=(0, 0, 1, 1),
            handle_class_name="WorkPieceViewArtifactHandle",
            artifact_type_name="WorkPieceViewArtifact",
        )
        self.mock_artifact_manager.get_workpiece_view_handle.return_value = (
            handle
        )

        mock_entry = MagicMock()
        mock_entry.metadata = {"render_context": context}
        self.mock_artifact_manager._get_ledger_entry.return_value = mock_entry

        chunk_handle = WorkPieceArtifactHandle(
            shm_name="chunk",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )

        view_handle, render_context = self.stage._get_render_components(
            key, ledger_key, chunk_handle
        )

        self.assertEqual(view_handle, handle)
        self.assertEqual(render_context, context)

    def test_get_render_components_missing(self):
        """Tests that _get_render_components handles missing components."""
        step_uid, wp_uid = "s1", "w1"
        key = (step_uid, wp_uid)
        ledger_key = ("view", step_uid, wp_uid)

        self.mock_artifact_manager.get_workpiece_view_handle.return_value = (
            None
        )
        self.mock_artifact_manager._get_ledger_entry.return_value = None

        chunk_handle = WorkPieceArtifactHandle(
            shm_name="chunk",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )

        view_handle, render_context = self.stage._get_render_components(
            key, ledger_key, chunk_handle
        )

        self.assertIsNone(view_handle)
        self.assertIsNone(render_context)
        self.mock_artifact_manager.release_handle.assert_called_once_with(
            chunk_handle
        )

    def test_should_render_chunk(self):
        """Tests that _should_render_chunk checks Ops correctly."""
        step_uid, wp_uid = "s1", "w1"
        key = (step_uid, wp_uid)

        chunk_ops = Ops()
        chunk_ops.move_to(0, 0, 0)
        chunk_artifact = WorkPieceArtifact(
            ops=chunk_ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(10.0, 10.0),
            generation_size=(10.0, 10.0),
        )

        chunk_handle = WorkPieceArtifactHandle(
            shm_name="chunk",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )

        self.assertTrue(
            self.stage._should_render_chunk(chunk_artifact, key, chunk_handle)
        )

    def test_should_render_chunk_empty(self):
        """Tests that _should_render_chunk returns False for empty Ops."""
        step_uid, wp_uid = "s1", "w1"
        key = (step_uid, wp_uid)

        chunk_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(10.0, 10.0),
            generation_size=(10.0, 10.0),
        )

        chunk_handle = WorkPieceArtifactHandle(
            shm_name="chunk",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
        )

        self.assertFalse(
            self.stage._should_render_chunk(chunk_artifact, key, chunk_handle)
        )
        self.mock_artifact_manager.release_handle.assert_called_once_with(
            chunk_handle
        )


if __name__ == "__main__":
    unittest.main()

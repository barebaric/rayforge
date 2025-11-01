import pytest
from unittest.mock import MagicMock, PropertyMock

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.artifact import (
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    WorkPieceArtifactHandle,
)
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.step_stage import StepPipelineStage
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)


@pytest.fixture
def mock_task_mgr():
    """Provides a mock TaskManager that captures created tasks."""
    mock_mgr = MagicMock()

    class MockTask:
        def __init__(self, key, when_done, when_event):
            self.key = key
            self.when_done = when_done
            self.when_event = when_event

    def run_process_mock(target_func, *args, **kwargs):
        task = MockTask(
            kwargs.get("key"),
            kwargs.get("when_done"),
            kwargs.get("when_event"),
        )
        # Store the task on the manager so the test can access it
        mock_mgr.last_created_task = task
        return task

    mock_mgr.run_process.side_effect = run_process_mock
    return mock_mgr


@pytest.fixture
def mock_artifact_cache():
    """Provides a mock ArtifactCache with correctly instantiated handles."""
    cache = MagicMock()
    cache.get_workpiece_handle.return_value = WorkPieceArtifactHandle(
        shm_name="wp_shm_1",
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
    )
    cache.has_step_render_handle.return_value = False
    # Make pops return a value for release testing.
    # Provide all required arguments for the handles.
    cache.pop_step_ops_handle.return_value = StepOpsArtifactHandle(
        shm_name="ops_shm_old",
        time_estimate=1.0,
        handle_class_name="StepOpsArtifactHandle",
        artifact_type_name="StepOpsArtifact",
    )
    cache.pop_step_render_handle.return_value = StepRenderArtifactHandle(
        shm_name="render_shm_old",
        handle_class_name="StepRenderArtifactHandle",
        artifact_type_name="StepRenderArtifact",
    )
    return cache


@pytest.fixture
def mock_doc_and_step():
    """Provides a mock Doc object with structure."""
    doc = MagicMock(spec=Doc)
    layer = MagicMock(spec=Layer)
    step = MagicMock(spec=Step)
    step.uid = "step1"
    step.visible = True
    step.per_step_transformers_dicts = []
    type(step).layer = PropertyMock(return_value=layer)

    wp = WorkPiece(name="wp1")
    wp.uid = "wp1"
    wp.set_size(20, 20)  # Different from source dimensions

    layer.workflow.steps = [step]
    layer.all_workpieces = [wp]
    doc.layers = [layer]
    return doc, step, wp


def _complete_step_task(task, time=42.0, gen_id=1):
    """Helper to simulate the completion of a step assembly task."""
    mock_task_obj = MagicMock()
    mock_task_obj.key = task.key
    mock_task_obj.id = task.id
    mock_task_obj.get_status.return_value = "completed"

    if task.when_event:
        render_handle = StepRenderArtifactHandle(
            shm_name="dummy_render",
            handle_class_name="StepRenderArtifactHandle",
            artifact_type_name="StepRenderArtifact",
        )
        render_event = {
            "handle_dict": render_handle.to_dict(),
            "generation_id": gen_id,
        }
        task.when_event(mock_task_obj, "render_artifact_ready", render_event)

        ops_handle = StepOpsArtifactHandle(
            shm_name="dummy_ops",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            time_estimate=time,
        )
        ops_event = {
            "handle_dict": ops_handle.to_dict(),
            "generation_id": gen_id,
        }
        task.when_event(mock_task_obj, "ops_artifact_ready", ops_event)

        time_event = {"time_estimate": time, "generation_id": gen_id}
        task.when_event(mock_task_obj, "time_estimate_ready", time_event)

    mock_task_obj.result.return_value = gen_id
    if task.when_done:
        task.when_done(mock_task_obj)


@pytest.mark.usefixtures("context_initializer")
class TestStepPipelineStage:
    def test_instantiation(self, mock_task_mgr, mock_artifact_cache):
        """Test that StepPipelineStage can be created."""
        stage = StepPipelineStage(mock_task_mgr, mock_artifact_cache)
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_cache is mock_artifact_cache

    def test_reconcile_triggers_assembly_for_missing_artifact(
        self,
        context_initializer,
        mock_task_mgr,
        mock_artifact_cache,
        mock_doc_and_step,
    ):
        """
        Tests that reconcile() starts a task if a step artifact is missing.
        """
        context_initializer.artifact_store.acquire = MagicMock(
            return_value=True
        )
        doc, step, _ = mock_doc_and_step
        stage = StepPipelineStage(mock_task_mgr, mock_artifact_cache)

        stage.reconcile(doc)

        mock_artifact_cache.has_step_render_handle.assert_called_with(step.uid)
        mock_task_mgr.run_process.assert_called_once()
        called_func = mock_task_mgr.run_process.call_args[0][0]
        assert called_func is make_step_artifact_in_subprocess

    def test_mark_stale_triggers_assembly(
        self,
        context_initializer,
        mock_task_mgr,
        mock_artifact_cache,
        mock_doc_and_step,
    ):
        """Tests that marking a step as stale triggers assembly."""
        context_initializer.artifact_store.acquire = MagicMock(
            return_value=True
        )
        doc, step, _ = mock_doc_and_step
        stage = StepPipelineStage(mock_task_mgr, mock_artifact_cache)

        stage.mark_stale_and_trigger(step)

        mock_task_mgr.run_process.assert_called_once()
        assert hasattr(mock_task_mgr, "last_created_task")

    def test_mark_stale_does_not_remove_render_artifact(
        self,
        context_initializer,
        mock_task_mgr,
        mock_artifact_cache,
        mock_doc_and_step,
    ):
        """
        Tests that staleness cleanup avoids flicker by not removing the
        render artifact.
        """
        context_initializer.artifact_store.acquire = MagicMock(
            return_value=True
        )
        _, step, _ = mock_doc_and_step
        stage = StepPipelineStage(mock_task_mgr, mock_artifact_cache)

        stage.mark_stale_and_trigger(step)

        # It should remove the ops artifact, but not the render one
        mock_artifact_cache.pop_step_ops_handle.assert_called_with(step.uid)
        mock_artifact_cache.pop_step_render_handle.assert_not_called()

    def test_full_assembly_flow_and_cleanup(
        self,
        context_initializer,
        mock_task_mgr,
        mock_artifact_cache,
        mock_doc_and_step,
    ):
        """
        Tests the full successful flow: trigger, dependency acquisition,
        event handling, and final dependency release on completion.
        """
        # Arrange
        mock_store = MagicMock()
        mock_store.acquire = MagicMock(return_value=True)
        mock_store.release = MagicMock()
        mock_store.adopt = MagicMock()
        # Replace the real artifact store with our mock
        context_initializer.artifact_store = mock_store
        doc, step, wp = mock_doc_and_step
        stage = StepPipelineStage(mock_task_mgr, mock_artifact_cache)

        # Setup mock workpiece handle to check for acquisition/release
        wp_handle = mock_artifact_cache.get_workpiece_handle.return_value

        render_handler = MagicMock()
        time_handler = MagicMock()
        finished_handler = MagicMock()
        stage.render_artifact_ready.connect(render_handler)
        stage.time_estimate_ready.connect(time_handler)
        stage.generation_finished.connect(finished_handler)

        # Act 1: Trigger assembly
        stage.mark_stale_and_trigger(step)

        # Assert 1: Task started and dependencies acquired
        mock_store.acquire.assert_called_once_with(wp_handle)
        mock_task_mgr.run_process.assert_called_once()
        task = mock_task_mgr.last_created_task
        assert step.uid in stage._active_tasks

        # Act 2: Simulate events from runner
        mock_task_obj = MagicMock(key=task.key)
        gen_id = 1

        render_handle = StepRenderArtifactHandle(
            shm_name="new_render",
            handle_class_name="StepRenderArtifactHandle",
            artifact_type_name="StepRenderArtifact",
        )
        task.when_event(
            mock_task_obj,
            "render_artifact_ready",
            {"handle_dict": render_handle.to_dict(), "generation_id": gen_id},
        )
        ops_handle = StepOpsArtifactHandle(
            shm_name="new_ops",
            time_estimate=123.4,
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
        )
        task.when_event(
            mock_task_obj,
            "ops_artifact_ready",
            {"handle_dict": ops_handle.to_dict(), "generation_id": gen_id},
        )
        task.when_event(
            mock_task_obj,
            "time_estimate_ready",
            {"time_estimate": 123.4, "generation_id": gen_id},
        )

        # Assert 2: Events processed correctly
        mock_store.adopt.assert_any_call(render_handle)
        mock_store.adopt.assert_any_call(ops_handle)
        mock_artifact_cache.put_step_render_handle.assert_called_with(
            step.uid, render_handle
        )
        mock_artifact_cache.put_step_ops_handle.assert_called_with(
            step.uid, ops_handle
        )
        render_handler.assert_called_once_with(stage, step=step)
        time_handler.assert_called_once_with(stage, step=step, time=123.4)
        assert stage.get_estimate(step.uid) == 123.4

        # Act 3: Simulate task completion
        mock_task_obj.get_status.return_value = "completed"
        mock_task_obj.result.return_value = gen_id
        task.when_done(mock_task_obj)

        # Assert 3: Cleanup is correct
        assert step.uid not in stage._active_tasks
        finished_handler.assert_called_once_with(
            stage, step=step, generation_id=gen_id
        )
        # CRITICAL: Assert that the original dependency handle was released
        mock_store.release.assert_called_with(wp_handle)

    def test_invalidate_cleans_up_everything(
        self,
        context_initializer,
        mock_task_mgr,
        mock_artifact_cache,
        mock_doc_and_step,
    ):
        """
        Tests that invalidating a step cancels tasks and removes all artifacts.
        """
        # Arrange: Put stage in a busy state
        mock_store = MagicMock()
        mock_store.acquire = MagicMock(return_value=True)
        mock_store.release = MagicMock()
        mock_store.adopt = MagicMock()
        # Replace the real artifact store with our mock
        context_initializer.artifact_store = mock_store
        _, step, _ = mock_doc_and_step
        stage = StepPipelineStage(mock_task_mgr, mock_artifact_cache)
        stage.mark_stale_and_trigger(step)
        assert step.uid in stage._active_tasks

        # Reset the mock after the setup phase (mark_stale_and_trigger)
        # to isolate the test to only the 'invalidate' call's behavior.
        mock_artifact_cache.reset_mock()

        # Act
        stage.invalidate(step.uid)

        # Assert
        mock_task_mgr.cancel_task.assert_called_once()
        assert step.uid not in stage._active_tasks

        # pop_... returns the mock handles, which should then be released
        mock_artifact_cache.pop_step_ops_handle.assert_called_with(step.uid)
        mock_artifact_cache.pop_step_render_handle.assert_called_with(step.uid)
        mock_store.release.assert_any_call(
            mock_artifact_cache.pop_step_ops_handle.return_value
        )
        mock_store.release.assert_any_call(
            mock_artifact_cache.pop_step_render_handle.return_value
        )
        mock_artifact_cache.invalidate_for_job.assert_called_once()

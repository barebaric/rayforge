import pytest
import logging
from unittest.mock import MagicMock, ANY
from pathlib import Path
import asyncio
from rayforge.image import SVG_RENDERER
from rayforge.context import get_context
from rayforge.core.doc import Doc
from rayforge.core.geo import Geometry
from rayforge.core.ops import Ops
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.pipeline import Pipeline
from rayforge.pipeline.steps import create_contour_step
from rayforge.pipeline.artifact import (
    ArtifactKey,
    JobArtifact,
    StepOpsArtifact,
    StepRenderArtifact,
    WorkPieceArtifactHandle,
)
from rayforge.pipeline.stage.workpiece_runner import (
    make_workpiece_artifact_in_subprocess,
)
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)
from rayforge.pipeline.stage.job_runner import make_job_artifact_in_subprocess


logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _zero_debounce(zero_debounce_delay):
    """Apply zero debounce delay to all tests in this file."""
    pass


@pytest.fixture
def real_workpiece():
    """Creates a lightweight WorkPiece with transforms, but no source."""
    workpiece = WorkPiece(name="real_workpiece.svg")
    # Importer will set size and pos, we simulate it in the setup helper.
    return workpiece


@pytest.fixture
def doc():
    d = Doc()
    # Get the active layer (the first workpiece layer) and clear its steps
    active_layer = d.active_layer
    assert active_layer.workflow is not None
    active_layer.workflow.set_steps([])
    return d


@pytest.mark.usefixtures("context_initializer")
class TestPipeline:
    # This data is used by multiple tests to create the SourceAsset.
    svg_data = b"""
    <svg width="50mm" height="30mm" xmlns="http://www.w3.org/2000/svg">
    <rect width="50" height="30" />
    </svg>"""

    def _setup_doc_with_workpiece(self, doc, workpiece):
        """Helper to correctly link a workpiece to a source within a doc."""
        source = SourceAsset(
            Path(workpiece.name),
            original_data=self.svg_data,
            renderer=SVG_RENDERER,
        )
        doc.add_asset(source)
        gen_config = SourceAssetSegment(
            source_asset_uid=source.uid,
            pristine_geometry=Geometry(),
            vectorization_spec=PassthroughSpec(),
        )
        workpiece.source_segment = gen_config
        # Simulate importer setting the size and pos
        workpiece.set_size(50, 30)
        workpiece.pos = 10, 20
        doc.active_layer.add_workpiece(workpiece)
        return doc.active_layer

    def _complete_all_tasks(
        self, mock_task_mgr, workpiece_handle, step_time=42.0
    ):
        """
        Helper to find and complete all outstanding tasks to bring the
        pipeline to an idle state. Simulates the new event-driven flow.
        """
        processed_tasks = set()
        while True:
            # Filter by object identity to handle multiple tasks with same key
            tasks_to_process = [
                t
                for t in mock_task_mgr.created_tasks
                if id(t) not in processed_tasks
            ]
            if not tasks_to_process:
                break

            for task_info in tasks_to_process:
                # Use the actual task object that the stage is holding
                task_obj = task_info.returned_task_obj
                task_obj.key = task_info.key
                task_obj.get_status.return_value = "completed"
                task_obj.result.return_value = None

                if task_info.target is make_job_artifact_in_subprocess:
                    if task_info.when_event:
                        store = get_context().artifact_store
                        job_artifact = JobArtifact(ops=Ops(), distance=0.0)
                        job_handle = store.put(job_artifact)
                        # Extract gen_id from kwargs
                        gen_id = task_info.kwargs.get("generation_id")
                        event_data = {
                            "handle_dict": job_handle.to_dict(),
                            "generation_id": gen_id,
                        }
                        task_info.when_event(
                            task_obj, "artifact_created", event_data
                        )
                    if task_info.when_done:
                        task_info.when_done(task_obj)

                elif task_info.target is make_step_artifact_in_subprocess:
                    gen_id = task_info.args[3]
                    task_obj.result.return_value = gen_id
                    if task_info.when_event:
                        # Create real artifacts and handles so the SHM blocks
                        # exist and can be adopted by the main process store.
                        store = get_context().artifact_store
                        render_artifact = StepRenderArtifact()
                        render_handle = store.put(render_artifact)
                        ops_artifact = StepOpsArtifact(ops=Ops())
                        ops_handle = store.put(ops_artifact)

                        # 1. Simulate render artifact event
                        render_event = {
                            "handle_dict": render_handle.to_dict(),
                            "generation_id": gen_id,
                        }
                        task_info.when_event(
                            task_obj,
                            "render_artifact_ready",
                            render_event,
                        )

                        # 2. Simulate ops artifact event
                        ops_event = {
                            "handle_dict": ops_handle.to_dict(),
                            "generation_id": gen_id,
                        }
                        task_info.when_event(
                            task_obj, "ops_artifact_ready", ops_event
                        )

                        # 3. Simulate time estimate event
                        time_event = {
                            "time_estimate": step_time,
                            "generation_id": gen_id,
                        }
                        task_info.when_event(
                            task_obj, "time_estimate_ready", time_event
                        )
                    if task_info.when_done:
                        task_info.when_done(task_obj)

                elif task_info.target is make_workpiece_artifact_in_subprocess:
                    gen_id = task_info.args[7]
                    task_obj.result.return_value = gen_id
                    if task_info.when_event:
                        event_data = {
                            "handle_dict": workpiece_handle.to_dict(),
                            "generation_id": gen_id,
                        }
                        task_info.when_event(
                            task_obj, "artifact_created", event_data
                        )
                    if task_info.when_done:
                        task_info.when_done(task_obj)

                processed_tasks.add(id(task_info))

        mock_task_mgr.created_tasks.clear()

    def test_generate_job_fire_and_forget(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        """
        Tests that the fire-and-forget generate_job method correctly
        delegates to the callback-based version.
        """
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Mock generate_job_artifact to verify it's called
        pipeline.generate_job_artifact = MagicMock()

        # Act
        pipeline.generate_job()

        # Assert
        pipeline.generate_job_artifact.assert_called_once()
        # Check that it was called with a no-op callback
        assert callable(
            pipeline.generate_job_artifact.call_args.kwargs["when_done"]
        )

    def test_generate_job_artifact_no_machine(
        self, doc, mock_task_mgr, context_initializer
    ):
        """Tests that job generation fails if no machine is configured."""
        # Arrange
        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        pipeline._machine = None  # type: ignore

        callback_mock = MagicMock()

        # Act
        pipeline.generate_job_artifact(when_done=callback_mock)

        # Assert
        callback_mock.assert_called_once_with(ANY, ANY)
        error = callback_mock.call_args[0][1]
        assert isinstance(error, RuntimeError)
        assert "No machine is configured" in str(error)

    def test_generate_job_artifact_missing_dependencies(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        """
        Tests that job generation fails if step artifacts are not ready.
        """
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        callback_mock = MagicMock()

        # Act
        pipeline.generate_job_artifact(when_done=callback_mock)

        # Assert
        callback_mock.assert_called_once_with(None, ANY)
        error = callback_mock.call_args[0][1]
        assert isinstance(error, RuntimeError)
        assert "Job dependencies are not ready" in str(error)

    @pytest.mark.asyncio
    async def test_rapid_invalidation_does_not_corrupt_busy_state(
        self, doc, real_workpiece, task_mgr, context_initializer
    ):
        """
        Black-box integration test that simulates a rapid invalidation
        cancelling an in-progress task and starting a new one. This test
        verifies that the pipeline correctly handles rapid invalidations
        without corrupting its busy state, using the real task manager
        and subprocess execution.
        """
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        mock_processing_state_handler = MagicMock()

        # Act 1: Create pipeline with an empty doc, so it's idle.
        pipeline = Pipeline(
            doc=Doc(),
            task_manager=task_mgr,
            artifact_store=context_initializer.artifact_store,
            machine=context_initializer.machine,
        )
        pipeline.processing_state_changed.connect(
            mock_processing_state_handler
        )

        assert pipeline.is_busy is False

        # Act 2: Set the doc property. This triggers reconcile_data() and
        # starts the first task.
        pipeline.doc = doc

        # Wait for the pipeline to become busy and for the first task to start
        await asyncio.sleep(0.5)
        assert pipeline.is_busy is True, (
            "Pipeline should be busy after doc set"
        )

        # Verify the state change signal was fired
        mock_processing_state_handler.assert_called_with(
            ANY, is_processing=True
        )

        # Act 3: Trigger a second regeneration, cancelling the
        # first task and starting a new one. Changing the power emits a
        # changed signal that bubbles to the doc and from there to the
        # pipeline. We wait a bit to ensure the first task has started before
        # invalidating.
        step.set_power(0.5)

        # Wait for tasks to settle - the rapid invalidation should cancel
        # the first task and start the second one.
        await asyncio.sleep(0.1)
        assert pipeline.is_busy is True, (
            "Pipeline should remain busy during rapid invalidation"
        )

        deadline = asyncio.get_running_loop().time() + 10.0
        while (
            pipeline.is_busy and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)

        assert pipeline.is_busy is False, (
            "Pipeline should be idle after all tasks complete"
        )

        assert mock_processing_state_handler.call_count >= 2, (
            f"Expected at least 2 state changes, got "
            f"{mock_processing_state_handler.call_count}"
        )

        # Verify the final state change was to idle
        last_call_args, last_call_kwargs = (
            mock_processing_state_handler.call_args_list[-1]
        )
        assert last_call_kwargs.get("is_processing") is False, (
            "Final state change should be to idle"
        )

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_reconcile_data_triggers_view_rerender_on_workpiece_resize(
        self, doc, real_workpiece, mock_task_mgr
    ):
        """
        Tests that pipeline.reconcile_data() triggers view re-rendering
        when a workpiece is resized.

        This reproduces the issue where rasters don't update properly
        after resize because:
        1. pipeline.reconcile_data() does NOT call view_stage.reconcile()
        2. If it did, request_view_render() would return early
           when old task is active
        """
        # Arrange: Set up doc with workpiece and step
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        ctx = get_context()
        step = create_contour_step(ctx)
        layer.workflow.add_step(step)

        # Create pipeline
        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            ctx.artifact_store,
            ctx.machine,
        )

        # Clear any tasks created during pipeline initialization
        mock_task_mgr.run_process.reset_mock()
        mock_task_mgr.created_tasks.clear()

        # Arrange: Simulate initial workpiece artifact being created
        initial_workpiece_handle = WorkPieceArtifactHandle(
            shm_name="initial_workpiece",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=False,  # Non-scalable (raster)
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(100, 100),
            generation_size=(50.0, 30.0),
        )
        ledger_key = ArtifactKey.for_workpiece(real_workpiece.uid)

        # Note: The pipeline initialization already triggered a task, so the
        # state is already PROCESSING with generation_id=1. We can just cache.
        pipeline.artifact_manager.cache_handle(
            ledger_key, initial_workpiece_handle, 1
        )

        # Arrange: Complete initial workpiece task
        # Don't call _complete_all_tasks here because we manually put the
        # handle. The pipeline will trigger view render when workpiece
        # artifact is adopted
        mock_task_mgr.run_process.reset_mock()
        mock_task_mgr.created_tasks.clear()

        # Act: Resize workpiece (double the size). This should trigger
        # a pipeline data reconcile.
        real_workpiece.set_size(20.0, 20.0)
        await asyncio.sleep(0)

        # Assert: Workpiece stage should detect size change and regenerate
        # Find the workpiece task that was created
        workpiece_tasks = [
            t
            for t in mock_task_mgr.created_tasks
            if t.target is make_workpiece_artifact_in_subprocess
        ]
        assert len(workpiece_tasks) == 1, (
            "Expected 1 workpiece task to be created after resize"
        )
        resized_workpiece_task = workpiece_tasks[0]

        # Verify the new workpiece has the resized dimensions
        new_workpiece_handle = WorkPieceArtifactHandle(
            shm_name="resized_workpiece",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
            is_scalable=False,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(200, 200),
            generation_size=(20.0, 20.0),
        )
        ledger_key = ArtifactKey.for_workpiece(real_workpiece.uid)
        # We just cache the result to simulate completion.
        pipeline.artifact_manager.cache_handle(
            ledger_key, new_workpiece_handle, 2
        )

        # Complete the resized workpiece task
        # The result should be the generation_id (2), not the number of chunks
        resized_workpiece_task.returned_task_obj.get_status.return_value = (
            "completed"
        )
        resized_workpiece_task.returned_task_obj.result.return_value = 2
        if resized_workpiece_task.when_done:
            resized_workpiece_task.when_done(
                resized_workpiece_task.returned_task_obj
            )

        # Reset mocks to check for view render task
        mock_task_mgr.run_process.reset_mock()
        mock_task_mgr.created_tasks.clear()

    def test_get_existing_job_handle_returns_none_when_no_job_cached(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        """
        Tests that get_existing_job_handle returns None when no job
        artifact has been cached yet.
        """
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Act - no job has been generated yet
        result = pipeline.get_existing_job_handle()

        # Assert
        assert result is None

    def test_get_existing_job_handle_returns_none_when_no_handle(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        """
        Tests that get_existing_job_handle returns None when no
        job handle exists.
        """
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Act - no job handle cached
        result = pipeline.get_existing_job_handle()

        # Assert
        assert result is None

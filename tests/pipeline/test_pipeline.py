import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from raygeo.geo import Geometry

from rayforge.core.doc import Doc
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER
from rayforge.pipeline.artifact import WorkPieceArtifactHandle
from rayforge.pipeline.pipeline import Pipeline

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

    def test_generate_job_fire_and_forget(
        self,
        doc,
        real_workpiece,
        mock_task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """
        Tests that the fire-and-forget generate_job method correctly
        delegates to the callback-based version.
        """
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = contour_step_class.create(context_initializer)
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
        """Tests that pipeline construction fails if no machine is
        configured."""
        # Arrange & Act & Assert
        with pytest.raises(RuntimeError, match="Machine is not configured"):
            Pipeline(
                doc,
                mock_task_mgr,
                context_initializer.artifact_store,
                None,  # type: ignore
            )

    @pytest.mark.asyncio
    async def test_generate_job_artifact_no_doc(
        self,
        doc,
        mock_task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Tests that job generation fails when no document is loaded."""
        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        pipeline.doc = None

        callback_mock = MagicMock()
        pipeline.generate_job_artifact(when_done=callback_mock)

        callback_mock.assert_called_once()
        handle, error = callback_mock.call_args[0]
        assert handle is None
        assert isinstance(error, RuntimeError)

    @pytest.mark.asyncio
    async def test_generate_job_artifact_no_workflow_content_fails_fast(
        self,
        doc,
        mock_task_mgr,
        context_initializer,
    ):
        """Job generation must fail fast without scheduling a rebuild when
        the document has no visible steps with workpieces.  Otherwise the
        UI settle->refresh loop keeps forcing rebuilds forever."""
        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        tasks_before = len(mock_task_mgr.created_tasks)

        callback_mock = MagicMock()
        pipeline.generate_job_artifact(when_done=callback_mock)

        callback_mock.assert_called_once()
        handle, error = callback_mock.call_args[0]
        assert handle is None
        assert isinstance(error, RuntimeError)
        assert len(mock_task_mgr.created_tasks) == tasks_before

    @pytest.mark.asyncio
    async def test_generate_job_artifact_hidden_steps_fails_fast(
        self,
        doc,
        real_workpiece,
        mock_task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """A workflow whose steps are all hidden cannot produce a job
        aggregate, so generation fails fast instead of rebuilding."""
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = contour_step_class.create(context_initializer)
        step.visible = False
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        tasks_before = len(mock_task_mgr.created_tasks)

        callback_mock = MagicMock()
        pipeline.generate_job_artifact(when_done=callback_mock)

        callback_mock.assert_called_once()
        handle, error = callback_mock.call_args[0]
        assert handle is None
        assert isinstance(error, RuntimeError)
        assert len(mock_task_mgr.created_tasks) == tasks_before

    @pytest.mark.asyncio
    async def test_rapid_invalidation_does_not_corrupt_busy_state(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """
        Black-box integration test that simulates a rapid invalidation
        cancelling an in-progress task and starting a new one. This test
        verifies that the pipeline correctly handles rapid invalidations
        without corrupting its busy state, using the real task manager
        and raygeo execution.
        """
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = contour_step_class.create(context_initializer)
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

        # Act 2: Set the doc property. This triggers a rebuild.
        pipeline.doc = doc

        # Wait for the pipeline to settle after the first rebuild.
        await asyncio.sleep(0.5)
        gen1 = pipeline.data_generation_id
        assert gen1 > 0, "Pipeline should have rebuilt after doc set"

        # Act 3: Trigger a second regeneration. Changing the power
        # emits a changed signal that bubbles to the doc and triggers
        # a rebuild.
        step.set_power(0.5)

        # Wait for tasks to settle - the rapid invalidation should
        # trigger a new rebuild.
        await asyncio.sleep(0.1)

        # The is_busy property flips to False as soon as the rebuild
        # finishes, but the matching "idle" processing_state_changed
        # signal is emitted a couple of main-thread hops later (via the
        # deferred _check_busy callback). Wait for the signal to catch
        # up with the property, otherwise the final-state assertion
        # below can observe a stale "busy" signal under load.
        def _last_signal_is_idle() -> bool:
            calls = mock_processing_state_handler.call_args_list
            if not calls:
                return False
            return calls[-1].kwargs.get("is_processing") is False

        deadline = asyncio.get_running_loop().time() + 10.0
        while (
            pipeline.is_busy or not _last_signal_is_idle()
        ) and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)

        assert pipeline.is_busy is False, (
            "Pipeline should be idle after all tasks complete"
        )

        gen2 = pipeline.data_generation_id
        assert gen2 > gen1, "Pipeline should have rebuilt after power change"

        # Verify the state change signal was fired for busy and idle.
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
    async def test_workpiece_resize_triggers_rebuild(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """
        Tests that resizing a workpiece triggers a pipeline rebuild
        and the artifact reflects the new size.
        """
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = contour_step_class.create(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Let the initial rebuild complete
        deadline = asyncio.get_running_loop().time() + 10.0
        while (
            pipeline.is_busy and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)

        handle1 = pipeline.get_artifact_handle(step.uid, real_workpiece.uid)
        assert handle1 is not None
        assert isinstance(handle1, WorkPieceArtifactHandle)
        size1 = handle1.generation_size

        # Resize the workpiece — this should trigger a new rebuild
        real_workpiece.set_size(20.0, 20.0)

        deadline = asyncio.get_running_loop().time() + 10.0
        while (
            pipeline.is_busy and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)

        handle2 = pipeline.get_artifact_handle(step.uid, real_workpiece.uid)
        assert handle2 is not None
        assert isinstance(handle2, WorkPieceArtifactHandle)
        assert handle2.generation_size != size1, (
            "Artifact generation_size should reflect the new workpiece "
            "size after resize"
        )

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    def test_get_existing_job_handle_returns_none_when_no_job_cached(
        self,
        doc,
        mock_task_mgr,
        context_initializer,
    ):
        """
        Tests that get_existing_job_handle returns None when no job
        artifact has been cached yet (no workflow content).
        """
        # Arrange - empty doc with no steps or workpieces
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
        self,
        doc,
        mock_task_mgr,
        context_initializer,
    ):
        """
        Tests that get_existing_job_handle returns None when no
        job handle exists (empty doc, no steps).
        """
        # Arrange - empty doc with no steps or workpieces
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

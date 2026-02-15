import pytest
import logging
import asyncio
from unittest.mock import MagicMock
from pathlib import Path
from rayforge.image import SVG_RENDERER
from rayforge.core.doc import Doc
from rayforge.core.source_asset import SourceAsset
from rayforge.core.workpiece import WorkPiece
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.geo import Geometry
from rayforge.core.ops import Ops
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.pipeline import Pipeline
from rayforge.pipeline.steps import create_contour_step
from rayforge.pipeline.stage.workpiece_runner import (
    make_workpiece_artifact_in_subprocess,
)
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)
from rayforge.pipeline.stage.job_runner import make_job_artifact_in_subprocess
from rayforge.pipeline.artifact import (
    ArtifactKey,
    ArtifactManager,
    JobArtifact,
    StepOpsArtifact,
    StepRenderArtifact,
    WorkPieceArtifact,
)
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.dag.node import ArtifactNode, NodeState
from rayforge.context import get_context


logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _zero_debounce(zero_debounce_delay):
    """Apply zero debounce delay to all tests in this file."""
    pass


@pytest.fixture
def real_workpiece():
    """Creates a lightweight WorkPiece with transforms, but no source."""
    workpiece = WorkPiece(name="real_workpiece.svg")
    return workpiece


@pytest.fixture
def doc():
    d = Doc()
    active_layer = d.active_layer
    assert active_layer.workflow is not None
    active_layer.workflow.set_steps([])
    return d


@pytest.mark.usefixtures("context_initializer")
class TestPipelineState:
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
        workpiece.set_size(50, 30)
        workpiece.pos = 10, 20
        doc.active_layer.add_workpiece(workpiece)
        return doc.active_layer

    def _complete_all_tasks(
        self, mock_task_mgr, workpiece_handle, step_time=42.0
    ):
        """Helper to find and complete all outstanding tasks."""
        processed_tasks = set()
        while True:
            tasks_to_process = [
                t
                for t in mock_task_mgr.created_tasks
                if id(t) not in processed_tasks
            ]
            if not tasks_to_process:
                break

            for task_info in tasks_to_process:
                task_obj = task_info.returned_task_obj
                task_obj.key = task_info.key
                task_obj.get_status.return_value = "completed"
                task_obj.result.return_value = None

                if task_info.target is make_job_artifact_in_subprocess:
                    if task_info.when_event:
                        store = get_context().artifact_store
                        job_artifact = JobArtifact(ops=Ops(), distance=0.0)
                        job_handle = store.put(job_artifact)
                        event_data = {"handle_dict": job_handle.to_dict()}
                        task_info.when_event(
                            task_obj, "artifact_created", event_data
                        )
                    if task_info.when_done:
                        task_info.when_done(task_obj)

                elif task_info.target is make_step_artifact_in_subprocess:
                    gen_id = task_info.args[3]
                    task_obj.result.return_value = gen_id
                    if task_info.when_event:
                        store = get_context().artifact_store
                        render_artifact = StepRenderArtifact()
                        render_handle = store.put(render_artifact)
                        ops_artifact = StepOpsArtifact(ops=Ops())
                        ops_handle = store.put(ops_artifact)

                        render_event = {
                            "handle_dict": render_handle.to_dict(),
                            "generation_id": gen_id,
                        }
                        task_info.when_event(
                            task_obj,
                            "render_artifact_ready",
                            render_event,
                        )

                        ops_event = {
                            "handle_dict": ops_handle.to_dict(),
                            "generation_id": gen_id,
                        }
                        task_info.when_event(
                            task_obj, "ops_artifact_ready", ops_event
                        )

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

    def test_shutdown_releases_all_artifacts(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
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

        # Simulate completion of a task to populate the cache
        task_info = mock_task_mgr.created_tasks[0]
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        task_obj_for_stage = task_info.returned_task_obj
        task_obj_for_stage.key = task_info.key
        task_obj_for_stage.get_status.return_value = "completed"
        task_obj_for_stage.result.return_value = 1

        try:
            event_data = {
                "handle_dict": handle.to_dict(),
                "generation_id": 1,
            }
            task_info.when_event(
                task_obj_for_stage, "artifact_created", event_data
            )
            task_info.when_done(task_obj_for_stage)

            # Verify handle is in cache
            assert (
                pipeline.get_artifact_handle(step.uid, real_workpiece.uid)
                is not None
            )

            # Act
            pipeline.shutdown()

            # Assert
            assert (
                pipeline.get_artifact_handle(step.uid, real_workpiece.uid)
                is None
            )
        finally:
            # handle should already be released by shutdown
            pass

    def test_doc_property_getter(
        self, doc, mock_task_mgr, context_initializer
    ):
        # Arrange
        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Act & Assert
        assert pipeline.doc is doc

    def test_doc_property_setter_with_same_doc(
        self, doc, mock_task_mgr, context_initializer
    ):
        # Arrange
        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Act - setting the same document should not cause issues
        pipeline.doc = doc

        # Assert
        assert pipeline.doc is doc

    def test_doc_property_setter_with_different_doc(
        self, doc, mock_task_mgr, context_initializer
    ):
        # Arrange
        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        new_doc = Doc()

        # Act
        pipeline.doc = new_doc

        # Assert
        assert pipeline.doc is new_doc

    def test_is_busy_property(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
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

        # Initial state - should be busy with one task
        assert pipeline.is_busy is True

        # Create a dummy workpiece artifact to allow the pipeline to proceed
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
        )
        wp_handle = get_context().artifact_store.put(artifact)

        # Act - complete all tasks
        self._complete_all_tasks(mock_task_mgr, wp_handle)

        # Assert - should not be busy anymore
        assert pipeline.is_busy is False

    @pytest.mark.asyncio
    async def test_pause_resume_functionality(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
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
        # Complete initial task to clear active_tasks
        # Create a workpiece handle for the helper
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        wp_handle = get_context().artifact_store.put(artifact)
        self._complete_all_tasks(mock_task_mgr, wp_handle)
        mock_task_mgr.run_process.reset_mock()  # Reset after initialization

        # Act - pause the pipeline
        pipeline.pause()
        assert pipeline.is_paused is True

        # Try to trigger regeneration - should not happen while paused
        real_workpiece.set_size(20, 20)
        mock_task_mgr.run_process.assert_not_called()

        # Resume the pipeline
        pipeline.resume()
        assert pipeline.is_paused is False
        await asyncio.sleep(0)  # Allow debounced task to run

        # Assert - reconciliation should happen after resume
        mock_task_mgr.run_process.assert_called()

    @pytest.mark.asyncio
    async def test_paused_context_manager(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
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
        # Complete initial task to clear active_tasks
        # Create a workpiece handle for the helper
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        wp_handle = get_context().artifact_store.put(artifact)
        self._complete_all_tasks(mock_task_mgr, wp_handle)
        mock_task_mgr.run_process.reset_mock()  # Reset after initialization

        # Act - use context manager
        with pipeline.paused():
            assert pipeline.is_paused is True
            # Try to trigger regeneration - should not happen while paused
            real_workpiece.set_size(20, 20)
            mock_task_mgr.run_process.assert_not_called()

        # Assert - should be resumed after context
        assert pipeline.is_paused is False
        await asyncio.sleep(0)  # Allow debounced task to run
        # Reconciliation should happen after resume
        mock_task_mgr.run_process.assert_called()

    def test_is_paused_property(self, doc, mock_task_mgr, context_initializer):
        # Arrange
        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Initial state
        assert pipeline.is_paused is False

        # After pause
        pipeline.pause()
        assert pipeline.is_paused is True

        # After resume
        pipeline.resume()
        assert pipeline.is_paused is False

    def test_preview_time_updated_signal_is_correct(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        """
        Tests the new end-to-end time estimation by checking the final
        signal received by the UI.
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

        # Create a dummy workpiece artifact to allow the pipeline to proceed
        wp_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        wp_handle = get_context().artifact_store.put(wp_artifact)

        mock_handler = MagicMock()
        pipeline.job_time_updated.connect(mock_handler)

        # Act
        try:
            # Complete all tasks, simulating a time of 55.5s for the step
            self._complete_all_tasks(mock_task_mgr, wp_handle, step_time=55.5)

            # Assert
            # The handler is called multiple times (e.g., initially with None)
            # We check the final call to see if it received the correct value.
            mock_handler.assert_called()
            last_call_args, last_call_kwargs = mock_handler.call_args_list[-1]
            assert last_call_kwargs.get("total_seconds") == 55.5
        finally:
            get_context().artifact_store.release(wp_handle)


class TestStateConsistency:
    """Tests for state consistency between DAG nodes and ArtifactManager."""

    @pytest.fixture
    def manager(self):
        """Set up a fresh manager and mock store."""
        mock_store = MagicMock(spec=ArtifactStore)
        return ArtifactManager(mock_store)

    def test_node_state_queries_manager(self, manager):
        """Test that node state queries the manager when available."""
        key = ArtifactKey.for_workpiece("wp1")
        mock_handle = MagicMock()
        mock_handle.shm_name = "shm_wp1"
        manager.cache_handle(key, mock_handle, 1)

        node = ArtifactNode(
            key=key,
            generation_id=1,
            _artifact_manager=manager,
        )

        assert node.state == NodeState.VALID

    def test_node_state_updates_manager(self, manager):
        """Test that setting node state updates the manager."""
        key = ArtifactKey.for_workpiece("wp1")
        manager.declare_generation({key}, 1)

        node = ArtifactNode(
            key=key,
            generation_id=1,
            _artifact_manager=manager,
        )

        node.state = NodeState.PROCESSING

        assert manager.get_state(key, 1) == NodeState.PROCESSING

    def test_state_propagates_to_node_after_invalidation(self, manager):
        """Test that invalidation state is visible through node."""
        key = ArtifactKey.for_workpiece("wp1")
        mock_handle = MagicMock()
        mock_handle.shm_name = "shm_wp1"
        manager.cache_handle(key, mock_handle, 1)

        node = ArtifactNode(
            key=key,
            generation_id=1,
            _artifact_manager=manager,
        )
        assert node.state == NodeState.VALID

        manager.invalidate_for_workpiece(key)

        assert node.state == NodeState.DIRTY

    def test_multiple_nodes_share_state_via_manager(self, manager):
        """Test that multiple nodes for same key share state via manager."""
        key = ArtifactKey.for_workpiece("wp1")
        mock_handle = MagicMock()
        mock_handle.shm_name = "shm_wp1"
        manager.cache_handle(key, mock_handle, 1)

        node1 = ArtifactNode(
            key=key,
            generation_id=1,
            _artifact_manager=manager,
        )
        node2 = ArtifactNode(
            key=key,
            generation_id=1,
            _artifact_manager=manager,
        )

        assert node1.state == NodeState.VALID
        assert node2.state == NodeState.VALID

        node1.state = NodeState.PROCESSING

        assert node1.state == NodeState.PROCESSING
        assert node2.state == NodeState.PROCESSING

    def test_generation_lifecycle_state_changes(self, manager):
        """Test state changes through a typical generation lifecycle."""
        key = ArtifactKey.for_workpiece("wp1")

        manager.declare_generation({key}, 1)
        node = ArtifactNode(
            key=key,
            generation_id=1,
            _artifact_manager=manager,
        )
        assert node.state == NodeState.DIRTY

        node.state = NodeState.PROCESSING
        assert node.state == NodeState.PROCESSING

        mock_handle = MagicMock()
        mock_handle.shm_name = "shm_wp1"
        manager.cache_handle(key, mock_handle, 1)
        assert node.state == NodeState.VALID

        manager.invalidate_for_workpiece(key)
        assert node.state == NodeState.DIRTY

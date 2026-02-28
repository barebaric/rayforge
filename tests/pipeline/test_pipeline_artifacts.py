import pytest
import logging
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
from rayforge.pipeline.steps import ContourStep
from rayforge.pipeline.artifact import (
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
)
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
class TestPipelineArtifacts:
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

    def test_get_artifact_handle(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = ContourStep.create(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Act & Assert - No handle initially
        assert (
            pipeline.get_artifact_handle(step.uid, real_workpiece.uid) is None
        )

        # Simulate a completed task
        task_info = mock_task_mgr.created_tasks[0]
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
            generation_id=1,
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

            # Act & Assert - Should return the handle
            retrieved_handle = pipeline.get_artifact_handle(
                step.uid, real_workpiece.uid
            )
            assert retrieved_handle is not None
            assert isinstance(retrieved_handle, WorkPieceArtifactHandle)
            assert retrieved_handle.generation_size == real_workpiece.size
        finally:
            get_context().artifact_store.release(handle)

    def test_get_scaled_ops(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = ContourStep.create(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Act & Assert - No ops initially
        world_transform = real_workpiece.get_world_transform()
        assert (
            pipeline.get_scaled_ops(
                step.uid, real_workpiece.uid, world_transform
            )
            is None
        )

        # Simulate a completed task with scalable artifact
        task_info = mock_task_mgr.created_tasks[0]
        expected_ops = Ops()
        expected_ops.move_to(0, 0, 0)
        expected_ops.line_to(10, 10, 0)

        artifact = WorkPieceArtifact(
            ops=expected_ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
            generation_id=1,
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

            # Act
            scaled_ops = pipeline.get_scaled_ops(
                step.uid, real_workpiece.uid, world_transform
            )

            # Assert
            assert scaled_ops is not None
            assert len(scaled_ops) == 2  # MoveTo + LineTo
        finally:
            get_context().artifact_store.release(handle)

    def test_get_scaled_ops_with_stale_non_scalable_artifact(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = ContourStep.create(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Simulate a completed task with non-scalable artifact at
        # different size
        task_info = mock_task_mgr.created_tasks[0]
        original_size = (25, 15)  # Different from workpiece size
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,  # Not scalable
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=original_size,
            generation_size=original_size,
            generation_id=1,
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

            # Act - Try to get scaled ops for different size
            world_transform = real_workpiece.get_world_transform()
            scaled_ops = pipeline.get_scaled_ops(
                step.uid, real_workpiece.uid, world_transform
            )

            # Assert - Should return None for stale non-scalable artifact
            assert scaled_ops is None
        finally:
            get_context().artifact_store.release(handle)

    def test_get_artifact(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = ContourStep.create(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Act & Assert - No artifact initially
        assert pipeline.get_artifact(step, real_workpiece) is None

        # Simulate a completed task
        task_info = mock_task_mgr.created_tasks[0]
        expected_ops = Ops()
        expected_ops.move_to(0, 0, 0)
        expected_ops.line_to(10, 10, 0)

        artifact = WorkPieceArtifact(
            ops=expected_ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
            generation_id=1,
        )
        handle = get_context().artifact_store.put(artifact)
        task_obj_for_stage = task_info.returned_task_obj
        task_obj_for_stage.key = task_info.key
        task_obj_for_stage.get_status.return_value = "completed"
        gen_id = pipeline._data_generation_id
        task_obj_for_stage.result.return_value = gen_id

        try:
            event_data = {
                "handle_dict": handle.to_dict(),
                "generation_id": gen_id,
            }
            task_info.when_event(
                task_obj_for_stage, "artifact_created", event_data
            )
            task_info.when_done(task_obj_for_stage)

            # Act
            retrieved_artifact = pipeline.get_artifact(step, real_workpiece)

            # Assert
            assert retrieved_artifact is not None
            assert retrieved_artifact.is_scalable is True
            assert len(retrieved_artifact.ops) == 2  # MoveTo + LineTo
            assert retrieved_artifact.source_dimensions == real_workpiece.size
        finally:
            get_context().artifact_store.release(handle)

    def test_get_artifact_with_stale_non_scalable_artifact(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = ContourStep.create(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Simulate a completed task with non-scalable artifact at
        # different size
        task_info = mock_task_mgr.created_tasks[0]
        original_size = (25, 15)  # Different from workpiece size
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,  # Not scalable
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=original_size,
            generation_size=original_size,
            generation_id=1,
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

            # Act - Try to get artifact for different size
            retrieved_artifact = pipeline.get_artifact(step, real_workpiece)

            # Assert - Should return None for stale non-scalable artifact
            assert retrieved_artifact is None
        finally:
            get_context().artifact_store.release(handle)

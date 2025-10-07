import pytest
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path
from rayforge.shared.tasker.task import Task
from rayforge.image import SVG_RENDERER
from rayforge.core.doc import Doc
from rayforge.core.import_source import ImportSource
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import Ops
from rayforge.machine.models.machine import Laser, Machine
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.generator import OpsGenerator
from rayforge.pipeline.steps import create_contour_step
from rayforge.pipeline.artifact.base import Artifact, VertexData
from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.pipeline.steprunner import run_step_in_subprocess
from rayforge.pipeline.timerunner import run_time_estimation_in_subprocess


@pytest.fixture(autouse=True)
def setup_real_config(mocker):
    test_laser = Laser()
    test_laser.max_power = 1000
    test_machine = Machine()
    test_machine.dimensions = (200, 150)
    test_machine.max_cut_speed = 5000
    test_machine.max_travel_speed = 10000
    test_machine.acceleration = 1000  # Add acceleration for tests
    test_machine.heads.clear()
    test_machine.add_head(test_laser)

    class TestConfig:
        machine = test_machine

    test_config = TestConfig()
    mocker.patch("rayforge.config.config", test_config)
    mocker.patch("builtins._", lambda s: s, create=True)
    return test_config


@pytest.fixture
def mock_task_mgr():
    """
    Creates a MagicMock for the TaskManager.
    It no longer patches any modules.
    """
    mock_mgr = MagicMock()
    created_tasks_info = []

    class MockTask:
        def __init__(self, target, args, kwargs):
            self.target = target
            self.args = args
            self.kwargs = kwargs
            self.when_done = kwargs.get("when_done")
            self.key = kwargs.get("key")

    def run_process_mock(target_func, *args, **kwargs):
        task = MockTask(target_func, args, kwargs)
        created_tasks_info.append(task)
        return task

    mock_mgr.run_process = MagicMock(side_effect=run_process_mock)
    mock_mgr.created_tasks = created_tasks_info
    return mock_mgr


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


class TestOpsGenerator:
    # This data is used by multiple tests to create the ImportSource.
    svg_data = b"""
    <svg width="50mm" height="30mm" xmlns="http://www.w3.org/2000/svg">
    <rect width="50" height="30" />
    </svg>"""

    def _setup_doc_with_workpiece(self, doc, workpiece):
        """Helper to correctly link a workpiece to a source within a doc."""
        source = ImportSource(
            Path(workpiece.name),
            original_data=self.svg_data,
            renderer=SVG_RENDERER,
        )
        doc.add_import_source(source)
        workpiece.import_source_uid = source.uid
        # Simulate importer setting the size and pos
        workpiece.set_size(50, 30)
        workpiece.pos = 10, 20
        doc.active_layer.add_workpiece(workpiece)
        return doc.active_layer

    def test_reconcile_all_triggers_ops_generation(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        # Act
        # The OpsGenerator is now created with the mock manager injected.
        OpsGenerator(doc, mock_task_mgr)

        # Assert
        mock_task_mgr.run_process.assert_called_once()
        # Verify it was the correct kind of process
        called_func = mock_task_mgr.run_process.call_args[0][0]
        assert called_func is run_step_in_subprocess

    def test_generation_success_emits_signals_and_caches_result(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        generator = OpsGenerator(doc, mock_task_mgr)
        mock_task_mgr.run_process.assert_called_once()
        task_to_complete = mock_task_mgr.created_tasks[0]

        # Act
        expected_ops = Ops()
        expected_ops.move_to(0, 0, 0)
        expected_ops.line_to(1, 1, 0)

        # All artifacts now contain vertex data
        vertex_data = VertexData(
            powered_vertices=np.array([[0, 0, 0], [1, 1, 0]]),
            powered_colors=np.array([[1, 1, 1, 1], [1, 1, 1, 1]]),
        )

        expected_artifact = Artifact(
            ops=expected_ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
            vertex_data=vertex_data,
        )
        handle = ArtifactStore.put(expected_artifact)
        expected_result_dict = handle.to_dict()
        expected_result_tuple = (expected_result_dict, 1)

        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = task_to_complete.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = expected_result_tuple

        try:
            task_to_complete.when_done(mock_finished_task)

            # Assert
            cached_ops = generator.get_ops(step, real_workpiece)
            assert cached_ops is not None
            # MoveTo + LineTo
            assert len(cached_ops) == 2
        finally:
            ArtifactStore.release(handle)

    def test_generation_cancellation_is_handled(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        generator = OpsGenerator(doc, mock_task_mgr)
        # The constructor has already called run_process once.
        mock_task_mgr.run_process.assert_called_once()
        task_to_cancel = mock_task_mgr.created_tasks[0]

        # Act
        mock_cancelled_task = MagicMock(spec=Task)
        mock_cancelled_task.key = task_to_cancel.key
        mock_cancelled_task.get_status.return_value = "cancelled"
        task_to_cancel.when_done(mock_cancelled_task)

        # Assert
        assert generator.get_ops(step, real_workpiece) is None

    def test_step_change_triggers_time_estimation_not_regeneration(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)
        # Initial generation
        OpsGenerator(doc, mock_task_mgr)
        mock_task_mgr.run_process.assert_called_once()

        # Simulate completion of the initial Ops generation
        initial_task = mock_task_mgr.created_tasks[0]
        artifact = Artifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        )
        handle = ArtifactStore.put(artifact)
        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = initial_task.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = (handle.to_dict(), 1)

        try:
            initial_task.when_done(mock_finished_task)
            # This completion will trigger a time estimation task. We reset
            # the mock to ignore this and focus on the next action.
            mock_task_mgr.run_process.reset_mock()

            # Act
            step.set_power(0.5)

            # Assert
            mock_task_mgr.run_process.assert_called_once()
            called_func = mock_task_mgr.run_process.call_args[0][0]
            assert called_func is run_time_estimation_in_subprocess
        finally:
            ArtifactStore.release(handle)

    def test_workpiece_pos_change_triggers_time_estimation_only(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)
        OpsGenerator(doc, mock_task_mgr)  # Initial generation
        mock_task_mgr.run_process.assert_called_once()

        # Simulate the completion of the initial generation task
        initial_task = mock_task_mgr.created_tasks[0]
        artifact = Artifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        )
        handle = ArtifactStore.put(artifact)
        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = initial_task.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = (handle.to_dict(), 1)

        try:
            initial_task.when_done(mock_finished_task)
            # The completion of the ops task triggers a time estimation task.
            assert mock_task_mgr.run_process.call_count == 2
            assert (
                mock_task_mgr.created_tasks[1].target
                is run_time_estimation_in_subprocess
            )
            mock_task_mgr.run_process.reset_mock()

            # Act
            real_workpiece.pos = 50, 50

            # Assert: A new time estimation is triggered, but NOT an ops
            # regeneration
            mock_task_mgr.run_process.assert_called_once()
            called_func = mock_task_mgr.run_process.call_args[0][0]
            assert called_func is run_time_estimation_in_subprocess
        finally:
            ArtifactStore.release(handle)

    def test_workpiece_angle_change_triggers_time_estimation_only(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)
        OpsGenerator(doc, mock_task_mgr)  # Initial generation
        mock_task_mgr.run_process.assert_called_once()

        # Simulate the completion of the initial generation task
        initial_task = mock_task_mgr.created_tasks[0]
        artifact = Artifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        )
        handle = ArtifactStore.put(artifact)
        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = initial_task.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = (handle.to_dict(), 1)
        try:
            initial_task.when_done(mock_finished_task)
            # The completion of the ops task triggers a time estimation task.
            assert mock_task_mgr.run_process.call_count == 2
            assert (
                mock_task_mgr.created_tasks[1].target
                is run_time_estimation_in_subprocess
            )
            mock_task_mgr.run_process.reset_mock()

            # Act
            real_workpiece.angle = 45

            # Assert: A new time estimation is triggered, but NOT an ops
            # regeneration
            mock_task_mgr.run_process.assert_called_once()
            called_func = mock_task_mgr.run_process.call_args[0][0]
            assert called_func is run_time_estimation_in_subprocess
        finally:
            ArtifactStore.release(handle)

    def test_workpiece_size_change_triggers_regeneration(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)
        OpsGenerator(doc, mock_task_mgr)  # Initial generation
        mock_task_mgr.run_process.assert_called_once()  # Verify initial call

        # Simulate the completion of the initial generation task to populate
        # the cache.
        initial_task = mock_task_mgr.created_tasks[0]
        initial_artifact = Artifact(
            ops=Ops(),
            is_scalable=False,  # Not scalable to ensure size change matters
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            generation_size=real_workpiece.size,  # size it was generated for
        )
        handle = ArtifactStore.put(initial_artifact)
        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = initial_task.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = (
            handle.to_dict(),
            1,
        )
        try:
            initial_task.when_done(mock_finished_task)
            # The completion of the ops task triggers a time estimation task.
            mock_task_mgr.run_process.reset_mock()

            # Act
            real_workpiece.set_size(10, 10)

            # Assert
            # The `transform_changed` signal from set_size bubbles up, and the
            # generator should see that the world size has changed.
            mock_task_mgr.run_process.assert_called_once()
            # Verify it's a full regeneration, not just a time estimate
            called_func = mock_task_mgr.run_process.call_args[0][0]
            assert called_func is run_step_in_subprocess
        finally:
            ArtifactStore.release(handle)

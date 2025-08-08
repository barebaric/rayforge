import pytest
from unittest.mock import MagicMock
from rayforge.shared.tasker.task import Task
from rayforge.importer import SvgImporter
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import Ops, LineToCommand
from rayforge.machine.models.machine import Laser, Machine
from rayforge.core.doc import Doc
from rayforge.pipeline.generator import OpsGenerator
from rayforge.pipeline.steps import create_contour_step


@pytest.fixture(autouse=True)
def setup_real_config(mocker):
    test_laser = Laser()
    test_laser.max_power = 1000
    test_machine = Machine()
    test_machine.dimensions = (200, 150)
    test_machine.max_cut_speed = 5000
    test_machine.max_travel_speed = 10000
    test_machine.heads.clear()
    test_machine.add_head(test_laser)

    class TestConfig:
        machine = test_machine

    test_config = TestConfig()
    mocker.patch("rayforge.pipeline.steps.config", test_config)
    mocker.patch("rayforge.core.workpiece.config", test_config, create=True)
    mocker.patch("builtins._", lambda s: s, create=True)
    return test_config


@pytest.fixture
def mock_task_mgr(mocker):
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
    mocker.patch("rayforge.pipeline.generator.task_mgr", mock_mgr)
    return mock_mgr


@pytest.fixture
def real_workpiece():
    svg_data = b'''
    <svg width="10" height="10" xmlns="http://www.w3.org/2000/svg">
    <rect width="10" height="10" />
    </svg>'''
    workpiece = WorkPiece("real_workpiece", svg_data, SvgImporter)
    workpiece.size = 50, 30
    workpiece.pos = 10, 20
    return workpiece


@pytest.fixture
def doc():
    d = Doc()
    d.layers[0].workflow.set_steps([])
    return d


class TestOpsGenerator:
    def test_reconcile_all_triggers_ops_generation(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = doc.layers[0]
        step = create_contour_step(layer.workflow)
        layer.workflow.add_step(step)
        layer.add_workpiece(real_workpiece)

        # Act
        OpsGenerator(doc)

        # Assert
        mock_task_mgr.run_process.assert_called_once()

    def test_generation_success_emits_signals_and_caches_result(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = doc.layers[0]
        step = create_contour_step(layer.workflow)
        layer.workflow.add_step(step)
        layer.add_workpiece(real_workpiece)

        generator = OpsGenerator(doc)
        mock_task_mgr.run_process.assert_called_once()
        task_to_complete = mock_task_mgr.created_tasks[0]

        # Act
        expected_ops = Ops()
        expected_ops.commands.append(LineToCommand((1, 1, 0)))
        expected_pixel_size = (100, 100)
        expected_result = (expected_ops, expected_pixel_size)

        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = task_to_complete.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = expected_result

        task_to_complete.when_done(mock_finished_task)

        # Assert
        cached_ops = generator.get_ops(step, real_workpiece)
        assert cached_ops is not None
        assert len(cached_ops.commands) == 1

    def test_generation_cancellation_is_handled(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = doc.layers[0]
        step = create_contour_step(layer.workflow)
        layer.workflow.add_step(step)
        layer.add_workpiece(real_workpiece)

        generator = OpsGenerator(doc)
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

    def test_step_change_triggers_regeneration(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = doc.layers[0]
        step = create_contour_step(layer.workflow)
        layer.workflow.add_step(step)
        layer.add_workpiece(real_workpiece)
        OpsGenerator(
            doc
        )  # This creates the generator and runs the initial task.

        # Reset the mock to ignore the initial setup call.
        mock_task_mgr.run_process.reset_mock()

        # Act
        step.set_power(500)

        # Assert
        mock_task_mgr.run_process.assert_called_once()

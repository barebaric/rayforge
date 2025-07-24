import pytest
import asyncio
from unittest.mock import MagicMock
from rayforge.tasker import ExecutionContextProxy
from rayforge.models.workstep import Contour, _execute_workstep_in_subprocess
from rayforge.models.workpiece import WorkPiece
from rayforge.render import SVGRenderer
from rayforge.models.ops import (
    Ops,
    MoveToCommand,
    DisableAirAssistCommand,
    SetPowerCommand,
    EnableAirAssistCommand,
)
from rayforge.models.machine import Laser, Machine


@pytest.fixture(autouse=True)
def setup_real_config(mocker):
    """
    Fixture to set up and inject a real, predictable config for all tests.
    `autouse=True` ensures it runs for every test automatically.
    """
    test_laser = Laser()
    test_laser.max_power = 1000
    test_machine = Machine()
    test_machine.dimensions = (200, 150)
    test_machine.max_cut_speed = 5000
    test_machine.max_travel_speed = 10000
    test_machine.heads.clear()
    test_machine.add_head(test_laser)

    class TestConfig:
        def __init__(self):
            self.machine = test_machine

    test_config = TestConfig()
    mocker.patch("rayforge.models.workplan.config", test_config)
    # Patch the gettext function `_` which is expected to be a builtin.
    mocker.patch("builtins._", lambda s: s, create=True)
    return test_config


@pytest.fixture
def mock_task_mgr(mocker):
    """
    Mocks the task manager to control process execution synchronously in tests.
    """
    mock_mgr = MagicMock()
    created_tasks_info = []

    class MockTask:
        """A mock Task to simulate the real Task API for testing."""
        def __init__(self, target, args, callback):
            self.target = target
            self.args = args
            self.callback = callback
            self._status = "pending"
            self._result = None
            self._exception = None
            self._cancelled = False

        def run_sync(self):
            """Runs the target function synchronously."""
            if self._cancelled:
                self._status = "cancelled"
                self._exception = asyncio.CancelledError()
                if self.callback:
                    self.callback(self)
                return

            try:
                # The real function expects a proxy. We can create a mock one.
                mock_proxy = MagicMock(spec=ExecutionContextProxy)
                mock_proxy.sub_context.return_value = mock_proxy
                
                self._result = self.target(mock_proxy, *self.args)
                self._status = "completed"
            except Exception as e:
                self._status = "failed"
                self._exception = e
            
            # After running, invoke the callback to simulate completion.
            if self.callback:
                self.callback(self)

        def get_status(self):
            return self._status

        def result(self):
            if self._status == "completed":
                return self._result
            if self._status == "failed":
                raise self._exception
            if self._status == "cancelled":
                raise asyncio.CancelledError()

        def cancel(self):
            self._cancelled = True

    def run_process_mock(target_func, *args, key=None, when_done=None):
        task = MockTask(target_func, args, when_done)
        created_tasks_info.append(task)
        return task

    mock_mgr.run_process = MagicMock(side_effect=run_process_mock)
    mock_mgr.created_tasks = created_tasks_info
    mocker.patch("rayforge.models.workstep.task_mgr", mock_mgr)
    return mock_mgr


@pytest.fixture
def real_workpiece():
    """Creates a real WorkPiece with simple, traceable SVG content."""
    svg_data = b'<svg width="10" height="10" xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10" style="fill:rgb(0,0,0);" /></svg>'
    workpiece = WorkPiece("real_workpiece", svg_data, SVGRenderer)
    workpiece.size = (50, 30)
    workpiece.pos = (10, 20)
    return workpiece


@pytest.fixture
def contour_step(setup_real_config):
    """Creates a real Contour WorkStep instance."""
    config = setup_real_config
    step = Contour(
        laser=config.machine.heads[0],
        max_cut_speed=config.machine.max_cut_speed,
        max_travel_speed=config.machine.max_travel_speed,
    )
    step.opstransformers = []  # Clear transformers to isolate testing
    return step


class TestWorkStepGeneration:
    @pytest.mark.asyncio
    async def test_add_workpiece_triggers_ops_generation(self, contour_step, real_workpiece, mock_task_mgr):
        """Verify that adding a workpiece triggers the ops generation process."""
        # Act
        contour_step.add_workpiece(real_workpiece)

        # Assert
        mock_task_mgr.run_process.assert_called_once()
        call_args = mock_task_mgr.run_process.call_args.args
        assert call_args[0] == _execute_workstep_in_subprocess
        assert isinstance(call_args[1], dict)
        assert call_args[1]["uid"] == real_workpiece.uid

    @pytest.mark.asyncio
    async def test_generation_success_emits_signals_and_caches_result(self, contour_step, real_workpiece, mock_task_mgr):
        """Test that a successful ops generation emits signals and caches the result."""
        # Arrange
        start_handler = MagicMock()
        finish_handler = MagicMock()

        contour_step.ops_generation_starting.connect(start_handler)
        contour_step.ops_generation_finished.connect(finish_handler)
        contour_step.air_assist = True

        # Act: This will create and schedule a mock task
        contour_step.add_workpiece(real_workpiece)
        
        # Assert start signal is called immediately
        start_handler.assert_called_once_with(contour_step, workpiece=real_workpiece)
        assert not finish_handler.called
        assert len(mock_task_mgr.created_tasks) == 1

        # Act: Manually run the captured task, which will also invoke the callback
        task = mock_task_mgr.created_tasks[0]
        task.run_sync()

        # Assert
        finish_handler.assert_called_once_with(contour_step, workpiece=real_workpiece)

        # Check the final cached ops to ensure initial state was applied
        assert real_workpiece.uid in contour_step._ops_cache
        cached_ops, _ = contour_step._ops_cache[real_workpiece.uid]
        
        assert isinstance(cached_ops, Ops)
        assert any(isinstance(c, SetPowerCommand) for c in cached_ops)
        assert any(isinstance(c, EnableAirAssistCommand) for c in cached_ops)
        assert any(isinstance(c, DisableAirAssistCommand) for c in cached_ops)
        assert any(isinstance(c, MoveToCommand) for c in cached_ops)

    @pytest.mark.asyncio
    async def test_generation_cancellation_is_handled(self, contour_step, real_workpiece, mock_task_mgr):
        """Test that ops generation can be cancelled correctly."""
        # Arrange
        finish_handler = MagicMock()
        contour_step.ops_generation_finished.connect(finish_handler)
        
        # Act: This will schedule the coroutine via the mock task manager
        contour_step.add_workpiece(real_workpiece)

        # Get the created task and cancel it before it "runs"
        assert len(mock_task_mgr.created_tasks) == 1
        task = mock_task_mgr.created_tasks[0]
        task.cancel()
        
        # Run the task. Since it's cancelled, it should do nothing and then call the callback
        task.run_sync()

        # Assert
        # The 'finished' signal should still be called to notify listeners the attempt is over.
        finish_handler.assert_called_once_with(contour_step, workpiece=real_workpiece)
        # Verify the cache was set to None upon cancellation/failure.
        assert contour_step._ops_cache.get(real_workpiece.uid) == (None, None)

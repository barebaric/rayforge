import pytest
import asyncio
from unittest.mock import MagicMock
import cairo
from blinker import Signal
from rayforge.task import ExecutionContext
from rayforge.models.workstep import Contour
from rayforge.models.workpiece import WorkPiece
from rayforge.models.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    DisableAirAssistCommand,
    SetPowerCommand,
    EnableAirAssistCommand,
)
from rayforge.models.machine import Laser, Machine
from rayforge.opsproducer import OpsProducer


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
    return test_config


@pytest.fixture
def mock_task_mgr(mocker):
    """
    Mocks the task manager to control async task execution in tests.
    """
    mock_mgr = MagicMock()
    created_tasks = []

    def add_coroutine_mock(coro_func, *args, key=None, when_done=None):
        mock_context = ExecutionContext()
        coro_obj = coro_func(mock_context, *args)
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro_obj)
        created_tasks.append(task)
        return task

    mock_mgr.add_coroutine = MagicMock(side_effect=add_coroutine_mock)
    mock_mgr.created_tasks = created_tasks
    mocker.patch("rayforge.models.workplan.CancelledError", asyncio.CancelledError)
    mocker.patch("rayforge.models.workstep.task_mgr", mock_mgr)
    return mock_mgr


@pytest.fixture
def mock_workpiece():
    """Creates a MagicMock of a WorkPiece for isolated testing."""
    workpiece = MagicMock(spec=WorkPiece)
    workpiece.name = "mock_workpiece"
    workpiece.size = (50, 30)
    workpiece.pos = (10, 20)
    workpiece.get_current_size.return_value = (50, 30)
    workpiece.size_changed = Signal()
    workpiece.pos_changed = Signal()
    workpiece.changed = Signal()
    real_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    workpiece.render_chunk.return_value = iter([(real_surface, (0, 0))])
    return workpiece


@pytest.fixture
def mock_opsproducer():
    """Mocks the OpsProducer dependency."""
    producer = MagicMock(spec=OpsProducer)
    ops_chunk = Ops()
    ops_chunk.add(MoveToCommand(end=(1, 1)))
    ops_chunk.add(LineToCommand(end=(5, 5)))
    producer.run.return_value = ops_chunk
    producer.can_scale.return_value = False
    return producer


@pytest.fixture
def contour_step(mock_opsproducer):
    """Creates a Contour WorkStep instance."""
    step = Contour()
    step.opsproducer = mock_opsproducer
    step.opstransformers = []
    return step


class TestWorkStepAsync:
    @pytest.mark.asyncio
    async def test_execute_is_async_generator(self, contour_step, mock_workpiece):
        """Verify WorkStep.execute returns an async generator yielding Ops chunks."""
        # Act
        generator = contour_step.execute(mock_workpiece, check_cancelled=lambda: False)

        # Assert
        assert hasattr(generator, "__aiter__")
        chunks = [chunk async for chunk in generator]
        assert len(chunks) > 0
        assert isinstance(chunks[0][0], Ops)
        mock_workpiece.render_chunk.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_ops_and_cache_signals(self, contour_step, mock_workpiece, mock_task_mgr):
        """Test that _stream_ops_and_cache emits signals correctly and caches the result."""
        # Arrange
        start_handler = MagicMock()
        chunk_handler = MagicMock()
        finish_handler = MagicMock()

        chunk1 = Ops()
        chunk1.add(MoveToCommand(end=(1, 1)))

        # Mock the async generator to return our single data chunk
        async def mock_execute(*args, **kwargs):
            yield (chunk1, None, 1.0)

        contour_step.execute = mock_execute
        contour_step.ops_generation_starting.connect(start_handler)
        contour_step.ops_chunk_available.connect(chunk_handler)
        contour_step.ops_generation_finished.connect(finish_handler)
        contour_step.air_assist = True

        # Wait for the task created by add_workpiece to complete
        contour_step.add_workpiece(mock_workpiece)
        await asyncio.gather(*mock_task_mgr.created_tasks)

        # Assert
        start_handler.assert_called_once_with(contour_step, workpiece=mock_workpiece)
        assert chunk_handler.call_count == 1

        # Check the final cached ops to ensure initial state was still applied
        assert mock_workpiece in contour_step.workpiece_to_ops
        cached_ops, _ = contour_step.workpiece_to_ops[mock_workpiece]
        assert isinstance(cached_ops, Ops)
        assert any(isinstance(c, SetPowerCommand) for c in cached_ops)
        assert any(isinstance(c, EnableAirAssistCommand) for c in cached_ops)
        assert any(isinstance(c, DisableAirAssistCommand) for c in cached_ops)
        assert any(isinstance(c, MoveToCommand) for c in cached_ops)

        finish_handler.assert_called_once_with(contour_step, workpiece=mock_workpiece)


    @pytest.mark.asyncio
    async def test_stream_ops_handles_cancellation(self, contour_step, mock_workpiece, mock_task_mgr):
        """Test that ops generation can be cancelled correctly."""
        # Arrange
        finish_handler = MagicMock()
        pause_event = asyncio.Event()

        # This async generator will pause, allowing the test to cancel it mid-execution
        async def slow_pausing_generator(*args, **kwargs):
            chunk = Ops()
            chunk.add(MoveToCommand(end=(1, 1)))
            yield chunk, None, 0.5  # Yield the first chunk
            await pause_event.wait() # Pause here indefinitely

        contour_step.execute = slow_pausing_generator
        contour_step.ops_generation_finished.connect(finish_handler)

        # This will schedule the coroutine, which will run and hit the pause_event
        contour_step.add_workpiece(mock_workpiece)
        await asyncio.sleep(0)  # Allow the coroutine to start and hit the pause_event

        # Cancel the running task
        assert len(mock_task_mgr.created_tasks) == 1
        task = mock_task_mgr.created_tasks[-1]
        task.cancel()

        # Awaiting the cancelled task should now correctly raise the exception
        with pytest.raises(asyncio.CancelledError):
            await task

        # Assert
        finish_handler.assert_not_called()
        # Verify the cache was cleared correctly upon cancellation
        assert contour_step.workpiece_to_ops.get(mock_workpiece) == (None, None)

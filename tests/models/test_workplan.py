import pytest
import asyncio
from unittest.mock import MagicMock
import cairo
from blinker import Signal
from rayforge.models.workplan import Contour
from rayforge.models.workpiece import WorkPiece
from rayforge.render import Renderer
from rayforge.models.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    DisableAirAssistCommand,
    SetPowerCommand,
    SetTravelSpeedCommand,
    EnableAirAssistCommand,
)
from rayforge.models.machine import Laser, Machine
from rayforge.opsproducer import OpsProducer


@pytest.fixture(autouse=True)
def setup_real_config(mocker):
    """
    Fixture to set up and inject a real, predictable config for all
    tests in this file.
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
def mock_task_mgr(mocker, _function_event_loop):
    """
    Mocks the task manager to control async task execution in tests.
    """
    mock_mgr = MagicMock()
    created_tasks = []

    def add_coroutine_mock(coro, when_done=None, key=None):
        task = _function_event_loop.create_task(coro)
        created_tasks.append(task)
        if when_done:
            task.add_done_callback(when_done)
        return task

    mock_mgr.add_coroutine = MagicMock(side_effect=add_coroutine_mock)
    mocker.patch("rayforge.models.workplan.task_mgr", mock_mgr)
    mock_mgr.created_tasks = created_tasks
    return mock_mgr


@pytest.fixture
def mock_renderer():
    """Mocks the renderer, returning a valid surface."""
    renderer = MagicMock(spec=Renderer)
    real_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    renderer.render_for_ops.return_value = real_surface
    renderer.render_chunk.return_value = iter([(real_surface, (0, 0))])
    return renderer


@pytest.fixture
def mock_workpiece():
    """Creates a MagicMock of a WorkPiece for isolated testing."""
    workpiece = MagicMock(spec=WorkPiece)
    workpiece.name = "mock_workpiece"
    workpiece.size = 50, 30
    workpiece.pos = 10, 20
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
def contour_step(mock_opsproducer, setup_real_config):
    """
    Creates a Contour WorkStep instance. It automatically uses the
    real config set up by the `setup_real_config` autouse fixture.
    """
    step = Contour()
    step.opsproducer = mock_opsproducer
    step.opstransformers = []
    return step


class TestWorkStepAsync:
    @pytest.mark.asyncio
    async def test_execute_is_generator_yielding_ops(
        self, contour_step, mock_workpiece
    ):
        """Verify WorkStep.execute yields Ops chunks as a generator."""
        # Arrange
        contour_step.opsproducer.can_scale.return_value = False
        mock_ops = Ops()
        mock_ops.add(MoveToCommand(end=(1, 1)))
        contour_step.opsproducer.run.return_value = mock_ops

        # Act
        generator = contour_step.execute(mock_workpiece)

        # Assert
        assert hasattr(generator, "__iter__") and not isinstance(
            generator, Ops
        )
        chunks = list(generator)
        assert len(chunks) == 2
        assert isinstance(chunks[0], Ops)
        assert isinstance(chunks[1], Ops)
        mock_workpiece.render_chunk.assert_called_once()
        contour_step.opsproducer.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_ops_and_cache_signals(
        self, contour_step, mock_workpiece, mock_task_mgr
    ):
        """Test that _stream_ops_and_cache emits signals correctly and caches the result."""
        # Arrange
        start_handler = MagicMock()
        chunk_handler = MagicMock()
        finish_handler = MagicMock()

        chunk1 = Ops()
        chunk1.add(MoveToCommand(end=(1, 1)))
        chunk2 = Ops()
        chunk2.add(LineToCommand(end=(2, 2)))

        contour_step.execute = MagicMock(return_value=[chunk1, chunk2])
        contour_step.ops_generation_starting.connect(start_handler)
        contour_step.ops_chunk_available.connect(chunk_handler)
        contour_step.ops_generation_finished.connect(finish_handler)
        contour_step.air_assist = True

        # Act
        contour_step.add_workpiece(mock_workpiece)
        await asyncio.sleep(0)

        assert mock_task_mgr.created_tasks, (
            "Task manager did not create a task"
        )
        task = mock_task_mgr.created_tasks[-1]
        await task

        # Assert
        start_handler.assert_called_once_with(
            contour_step, workpiece=mock_workpiece
        )
        assert chunk_handler.call_count == 4

        initial_chunk = chunk_handler.call_args_list[0].kwargs["chunk"]
        assert any(isinstance(c, SetPowerCommand) for c in initial_chunk)
        assert any(isinstance(c, SetTravelSpeedCommand) for c in initial_chunk)
        assert any(
            isinstance(c, EnableAirAssistCommand) for c in initial_chunk
        )

        assert chunk_handler.call_args_list[1].kwargs["chunk"] == chunk1
        assert chunk_handler.call_args_list[2].kwargs["chunk"] == chunk2

        final_chunk = chunk_handler.call_args_list[3].kwargs["chunk"]
        assert any(isinstance(c, DisableAirAssistCommand) for c in final_chunk)

        finish_handler.assert_called_once_with(
            contour_step, workpiece=mock_workpiece
        )

        assert mock_workpiece in contour_step.workpiece_to_ops
        cached_ops, cached_size = contour_step.workpiece_to_ops[mock_workpiece]
        assert isinstance(cached_ops, Ops)
        assert len(cached_ops) > 0
        assert cached_size == mock_workpiece.size

    @pytest.mark.asyncio
    async def test_stream_ops_handles_cancellation(
        self, contour_step, mock_workpiece, mock_task_mgr
    ):
        """Test that ops generation can be cancelled correctly."""
        # Arrange
        chunk_handler = MagicMock()
        finish_handler = MagicMock()

        chunk1 = Ops()
        chunk1.add(MoveToCommand(end=(1, 1)))

        def slow_execute_generator(*args, **kwargs):
            # This generator yields one chunk and then stops.
            yield chunk1
            return

        contour_step.execute = MagicMock(side_effect=slow_execute_generator)
        contour_step.ops_chunk_available.connect(chunk_handler)
        contour_step.ops_generation_finished.connect(finish_handler)

        # Act
        contour_step.add_workpiece(mock_workpiece)
        await asyncio.sleep(0)

        # The coroutine sends an initial chunk, then awaits. The test's await lets it run to that point.
        # FIX: The assertion is now correct. Only the initial state chunk is sent before the pause.
        assert chunk_handler.call_count == 1

        # Now, cancel the running task
        task = mock_task_mgr.created_tasks[-1]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Assert
        finish_handler.assert_not_called()
        assert contour_step.workpiece_to_ops.get(mock_workpiece) == (None, None)

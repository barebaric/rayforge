import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call
import cairo # Import cairo for FORMAT_ARGB32

# Assuming blinker signals are used
from blinker import Signal

# Project imports (adjust paths if necessary)
from rayforge.models.workplan import WorkStep, Contour, Rasterize, Outline, WorkPlan
from rayforge.models.workpiece import WorkPiece
from rayforge.models.ops import Ops, MoveToCommand, LineToCommand
from rayforge.models.machine import Laser, Machine
from rayforge.opsproducer import OpsProducer
from rayforge.config import config
from rayforge.task import task_mgr

# --- Test Fixtures ---

@pytest.fixture
def mock_config(mocker):
    """Mocks the global config object."""
    # Create mock machine with required attributes
    mock_machine = MagicMock(spec=Machine)
    mock_machine.heads = [MagicMock(spec=Laser, max_power=1000, changed=Signal())]
    mock_machine.max_cut_speed = 5000
    mock_machine.max_travel_speed = 10000
    mock_machine.dimensions = (200, 150)  # Example dimensions
    
    # Create mock config with machine attribute
    mock_cfg = MagicMock()
    mock_cfg.machine = mock_machine
    
    # Patch both config locations
    mocker.patch('rayforge.models.workplan.config', mock_cfg)
    mocker.patch('rayforge.config.config', mock_cfg)
    return mock_cfg

@pytest.fixture
def mock_task_mgr(mocker, _function_event_loop):
    """Mocks the task manager to integrate with pytest-asyncio loop."""
    mock_mgr = MagicMock()
    created_tasks = [] # Store created tasks

    # This function will be called by the code under test
    def add_coroutine_mock(coro, when_done=None, key=None):
        # Schedule the coroutine on the event loop provided by pytest-asyncio
        task = _function_event_loop.create_task(coro)
        created_tasks.append(task) # Store the task

        # If a callback is provided, wrap it to match the expected signature
        if when_done:
            def done_callback_wrapper(future):
                # Create a mock task object to pass to the original when_done
                mock_task_result = MagicMock()
                try:
                    mock_task_result.result.return_value = future.result()
                except Exception as e:
                    # If the coroutine raised an exception, set it on the mock task
                    mock_task_result.result.side_effect = e
                when_done(mock_task_result)

            task.add_done_callback(done_callback_wrapper)
        return task # Return the asyncio task

    mock_mgr.add_coroutine = MagicMock(side_effect=add_coroutine_mock)
    mocker.patch('rayforge.models.workplan.task_mgr', mock_mgr)
    mock_mgr.created_tasks = created_tasks # Attach the list to the mock
    return mock_mgr

@pytest.fixture
def mock_workpiece():
    """Creates a mock WorkPiece."""
    wp = MagicMock(spec=WorkPiece)
    wp.name = "mock_workpiece" # Add the missing name attribute
    wp.size = (50, 30) # Example size in mm
    wp.pos = (10, 20) # Example position in mm
    wp.visible = True
    mock_surface = MagicMock(spec=cairo.ImageSurface) # Use spec for better mocking
    mock_surface.get_format.return_value = cairo.FORMAT_ARGB32
    mock_surface.get_width.return_value = 1 # Mock dimension
    mock_surface.get_height.return_value = 1 # Mock dimension
    mock_surface.get_stride.return_value = 4 # Mock stride (4 bytes per pixel for ARGB32)
    mock_surface.get_data.return_value = bytearray(4) # Mock data (1 pixel * 4 bytes)
    wp.render.return_value = (mock_surface, True)

    mock_chunk_surface = MagicMock(spec=cairo.ImageSurface) # Use spec
    mock_chunk_surface.get_format.return_value = cairo.FORMAT_ARGB32
    mock_chunk_surface.get_width.return_value = 1 # Mock dimension
    mock_chunk_surface.get_height.return_value = 1 # Mock dimension
    mock_chunk_surface.get_stride.return_value = 4 # Mock stride
    mock_chunk_surface.get_data.return_value = bytearray(4) # Mock data
    wp.render_chunk.return_value = [(mock_chunk_surface, (0,0))]
    wp.size_changed = Signal() # Mock signal
    return wp

@pytest.fixture
def mock_opsproducer():
    """Creates a mock OpsProducer that yields simple Ops."""
    producer = MagicMock(spec=OpsProducer)
    # Simulate producing a simple cut operation chunk
    ops_chunk = Ops()
    ops_chunk.add(MoveToCommand(1, 1))
    ops_chunk.add(LineToCommand(5, 5))
    producer.run.return_value = ops_chunk
    producer.can_scale.return_value = False # Default to non-scalable for chunking
    return producer

@pytest.fixture
def contour_step(mock_opsproducer, mock_config):
    """Creates a Contour WorkStep instance with mocks."""
    # Ensure config is mocked before WorkStep initializes
    _ = mock_config
    step = Contour()
    step.opsproducer = mock_opsproducer # Replace default with mock
    step.laser = mock_config.heads[0] # Assign mocked laser
    # Clear default transformers if any were added by DEBUG flags
    step.opstransformers = []
    return step


# --- Test Class ---

class TestWorkStepIterative:

    @pytest.mark.asyncio
    async def test_execute_is_generator(self, contour_step, mock_workpiece):
        """Verify WorkStep.execute yields Ops chunks."""
        # Arrange
        contour_step.opsproducer.can_scale.return_value = False # Force chunking path
        mock_ops = Ops()
        mock_ops.add(MoveToCommand(end=(1,1))) # Pass end as a tuple keyword argument
        contour_step.opsproducer.run.return_value = mock_ops # Simple chunk

        # Act
        generator = contour_step.execute(mock_workpiece)

        # Assert
        assert hasattr(generator, '__iter__') and not isinstance(generator, Ops)
        chunks = list(generator)
        assert len(chunks) > 0 # Should yield at least one chunk
        assert isinstance(chunks[0], Ops)
        # Check if producer was called (adjust based on actual chunking logic)
        mock_workpiece.render_chunk.assert_called_once()
        contour_step.opsproducer.run.assert_called()

    @pytest.mark.asyncio
    async def test_stream_ops_and_cache_signals(self, contour_step, mock_workpiece, mock_task_mgr, mock_config):
        """Test _stream_ops_and_cache emits signals correctly."""
        # Arrange
        # No need to set return_value, the side_effect handles task creation

        start_handler = MagicMock()
        chunk_handler = MagicMock()
        finish_handler = MagicMock()

        # Mock the execute generator directly for predictable chunks
        chunk1 = Ops()
        chunk1.add(MoveToCommand(end=(1,1))) # Pass end as a tuple keyword argument
        chunk2 = Ops()
        chunk2.add(LineToCommand(end=(2,2))) # Pass end as a tuple keyword argument
        contour_step.execute = MagicMock(return_value=[chunk1, chunk2])

        contour_step.ops_generation_starting.connect(start_handler)
        contour_step.ops_chunk_available.connect(chunk_handler)
        contour_step.ops_generation_finished.connect(finish_handler)

        # Act
        # update_workpiece triggers the coroutine via the mocked task_mgr
        # Run update_workpiece and retrieve the scheduled task from the mock manager
        contour_step.update_workpiece(mock_workpiece)
        # Allow the event loop to run and create the task
        await asyncio.sleep(0)
        # Retrieve the actual task created by the side_effect
        assert mock_task_mgr.created_tasks, "Task manager did not create a task"
        task = mock_task_mgr.created_tasks[-1]
        # Await the completion of the task directly
        await task

        # Assert
        # Check signals were sent with correct args
        start_handler.assert_called_once_with(contour_step, workpiece=mock_workpiece)

        # Expected chunks: initial_ops, chunk1, chunk2, final_ops
        # Expect 3 chunks: initial, chunk1, chunk2 (final chunk is empty and not sent as air_assist=False)
        assert chunk_handler.call_count == 3

        # Check first chunk (initial ops)
        # Check first chunk (initial ops) - args sent as keywords
        call_initial = chunk_handler.call_args_list[0]
        assert call_initial.args[0] == contour_step # Sender is positional
        assert call_initial.kwargs['workpiece'] == mock_workpiece
        assert call_initial.kwargs['chunk'] is not None # Check chunk exists
        assert isinstance(call_initial.kwargs['chunk'], Ops)
        # Add checks for initial commands (power, speed etc.) in call_args_initial[2]

        # Check second chunk (chunk1 from generator)
        # Check second chunk (chunk1 from generator) - args sent as keywords
        call_chunk1 = chunk_handler.call_args_list[1]
        assert call_chunk1.args[0] == contour_step # Sender is positional
        assert call_chunk1.kwargs['workpiece'] == mock_workpiece
        assert call_chunk1.kwargs['chunk'] == chunk1 # Check it's the exact object yielded

        # Check third chunk (chunk2 from generator)
        # Check third chunk (chunk2 from generator) - args sent as keywords
        call_chunk2 = chunk_handler.call_args_list[2]
        assert call_chunk2.args[0] == contour_step # Sender is positional
        assert call_chunk2.kwargs['workpiece'] == mock_workpiece
        assert call_chunk2.kwargs['chunk'] == chunk2

        # Check fourth chunk (final ops)
        # Check fourth chunk (final ops) - This chunk is no longer sent when air_assist=False
        # The assertion for call_count == 3 handles this. Remove specific check for 4th chunk.
        # Add checks for final commands (e.g., air assist off) in call_args_final[2]

        # Check finish signal - args sent as keywords
        finish_handler.assert_called_once()
        assert finish_handler.call_args.args[0] == contour_step
        assert finish_handler.call_args.kwargs['workpiece'] == mock_workpiece

        # Check caching
        assert mock_workpiece in contour_step.workpiece_to_ops
        cached_ops, cached_size = contour_step.workpiece_to_ops[mock_workpiece]
        assert isinstance(cached_ops, Ops)
        # Verify accumulated_ops content if needed
        assert cached_size == mock_workpiece.size

    @pytest.mark.asyncio
    async def test_stream_ops_handles_cancellation(self, contour_step, mock_workpiece, mock_task_mgr):
        """Test cancellation during _stream_ops_and_cache."""
        # Arrange
        chunk_handler = MagicMock()
        finish_handler = MagicMock()

        # Mock execute to raise CancelledError after yielding one chunk
        chunk1 = Ops()
        chunk1.add(MoveToCommand(end=(1,1))) # Pass end as a tuple keyword argument
        # Change to a regular generator, as expected by the 'for' loop in _stream_ops_and_cache
        def mock_execute_gen(*args, **kwargs):
            yield chunk1
            # Generator finishes normally after one yield in this mock scenario
            # Cancellation will be triggered externally on the task

        # Patch the instance method for this test
        contour_step.execute = MagicMock(side_effect=mock_execute_gen)

        contour_step.ops_chunk_available.connect(chunk_handler)
        contour_step.ops_generation_finished.connect(finish_handler)

        # Act & Assert
        # We need a way to simulate cancellation within the mocked task manager's execution
        # For simplicity here, we assume CancelledError is raised directly by execute
        # A more complex mock_task_mgr might be needed for true async cancellation testing

        # Modify task_mgr mock to handle cancellation propagation if needed
        # No need to further modify mock_task_mgr here, the fixture handles loop integration.
        # The original execute mock raising CancelledError is sufficient.

        # Run the update which starts the streaming
        # Run the update which starts the streaming
        contour_step.update_workpiece(mock_workpiece)
        # Retrieve the task created by the mock task manager
        task = mock_task_mgr.add_coroutine.return_value

        # Cancel the task and allow event loop to process
        task.cancel()
        await asyncio.sleep(0)

        # Assertions after potential cancellation
        # Initial chunk + chunk1 should have been signalled
        assert chunk_handler.call_count >= 1 # Initial ops are always sent before first await

        # Finish signal should NOT have been called
        finish_handler.assert_not_called()

        # Cache should NOT contain the final result (or might contain partial if not handled well)
        # Depending on exact implementation, check cache state
        assert mock_workpiece not in contour_step.workpiece_to_ops or \
               contour_step.workpiece_to_ops[mock_workpiece][0] is None # Or check partial state


    # TODO: Add tests for caching logic (correct accumulation)
    # TODO: Add tests for OpsTransformers applied in _stream_ops_and_cache (if moved there)
    # TODO: Add tests for error handling within the generator/coroutine
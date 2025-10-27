import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from rayforge.core.doc import Doc
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand
from rayforge.machine.driver.dummy import NoDeviceDriver
from rayforge.machine.models.machine import Machine


class TestDummyDriverCallback:
    """Test suite for the dummy driver's on_command_done callback."""

    @pytest.fixture
    def driver(self, context_initializer):
        """Provides a fresh NoDeviceDriver instance for each test."""
        return NoDeviceDriver(context_initializer)

    @pytest.fixture
    def machine(self, context_initializer):
        """Provides a default Machine instance."""
        return Machine(context_initializer)

    @pytest.fixture
    def doc(self):
        """Provides a fresh Doc instance for each test."""
        return Doc()

    @pytest.fixture
    def simple_ops(self):
        """Creates a simple Ops object with a few commands."""
        ops = Ops()
        ops.add(MoveToCommand((10.0, 10.0, 0.0)))
        ops.add(LineToCommand((20.0, 20.0, 0.0)))
        ops.add(MoveToCommand((30.0, 30.0, 0.0)))
        return ops

    @pytest.fixture
    def complex_ops(self):
        """Creates a more complex Ops object with various command types."""
        ops = Ops()
        ops.add(MoveToCommand((0.0, 0.0, 0.0)))
        ops.add(LineToCommand((10.0, 0.0, 0.0)))
        ops.add(LineToCommand((10.0, 10.0, 0.0)))
        ops.add(LineToCommand((0.0, 10.0, 0.0)))
        ops.add(LineToCommand((0.0, 0.0, 0.0)))
        return ops

    @pytest.mark.asyncio
    async def test_run_without_callback(
        self, driver, machine, doc, simple_ops
    ):
        """Test that run() works without providing a callback."""
        # Should not raise any exceptions
        await driver.run(simple_ops, machine, doc)

    @pytest.mark.asyncio
    async def test_run_with_callback(self, driver, machine, doc, simple_ops):
        """Test that run() calls the callback for each command."""
        callback_mock = MagicMock()

        await driver.run(simple_ops, machine, doc, callback_mock)

        # Verify callback was called for each command
        assert callback_mock.call_count == len(simple_ops)

        # Verify callback was called with correct op_index
        for i, call in enumerate(callback_mock.call_args_list):
            args, kwargs = call
            assert args[0] == i  # op_index

    @pytest.mark.asyncio
    async def test_run_with_async_callback(
        self, driver, machine, doc, simple_ops
    ):
        """Test that run() works with an async callback."""
        callback_mock = AsyncMock()

        await driver.run(simple_ops, machine, doc, callback_mock)

        # Verify callback was called for each command
        assert callback_mock.call_count == len(simple_ops)

        # Verify all callbacks were awaited
        for call in callback_mock.call_args_list:
            args, kwargs = call
            assert args[0] in range(len(simple_ops))

    @pytest.mark.asyncio
    async def test_callback_parameters_and_order(
        self, driver, machine, doc, simple_ops
    ):
        """Test that the callback receives op_index in the correct order."""
        received_indices = []

        def collect_indices(op_index):
            received_indices.append(op_index)

        await driver.run(simple_ops, machine, doc, collect_indices)

        # Verify we got a callback for all commands
        assert len(received_indices) == len(simple_ops)

        # Verify indices are correct and were received in order
        assert received_indices == list(range(len(simple_ops)))

    @pytest.mark.asyncio
    async def test_callback_with_empty_ops(self, driver, machine, doc):
        """Test that run() handles empty Ops correctly."""
        empty_ops = Ops()
        callback_mock = MagicMock()

        await driver.run(empty_ops, machine, doc, callback_mock)

        # Callback should not be called for empty ops
        callback_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_with_complex_ops(
        self, driver, machine, doc, complex_ops
    ):
        """Test that run() works correctly with more complex operations."""
        callback_mock = MagicMock()

        await driver.run(complex_ops, machine, doc, callback_mock)

        # Verify callback was called for each command
        assert callback_mock.call_count == len(complex_ops)

        # Verify all op indices were passed to the callback
        received_indices = [
            call.args[0] for call in callback_mock.call_args_list
        ]
        assert sorted(received_indices) == list(range(len(complex_ops)))

    @pytest.mark.asyncio
    async def test_callback_exception_handling(
        self, driver, machine, doc, simple_ops
    ):
        """Test that exceptions in the callback don't stop execution."""
        call_count = 0

        def failing_callback(op_index):
            nonlocal call_count
            call_count += 1
            if op_index == 1:
                raise ValueError("Test exception")

        # The driver should catch the exception and continue
        await driver.run(simple_ops, machine, doc, failing_callback)

        # Verify that the driver still attempted to call for all ops
        assert call_count == len(simple_ops)

    @pytest.mark.asyncio
    async def test_multiple_concurrent_runs(
        self, driver, machine, doc, simple_ops
    ):
        """Test that multiple concurrent runs work correctly."""
        callback_mock1 = MagicMock()
        callback_mock2 = MagicMock()

        # Run two operations concurrently
        task1 = driver.run(simple_ops, machine, doc, callback_mock1)
        task2 = driver.run(simple_ops, machine, doc, callback_mock2)

        await asyncio.gather(task1, task2)

        # Both callbacks should have been called the correct number of times
        assert callback_mock1.call_count == len(simple_ops)
        assert callback_mock2.call_count == len(simple_ops)

    @pytest.mark.asyncio
    async def test_callback_with_document_context(self, driver, machine, doc):
        """Test callback with a document context (no workpieces needed)."""
        # Create simple ops for testing with document context
        ops = Ops()
        ops.add(MoveToCommand((5.0, 5.0, 0.0)))
        ops.add(LineToCommand((15.0, 15.0, 0.0)))

        callback_mock = MagicMock()

        await driver.run(ops, machine, doc, callback_mock)

        # Verify callback was called
        assert callback_mock.call_count == len(ops)

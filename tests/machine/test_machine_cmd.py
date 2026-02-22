import pytest
import pytest_asyncio
import asyncio
from unittest.mock import MagicMock, PropertyMock
from functools import partial
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand
from rayforge.machine.cmd import MachineCmd
from rayforge.machine.models.machine import Machine
from rayforge.machine.driver.driver import Axis
from rayforge.pipeline.artifact import JobArtifact
from rayforge.shared.tasker.manager import TaskManager


@pytest_asyncio.fixture(autouse=True)
async def task_mgr(monkeypatch):
    """
    Provides a test-isolated TaskManager, configured to bridge its main-thread
    callbacks to the asyncio event loop. This instance replaces the global
    task_mgr for the duration of the tests in this module.
    """
    main_loop = asyncio.get_running_loop()

    def asyncio_scheduler(callback, *args, **kwargs):
        # Use call_soon_threadsafe because the TaskManager runs on a separate
        # thread and schedules callbacks onto this main loop.
        main_loop.call_soon_threadsafe(partial(callback, *args, **kwargs))

    # Instantiate the TaskManager with our custom scheduler
    tm = TaskManager(main_thread_scheduler=asyncio_scheduler)

    # Patch the global singleton where it is imported and used by other modules
    monkeypatch.setattr("rayforge.machine.models.machine.task_mgr", tm)

    yield tm

    # Properly shut down the manager and its thread after tests are done
    tm.shutdown()


@pytest.fixture(autouse=True)
def test_config_manager(tmp_path, monkeypatch):
    """Provides a test-isolated ConfigManager."""
    from rayforge.core.config import ConfigManager

    # We need to patch get_context to return a mock config manager for the
    # editor
    mock_config_mgr = MagicMock(spec=ConfigManager)
    yield mock_config_mgr


@pytest.fixture
def machine(lite_context):
    """Provides a default Machine instance with NoDeviceDriver."""
    m = Machine(lite_context)
    lite_context.machine_mgr.add_machine(m)
    return m


@pytest.fixture
def machine_cmd(doc_editor):
    """Provides a MachineCmd instance."""
    return MachineCmd(doc_editor)


@pytest.fixture
def simple_ops():
    """Creates a simple Ops object with a few commands."""
    ops = Ops()
    ops.add(MoveToCommand((10, 10, 0)))
    ops.add(LineToCommand((20, 10, 0)))
    ops.add(LineToCommand((20, 20, 0)))
    return ops


@pytest.fixture
def job_artifact(simple_ops):
    """Creates a JobArtifact containing simple_ops."""
    return JobArtifact(
        ops=simple_ops, distance=simple_ops.distance(), generation_id=1
    )


async def wait_for_tasks_to_finish(task_mgr: TaskManager):
    """
    Asynchronously waits for the task manager to become idle.
    """
    # Yield to the loop to ensure pending callbacks (like adding tasks) run
    # first
    await asyncio.sleep(0)

    # Use the now-correct, thread-safe wait_until_settled in a non-blocking way
    if await asyncio.to_thread(task_mgr.wait_until_settled, 2000):
        return
    pytest.fail("Task manager did not become idle in time.")


class TestMachineCmdJobMonitoring:
    """Test suite for the job monitoring orchestration in MachineCmd."""

    @pytest.mark.asyncio
    async def test_send_job_granular_progress(
        self, machine_cmd, machine, simple_ops, job_artifact, mocker
    ):
        """
        Tests the full monitoring flow for a driver that reports
        granular progress.
        """
        assert machine.driver.reports_granular_progress is True

        # --- Arrange ---
        job_started_spy = MagicMock()
        progress_updated_spy = MagicMock()
        job_finished_spy = MagicMock()

        machine_cmd.job_started.connect(job_started_spy)
        machine.job_finished.connect(job_finished_spy)

        # --- Act ---
        # Setup progress spy AFTER the job starts and monitor is created
        def on_job_started(sender):
            monitor = machine_cmd._current_monitor
            assert monitor is not None
            monitor.progress_updated.connect(progress_updated_spy)

        job_started_spy.side_effect = on_job_started

        await machine_cmd._run_send_action(
            job_artifact, machine, on_progress=lambda metrics: None
        )

        # Yield control to the event loop to allow signal handlers
        # (like cleanup_monitor) that were scheduled with `call_soon`
        # to run before we proceed with assertions.
        await asyncio.sleep(0)

        # --- Assert ---
        # 1. Verify job lifecycle signals
        job_started_spy.assert_called_once()
        job_finished_spy.assert_called_once()
        assert machine_cmd._current_monitor is None  # Check cleanup

        # 2. Verify granular progress updates
        # The JobMonitor sends updates for all commands, including those with
        # zero distance (like MoveToCommand).
        # We must calculate the expected number of calls by counting all cmds.
        expected_call_count = len(simple_ops)
        assert progress_updated_spy.call_count == expected_call_count

    @pytest.mark.asyncio
    async def test_send_job_non_granular_progress(
        self, machine_cmd, machine, simple_ops, job_artifact, mocker
    ):
        """
        Tests the monitoring flow for a driver that does not report
        granular progress.
        """
        # --- Arrange ---
        mocker.patch.object(
            type(machine),
            "reports_granular_progress",
            new_callable=PropertyMock,
            return_value=False,
        )
        assert not machine.reports_granular_progress

        async def mock_run_and_finish(*args, **kwargs):
            # Simulate work and signal finish
            await asyncio.sleep(0)
            machine.driver.job_finished.send(machine.driver)

        run_mock = mocker.patch.object(
            machine.driver, "run", side_effect=mock_run_and_finish
        )

        # Use an asyncio.Event for robust synchronization
        job_finished_event = asyncio.Event()
        job_finished_spy = MagicMock(
            side_effect=lambda *a, **kw: job_finished_event.set()
        )

        machine.job_finished.connect(job_finished_spy)

        # --- Act ---
        await machine_cmd._run_send_action(
            job_artifact, machine, on_progress=lambda metrics: None
        )

        # Explicitly wait for the job_finished signal
        # handler to run. This eliminates the race condition.
        await asyncio.wait_for(job_finished_event.wait(), timeout=1)

        # --- Assert ---
        # 1. Verify driver was called correctly
        run_mock.assert_called_once()
        assert run_mock.call_args.kwargs["on_command_done"] is None

        # 2. Verify job lifecycle signals fired
        job_finished_spy.assert_called_once()

        # 3. Verify cleanup happened
        assert machine_cmd._current_monitor is None


class TestMachineCmdJog:
    """Test suite for the jogging functionality in MachineCmd."""

    @pytest.mark.asyncio
    async def test_jog_with_deltas(
        self, machine_cmd, machine, mocker, task_mgr
    ):
        """
        Test jogging using the dictionary of deltas.
        """
        # --- Arrange ---
        # Use a full AsyncMock replacement for machine.jog to avoid
        # dependency on Machine logic or driver state, and ensure correct
        # awaitable return for TaskManager.
        jog_mock = mocker.patch.object(
            machine, "jog", new_callable=mocker.AsyncMock
        )

        # --- Act ---
        # Jog X axis by 10mm at 1000mm/min
        deltas = {Axis.X: 10.0}
        machine_cmd.jog(machine, deltas, 1000)

        await wait_for_tasks_to_finish(task_mgr)

        # --- Assert ---
        # MachineCmd.jog should delegate to Machine.jog
        jog_mock.assert_called_once_with(deltas, 1000)

    @pytest.mark.asyncio
    async def test_jog_multi_axis(
        self, machine_cmd, machine, mocker, task_mgr
    ):
        """
        Test jogging multiple axes.
        """
        # --- Arrange ---
        jog_mock = mocker.patch.object(
            machine, "jog", new_callable=mocker.AsyncMock
        )

        # --- Act ---
        deltas = {Axis.X: 5.0, Axis.Y: -5.0}
        machine_cmd.jog(machine, deltas, 1500)

        await wait_for_tasks_to_finish(task_mgr)

        # --- Assert ---
        # Verify call arguments to Machine.jog
        jog_mock.assert_called_once_with(deltas, 1500)

import asyncio
from functools import partial
from unittest.mock import MagicMock, PropertyMock

import pytest
import pytest_asyncio
from raygeo.ops import Ops
from raygeo.ops.axis import Axis

from rayforge.core.config import ConfigManager
from rayforge.machine.cmd import JobAlreadyRunningError, MachineCmd
from rayforge.machine.models.machine import Machine
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
    ops.move_to(10, 10, 0)
    ops.line_to(20, 10, 0)
    ops.line_to(20, 20, 0)
    return ops


@pytest.fixture
def job_artifact(simple_ops, machine):
    """Creates a JobArtifact containing simple_ops with encoded G-code."""
    encoded = machine.driver.get_encoder().encode(simple_ops, machine, None)
    return JobArtifact(
        ops=simple_ops,
        distance=simple_ops.distance(),
        generation_id=1,
        encoded_output=encoded,
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
        self,
        machine_cmd,
        doc_editor,
        machine,
        simple_ops,
        job_artifact,
        mocker,
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

        # The job's machine hours update re-emits machine.changed via
        # the scheduler, which arms a debounced rebuild. Settle the
        # editor so no rebuild task lingers into teardown.
        await doc_editor.wait_until_settled()

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
        self,
        machine_cmd,
        doc_editor,
        machine,
        simple_ops,
        job_artifact,
        mocker,
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

        # The job's machine hours update re-emits machine.changed via
        # the scheduler, which arms a debounced rebuild. Settle the
        # editor so no rebuild task lingers into teardown.
        await doc_editor.wait_until_settled()

        # --- Assert ---
        # 1. Verify driver was called correctly
        run_mock.assert_called_once()
        assert run_mock.call_args.kwargs["on_command_done"] is None

        # 2. Verify job lifecycle signals fired
        job_finished_spy.assert_called_once()

        # 3. Verify cleanup happened
        assert machine_cmd._current_monitor is None

    @pytest.mark.asyncio
    async def test_second_job_rejected_with_actionable_error(
        self, machine_cmd, machine, job_artifact
    ):
        """
        Starting a second job while one runs must raise a dedicated
        error whose message tells the user what to do (Stop), and must
        not clobber the running job's monitor.
        """
        running_monitor = MagicMock()
        machine_cmd._current_monitor = running_monitor

        with pytest.raises(JobAlreadyRunningError) as excinfo:
            await machine_cmd._run_send_action(
                job_artifact, machine, on_progress=lambda metrics: None
            )

        assert "Stop" in str(excinfo.value)
        assert machine_cmd._current_monitor is running_monitor

    @pytest.mark.asyncio
    async def test_send_job_notifies_user_of_running_job(
        self, machine_cmd, machine, mocker
    ):
        """
        The rejection must surface as a user notification carrying the
        actionable message. Unlike real failures, the refusal is fully
        handled: it is not re-raised and is not prefixed with
        'Sending failed'.
        """
        notification_spy = MagicMock()
        machine_cmd._editor.notification_requested.connect(notification_spy)
        mocker.patch.object(
            machine_cmd,
            "_start_job",
            new_callable=mocker.AsyncMock,
            side_effect=JobAlreadyRunningError("A job is already running."),
        )

        await machine_cmd.send_job(machine)

        notification_spy.assert_called_once()
        message = notification_spy.call_args.kwargs["message"]
        assert "already running" in message
        assert "Sending failed" not in message


class TestMachineCmdPointerDryRun:
    """Tests for pointer dry-run generation in _start_job."""

    @pytest.fixture
    def pipeline_mocks(self, machine_cmd, mocker):
        pipeline = machine_cmd._editor.pipeline
        invalidate_spy = MagicMock()
        mocker.patch.object(
            pipeline, "invalidate_job_output", side_effect=invalidate_spy
        )
        return pipeline, invalidate_spy

    @pytest.mark.asyncio
    async def test_dry_run_shifts_only_during_generation(
        self,
        machine_cmd,
        machine,
        job_artifact,
        pipeline_mocks,
        mocker,
    ):
        """The shift flag is set for job generation and cleared right
        after; the shifted output is invalidated before and after so a
        regular send regenerates unshifted G-code."""
        self._enable_pointer_offset(machine)
        pipeline, invalidate_spy = pipeline_mocks
        handle = pipeline.artifact_store.put(job_artifact, creator_tag="test")
        flags_during_generation = []

        async def fake_generate():
            flags_during_generation.append(machine.pointer_job_shift_enabled)
            return handle

        mocker.patch.object(
            pipeline,
            "generate_job_artifact_async",
            side_effect=fake_generate,
        )
        actions = []

        async def final_job_action(
            artifact, machine, on_progress, dry_run=False
        ):
            actions.append(artifact)

        await machine_cmd._start_job(
            machine,
            final_job_action=final_job_action,
            pointer_dry_run=True,
        )

        assert flags_during_generation == [True]
        assert machine.pointer_job_shift_enabled is False
        assert invalidate_spy.call_count == 2
        assert actions == [job_artifact]

    @pytest.mark.asyncio
    async def test_dry_run_ignored_without_pointer_offset(
        self, machine_cmd, machine, job_artifact, pipeline_mocks, mocker
    ):
        """Without an enabled pointer offset, a dry-run send behaves
        like a regular send."""
        pipeline, invalidate_spy = pipeline_mocks
        handle = pipeline.artifact_store.put(job_artifact, creator_tag="test")

        async def fake_generate():
            return handle

        mocker.patch.object(
            pipeline,
            "generate_job_artifact_async",
            side_effect=fake_generate,
        )
        actions = []

        async def final_job_action(
            artifact, machine, on_progress, dry_run=False
        ):
            actions.append(artifact)

        await machine_cmd._start_job(
            machine,
            final_job_action=final_job_action,
            pointer_dry_run=True,
        )

        assert machine.pointer_job_shift_enabled is False
        assert invalidate_spy.call_count == 0
        assert actions == [job_artifact]

    @pytest.mark.asyncio
    async def test_regular_send_never_shifts(
        self, machine_cmd, machine, job_artifact, pipeline_mocks, mocker
    ):
        self._enable_pointer_offset(machine)
        pipeline, invalidate_spy = pipeline_mocks
        handle = pipeline.artifact_store.put(job_artifact, creator_tag="test")

        async def fake_generate():
            return handle

        mocker.patch.object(
            pipeline,
            "generate_job_artifact_async",
            side_effect=fake_generate,
        )

        async def final_job_action(
            artifact, machine, on_progress, dry_run=False
        ):
            pass

        await machine_cmd._start_job(
            machine, final_job_action=final_job_action
        )

        assert machine.pointer_job_shift_enabled is False
        assert invalidate_spy.call_count == 0

    @staticmethod
    def _enable_pointer_offset(machine):
        head = machine.get_default_laser_head()
        assert head is not None
        head.set_pointer_offset(10.0, 20.0)
        head.set_pointer_offset_enabled(True)
        return machine


class TestMachineCmdDryRunPower:
    """The pointer dry-run caps all laser power at framing power."""

    @pytest.fixture
    def powered_artifact(self, machine):
        ops = Ops()
        ops.set_power(0.8)
        ops.move_to(0, 0, 0)
        ops.line_to(10, 0, 0)
        ops.scan_to(10, 5, 0, power_values=[255, 128, 0])
        encoded = machine.driver.get_encoder().encode(ops, machine, None)
        return JobArtifact(
            ops=ops,
            distance=ops.distance(),
            generation_id=1,
            encoded_output=encoded,
        )

    @pytest.fixture
    def executed(self, machine_cmd, mocker):
        record = {}

        async def fake_execute(ops, machine, on_progress=None, encoded=None):
            record["ops"] = ops
            record["encoded"] = encoded

        mocker.patch.object(
            machine_cmd, "_execute_monitored_job", side_effect=fake_execute
        )
        return record

    @pytest.mark.asyncio
    async def test_dry_run_caps_power_at_framing_power(
        self, machine_cmd, machine, powered_artifact, executed
    ):
        head = machine.get_default_laser_head()
        assert head is not None
        head.set_frame_power(0.1)

        await machine_cmd._run_send_action(
            powered_artifact, machine, None, dry_run=True
        )

        run_ops = executed["ops"]
        assert run_ops is powered_artifact.ops
        assert run_ops.power(0) == pytest.approx(0.1)
        assert list(run_ops.scanline_data(3)) == [26, 26, 0]
        assert executed["encoded"] is not powered_artifact.encoded_output

    @pytest.mark.asyncio
    async def test_dry_run_with_zero_framing_power_disables_beam(
        self, machine_cmd, machine, powered_artifact, executed
    ):
        head = machine.get_default_laser_head()
        assert head is not None
        head.set_frame_power(0.0)

        await machine_cmd._run_send_action(
            powered_artifact, machine, None, dry_run=True
        )

        run_ops = executed["ops"]
        assert run_ops.power(0) == 0.0
        assert list(run_ops.scanline_data(3)) == [0, 0, 0]

    @pytest.mark.asyncio
    async def test_regular_send_keeps_job_power(
        self, machine_cmd, machine, powered_artifact, executed
    ):
        await machine_cmd._run_send_action(powered_artifact, machine, None)

        assert executed["ops"] is powered_artifact.ops
        assert executed["encoded"] is powered_artifact.encoded_output


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


class TestMachineCmdMoveTo:
    """Test suite for the absolute move_to command in MachineCmd."""

    @pytest.mark.asyncio
    async def test_move_to_xy(self, machine_cmd, machine, mocker, task_mgr):
        move_mock = mocker.patch.object(
            machine.driver, "move_to", new_callable=mocker.AsyncMock
        )

        machine_cmd.move_to(machine, 10.0, 20.0)

        await wait_for_tasks_to_finish(task_mgr)
        move_mock.assert_called_once_with(10.0, 20.0, None, None)

    @pytest.mark.asyncio
    async def test_move_to_xyz(self, machine_cmd, machine, mocker, task_mgr):
        move_mock = mocker.patch.object(
            machine.driver, "move_to", new_callable=mocker.AsyncMock
        )

        machine_cmd.move_to(machine, 10.0, 20.0, 5.0)

        await wait_for_tasks_to_finish(task_mgr)
        move_mock.assert_called_once_with(10.0, 20.0, 5.0, None)

    @pytest.mark.asyncio
    async def test_move_to_with_speed(
        self, machine_cmd, machine, mocker, task_mgr
    ):
        move_mock = mocker.patch.object(
            machine.driver, "move_to", new_callable=mocker.AsyncMock
        )

        machine_cmd.move_to(machine, 10.0, 20.0, speed=6000)

        await wait_for_tasks_to_finish(task_mgr)
        move_mock.assert_called_once_with(10.0, 20.0, None, 6000)


class TestMachineCmdLaserPower:
    """Test suite for manual laser power commands."""

    @pytest.mark.asyncio
    async def test_set_focus_power_uses_explicit_machine(
        self, machine_cmd, machine, mocker, task_mgr
    ):
        head = machine.get_default_head()
        set_focus_power_mock = mocker.patch.object(
            machine, "set_focus_power", new_callable=mocker.AsyncMock
        )

        machine_cmd.set_focus_power(head, 0.25, machine)

        await wait_for_tasks_to_finish(task_mgr)

        set_focus_power_mock.assert_called_once_with(head, 0.25)

    @pytest.mark.asyncio
    async def test_set_power_uses_explicit_machine(
        self, machine_cmd, machine, mocker, task_mgr
    ):
        head = machine.get_default_head()
        set_power_mock = mocker.patch.object(
            machine, "set_power", new_callable=mocker.AsyncMock
        )

        machine_cmd.set_power(head, 0.5, machine)

        await wait_for_tasks_to_finish(task_mgr)

        set_power_mock.assert_called_once_with(head, 0.5)

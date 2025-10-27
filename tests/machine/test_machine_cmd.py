import pytest
import pytest_asyncio
import asyncio
from unittest.mock import MagicMock, PropertyMock
from functools import partial
from rayforge.core.doc import Doc
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand
from rayforge.machine.cmd import MachineCmd
from rayforge.machine.models.machine import Machine
from rayforge.doceditor.editor import DocEditor
from rayforge.pipeline.artifact import (
    ArtifactStore,
    JobArtifact,
    JobArtifactHandle,
)
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
        # Use call_soon as we are in a single-threaded asyncio test
        # environment.
        main_loop.call_soon(partial(callback, *args, **kwargs))

    # Instantiate the TaskManager with our custom scheduler
    tm = TaskManager(main_thread_scheduler=asyncio_scheduler)

    # Patch the global singleton where it is imported and used by other modules
    monkeypatch.setattr("rayforge.machine.models.machine.task_mgr", tm)

    # Patch idle_add in modules that use it to schedule UI/main-thread
    # callbacks, redirecting them to the asyncio event loop for the test.
    monkeypatch.setattr("rayforge.machine.cmd.idle_add", asyncio_scheduler)
    monkeypatch.setattr(
        "rayforge.machine.models.machine.idle_add", asyncio_scheduler
    )

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
    monkeypatch.setattr(
        "rayforge.doceditor.editor.get_context",
        lambda: MagicMock(config_mgr=mock_config_mgr),
    )
    yield mock_config_mgr


@pytest.fixture
def machine(context_initializer):
    """Provides a default Machine instance with NoDeviceDriver."""
    return Machine(context_initializer)


@pytest.fixture
def doc():
    """Provides a fresh Doc instance for each test."""
    return Doc()


@pytest.fixture
def doc_editor(doc, task_mgr, test_config_manager):
    """Provides a DocEditor instance."""
    return DocEditor(task_mgr, test_config_manager, doc)


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
    return JobArtifact(ops=simple_ops, distance=simple_ops.distance())


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
        # Mock the artifact store to return our pre-made artifact
        mocker.patch.object(ArtifactStore, "get", return_value=job_artifact)

        job_started_spy = MagicMock()
        progress_updated_spy = MagicMock()

        # Use an asyncio.Event for robust synchronization
        job_finished_event = asyncio.Event()
        job_finished_spy = MagicMock(
            side_effect=lambda *a, **kw: job_finished_event.set()
        )

        machine_cmd.job_started.connect(job_started_spy)
        machine.job_finished.connect(job_finished_spy)

        # --- Act ---
        # The handle is just a placeholder; artifact_store.get is mocked
        dummy_handle = JobArtifactHandle(
            shm_name="dummy",
            handle_class_name="JobArtifactHandle",
            artifact_type_name="JobArtifact",
            time_estimate=0,
            distance=0,
        )

        # Setup progress spy AFTER the job starts and monitor is created
        def on_job_started(sender):
            monitor = machine_cmd._current_monitor
            assert monitor is not None
            monitor.progress_updated.connect(progress_updated_spy)

        job_started_spy.side_effect = on_job_started

        await machine_cmd._run_send_action(
            dummy_handle, machine, on_progress=lambda metrics: None
        )

        # Explicitly wait for the job_finished signal to be processed
        await asyncio.wait_for(job_finished_event.wait(), timeout=1)

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

        mocker.patch.object(ArtifactStore, "get", return_value=job_artifact)

        # Use an asyncio.Event for robust synchronization
        job_finished_event = asyncio.Event()
        job_finished_spy = MagicMock(
            side_effect=lambda *a, **kw: job_finished_event.set()
        )

        machine.job_finished.connect(job_finished_spy)

        # --- Act ---
        dummy_handle = JobArtifactHandle(
            shm_name="dummy",
            handle_class_name="JobArtifactHandle",
            artifact_type_name="JobArtifact",
            time_estimate=0,
            distance=0,
        )

        await machine_cmd._run_send_action(
            dummy_handle, machine, on_progress=lambda metrics: None
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

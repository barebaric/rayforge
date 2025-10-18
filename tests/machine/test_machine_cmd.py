import pytest
import pytest_asyncio
import asyncio
from unittest.mock import MagicMock, AsyncMock
from functools import partial
from rayforge.core.doc import Doc
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand
from rayforge.machine.cmd import MachineCmd, JobMonitor
from rayforge.machine.models.machine import Machine
from rayforge.machine.driver.grbl import GrblNetworkDriver
from rayforge.doceditor.editor import DocEditor
from rayforge.pipeline.artifact import (
    ArtifactStore,
    JobArtifact,
    JobArtifactHandle,
)
from rayforge.shared.tasker.manager import TaskManager
from rayforge.config import initialize_managers


@pytest_asyncio.fixture(autouse=True)
async def task_mgr(monkeypatch):
    """
    Provides a test-isolated TaskManager, configured to bridge its main-thread
    callbacks to the asyncio event loop. This instance replaces the global
    task_mgr for the duration of the tests in this module.
    """
    main_loop = asyncio.get_running_loop()

    def asyncio_scheduler(callback, *args, **kwargs):
        main_loop.call_soon_threadsafe(partial(callback, *args, **kwargs))

    # Instantiate the TaskManager with our custom scheduler
    tm = TaskManager(main_thread_scheduler=asyncio_scheduler)

    # Patch the global singleton where it is imported and used by other modules
    # that are not part of the test's dependency injection chain.
    monkeypatch.setattr("rayforge.machine.models.machine.task_mgr", tm)

    yield tm

    # Properly shut down the manager and its thread after tests are done
    tm.shutdown()


@pytest.fixture(autouse=True)
def test_config_manager(tmp_path):
    """Provides a test-isolated ConfigManager."""
    from rayforge import config

    temp_config_dir = tmp_path / "config"
    temp_machine_dir = temp_config_dir / "machines"
    config.CONFIG_DIR = temp_config_dir
    config.MACHINE_DIR = temp_machine_dir

    initialize_managers()
    yield config.config_mgr
    # Reset globals after test
    config.config = None
    config.config_mgr = None
    config.machine_mgr = None


@pytest.fixture
def machine():
    """Provides a default Machine instance with NoDeviceDriver."""
    return Machine()


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
        job_finished_spy = MagicMock()
        progress_updated_spy = MagicMock()
        active_monitor = None

        def on_job_started(sender, monitor: JobMonitor):
            nonlocal active_monitor
            active_monitor = monitor
            job_started_spy(sender, monitor=monitor)
            # Connect to the monitor's signal ONLY after it's started
            monitor.progress_updated.connect(progress_updated_spy)

        machine_cmd.job_started.connect(on_job_started)
        machine_cmd.job_finished.connect(job_finished_spy)

        # --- Act ---
        # The handle is just a placeholder; ArtifactStore.get is mocked
        dummy_handle = JobArtifactHandle(
            shm_name="dummy",
            handle_class_name="JobArtifactHandle",
            artifact_type_name="JobArtifact",
            time_estimate=0,
            distance=0,
        )
        await machine_cmd._run_send_action(dummy_handle, machine)

        # --- Assert ---
        # 1. Verify job lifecycle signals
        job_started_spy.assert_called_once()
        job_finished_spy.assert_called_once()
        assert (
            job_started_spy.call_args.kwargs["monitor"]
            is job_finished_spy.call_args.kwargs["monitor"]
        )
        assert active_monitor is not None

        # 2. Verify granular progress updates
        assert progress_updated_spy.call_count == len(simple_ops)

        # 3. Verify final state of the monitor
        final_metrics = active_monitor.metrics
        assert final_metrics["progress_fraction"] == 1.0
        assert (
            final_metrics["traveled_distance"]
            == final_metrics["total_distance"]
        )

    @pytest.mark.asyncio
    async def test_send_job_non_granular_progress(
        self, machine_cmd, machine, simple_ops, job_artifact, mocker
    ):
        """
        Tests the monitoring flow for a file-based driver that does not
        report granular progress.
        """
        # --- Arrange ---
        # Swap the driver for a non-granular one
        driver = GrblNetworkDriver()
        driver.setup(host="127.0.0.1")  # Needs valid setup
        run_mock = mocker.patch.object(driver, "run", new_callable=AsyncMock)
        machine.driver = driver
        assert machine.driver.reports_granular_progress is False

        mocker.patch.object(ArtifactStore, "get", return_value=job_artifact)

        job_started_spy = MagicMock()
        job_finished_spy = MagicMock()
        progress_updated_spy = MagicMock()
        active_monitor = None

        def on_job_started(sender, monitor: JobMonitor):
            nonlocal active_monitor
            active_monitor = monitor
            job_started_spy(sender, monitor=monitor)
            monitor.progress_updated.connect(progress_updated_spy)

        machine_cmd.job_started.connect(on_job_started)
        machine_cmd.job_finished.connect(job_finished_spy)

        # --- Act ---
        dummy_handle = JobArtifactHandle(
            shm_name="dummy",
            handle_class_name="JobArtifactHandle",
            artifact_type_name="JobArtifact",
            time_estimate=0,
            distance=0,
        )
        await machine_cmd._run_send_action(dummy_handle, machine)

        # --- Assert ---
        # 1. Verify driver was called correctly
        run_mock.assert_called_once()
        # Crucially, the callback must be None for non-granular drivers
        assert run_mock.call_args.kwargs["on_command_done"] is None

        # 2. Verify job lifecycle signals
        job_started_spy.assert_called_once()
        job_finished_spy.assert_called_once()
        assert active_monitor is not None

        # 3. Verify non-granular progress update (only one call)
        progress_updated_spy.assert_called_once()

        # 4. Verify final state of the monitor
        final_metrics = active_monitor.metrics
        assert final_metrics["progress_fraction"] == 1.0
        assert (
            final_metrics["traveled_distance"]
            == final_metrics["total_distance"]
        )

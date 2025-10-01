from typing import Tuple
import pytest
import pytest_asyncio
import asyncio
from pathlib import Path
from functools import partial

from rayforge.core.doc import Doc
from rayforge.core.import_source import ImportSource
from rayforge.core.ops import Ops
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.editor import DocEditor
from rayforge.image import SVG_RENDERER
from rayforge.machine.cmd import MachineCmd
from rayforge.machine.models.laser import Laser
from rayforge.machine.models.machine import Machine
from rayforge.machine.driver.dummy import NoDeviceDriver
from rayforge.pipeline.generator import OpsGenerator
from rayforge.shared.tasker import task_mgr as global_task_mgr
from rayforge.shared.tasker.manager import TaskManager
from rayforge.config import initialize_managers


@pytest_asyncio.fixture(autouse=True)
async def task_mgr_fixture(monkeypatch):
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

    # Patch the global singleton ONLY where it is defined.
    # All other modules that import it will get this patched instance.
    monkeypatch.setattr("rayforge.shared.tasker.task_mgr", tm)

    yield tm

    # Properly shut down the manager and its thread after tests are done
    tm.shutdown()


@pytest.fixture
def doc() -> Doc:
    """Provides a fresh Doc instance for each test."""
    return Doc()


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
def machine() -> Machine:
    """Provides a default Machine instance which uses NoDeviceDriver."""
    return Machine()


@pytest.fixture
def doc_editor(doc: Doc, test_config_manager) -> DocEditor:
    """
    Provides a DocEditor instance with real dependencies, configured
    to use the test's `doc` instance. Explicitly depends on
    test_config_manager to ensure correct initialization order.
    """
    # Use the yielded test_config_manager directly instead of the global name.
    editor = DocEditor(global_task_mgr, test_config_manager)
    editor.doc = doc
    editor.ops_generator = OpsGenerator(doc, global_task_mgr)
    return editor


def create_test_workpiece_and_source() -> Tuple[WorkPiece, ImportSource]:
    """Creates a simple WorkPiece and its linked ImportSource for testing."""
    svg_data = b'<svg><path d="M0,0 L10,10"/></svg>'
    source_file = Path("test.svg")
    source = ImportSource(
        source_file=source_file,
        original_data=svg_data,
        renderer=SVG_RENDERER,
    )
    workpiece = WorkPiece(name=source_file.name)
    workpiece.import_source_uid = source.uid
    return workpiece, source


async def wait_for_tasks_to_finish():
    """Polls the task manager until it is idle."""
    for _ in range(200):  # 2-second timeout
        if not global_task_mgr.has_tasks():
            return
        await asyncio.sleep(0.01)
    pytest.fail("Task manager did not become idle in time.")


class TestMachine:
    """Test suite for the Machine model and its command handlers."""

    def test_instantiation(self, machine: Machine):
        """Test that a new machine defaults to using NoDeviceDriver."""
        assert isinstance(machine.driver, NoDeviceDriver)
        assert machine.name is not None
        assert machine.id is not None

    @pytest.mark.asyncio
    async def test_set_driver(self, machine: Machine, mocker):
        """Test that changing the driver triggers a rebuild and cleanup."""
        assert isinstance(machine.driver, NoDeviceDriver)
        old_driver_cleanup_spy = mocker.spy(machine.driver, "cleanup")

        # set_driver schedules the rebuild asynchronously
        machine.set_driver(NoDeviceDriver, {})

        # Wait for the task to complete
        await wait_for_tasks_to_finish()

        # Verify the old driver was cleaned up and a new one is in place
        old_driver_cleanup_spy.assert_called_once()
        assert isinstance(machine.driver, NoDeviceDriver)

    @pytest.mark.asyncio
    async def test_send_job_calls_driver_run(
        self, doc: Doc, machine: Machine, doc_editor: DocEditor, mocker
    ):
        """
        Verify that sending a job correctly calls the driver's run method
        with the expected arguments, including the `doc`.
        """
        wp, source = create_test_workpiece_and_source()
        doc.add_import_source(source)
        doc.active_layer.add_child(wp)

        run_spy = mocker.spy(machine.driver, "run")
        machine_cmd = MachineCmd(doc_editor)

        machine_cmd.send_job(machine)
        await wait_for_tasks_to_finish()

        run_spy.assert_called_once()
        args, kwargs = run_spy.call_args
        assert len(args) == 3
        assert isinstance(args[0], Ops)
        assert args[1] is machine
        assert args[2] is doc
        assert not kwargs

    @pytest.mark.asyncio
    async def test_frame_job_calls_driver_run(
        self, doc: Doc, machine: Machine, doc_editor: DocEditor, mocker
    ):
        """Verify that framing a job calls the driver's run method."""
        wp, source = create_test_workpiece_and_source()
        doc.add_import_source(source)
        doc.active_layer.add_child(wp)

        laser = Laser()
        laser.frame_power = 1  # Must be an integer
        machine.heads = [laser]
        assert machine.can_frame() is True

        run_spy = mocker.spy(machine.driver, "run")
        machine_cmd = MachineCmd(doc_editor)

        machine_cmd.frame_job(machine)
        await wait_for_tasks_to_finish()

        run_spy.assert_called_once()
        args, kwargs = run_spy.call_args
        assert isinstance(args[0], Ops)
        assert args[1] is machine
        assert args[2] is doc

    @pytest.mark.asyncio
    async def test_simple_commands(self, machine: Machine, mocker):
        """
        Test simple fire-and-forget commands like home, cancel, etc.,
        ensuring they correctly delegate to the driver.
        """
        machine_cmd = MachineCmd(mocker.MagicMock(spec=DocEditor))

        # Spy on all relevant driver methods
        home_spy = mocker.spy(machine.driver, "home")
        cancel_spy = mocker.spy(machine.driver, "cancel")
        set_hold_spy = mocker.spy(machine.driver, "set_hold")
        clear_alarm_spy = mocker.spy(machine.driver, "clear_alarm")
        select_tool_spy = mocker.spy(machine.driver, "select_tool")

        # Home
        machine_cmd.home_machine(machine)
        await wait_for_tasks_to_finish()
        home_spy.assert_called_once()

        # Cancel
        machine_cmd.cancel_job(machine)
        await wait_for_tasks_to_finish()
        cancel_spy.assert_called_once()

        # Hold
        machine_cmd.set_hold(machine, True)
        await wait_for_tasks_to_finish()
        set_hold_spy.assert_called_once_with(True)

        # Resume
        machine_cmd.set_hold(machine, False)
        await wait_for_tasks_to_finish()
        assert set_hold_spy.call_count == 2
        set_hold_spy.assert_called_with(False)

        # Clear Alarm
        machine_cmd.clear_alarm(machine)
        await wait_for_tasks_to_finish()
        clear_alarm_spy.assert_called_once()

        # Select Tool
        # Add a second laser to make index 1 valid
        laser2 = Laser()
        laser2.tool_number = 5  # Give it a distinct tool number
        machine.add_head(laser2)
        assert len(machine.heads) == 2

        machine_cmd.select_tool(machine, 1)
        await wait_for_tasks_to_finish()
        select_tool_spy.assert_called_once_with(5)

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up_driver(self, machine: Machine, mocker):
        """Verify that shutting down the machine calls driver.cleanup()."""
        cleanup_spy = mocker.spy(machine.driver, "cleanup")
        await machine.shutdown()
        cleanup_spy.assert_called_once()

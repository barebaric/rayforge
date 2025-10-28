from typing import Tuple
import pytest
import asyncio
from pathlib import Path
from rayforge.core.doc import Doc
from rayforge.core.import_source import ImportSource
from rayforge.core.ops import Ops
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.editor import DocEditor
from rayforge.image import SVG_RENDERER
import rayforge.machine.driver as driver_module
from rayforge.machine.cmd import MachineCmd
from rayforge.machine.models.laser import Laser
from rayforge.machine.models.machine import Machine
from rayforge.machine.driver.dummy import NoDeviceDriver
from rayforge.machine.driver.driver import Axis
from rayforge.shared.tasker.manager import TaskManager
from rayforge.core.matrix import Matrix
from rayforge.pipeline import steps


# Define the test-specific driver in the test file where it is used.
class OtherDriver(NoDeviceDriver):
    """A second dummy driver class for testing purposes."""

    pass


# Register the driver at the module level to ensure it's available
# as soon as this test file is imported by pytest.
driver_module.register_driver(OtherDriver)


@pytest.fixture
def doc() -> Doc:
    """Provides a fresh Doc instance for each test."""
    return Doc()


@pytest.fixture
def machine(context_initializer) -> Machine:
    """Provides a default Machine instance which uses NoDeviceDriver."""
    return Machine(context_initializer)


@pytest.fixture
def doc_editor(
    doc: Doc, context_initializer, task_mgr: TaskManager
) -> DocEditor:
    """
    Provides a DocEditor instance with real dependencies, configured
    to use the test's `doc` and `task_mgr` instances.
    """
    from rayforge.context import get_context

    config_manager = get_context().config_mgr
    assert config_manager is not None, (
        "ConfigManager was not initialized in context"
    )
    return DocEditor(task_mgr, context_initializer, doc)


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
    workpiece.matrix = workpiece.matrix @ Matrix.scale(10, 10)
    workpiece.import_source_uid = source.uid
    return workpiece, source


async def wait_for_tasks_to_finish(task_mgr: TaskManager):
    """Polls the given task manager until it is idle."""
    for _ in range(200):  # 2-second timeout
        if not task_mgr.has_tasks():
            return
        await asyncio.sleep(0.01)
    pytest.fail("Task manager did not become idle in time.")


@pytest.mark.usefixtures("context_initializer")
class TestMachine:
    """Test suite for the Machine model and its command handlers."""

    def test_instantiation(self, machine: Machine):
        """Test that a new machine defaults to using NoDeviceDriver."""
        assert isinstance(machine.driver, NoDeviceDriver)
        assert machine.name is not None
        assert machine.id is not None
        assert machine.acceleration == 1000

    @pytest.mark.asyncio
    async def test_set_driver(
        self,
        machine: Machine,
        mocker,
        task_mgr: TaskManager,
        context_initializer,
    ):
        """Test that changing the driver triggers a rebuild and cleanup."""
        assert isinstance(machine.driver, NoDeviceDriver)
        assert machine.driver_name is None

        # Keep a reference to the original driver to ensure it's not the same
        old_driver = machine.driver
        old_driver_id = id(old_driver)

        # Spy on the INSTANCE method before it gets replaced.
        cleanup_spy = mocker.spy(old_driver, "cleanup")

        # set_driver schedules the rebuild asynchronously.
        # Use distinct args to ensure it's not a no-op compared to the fixture.
        machine.set_driver(OtherDriver, {"port": "/dev/null"})
        await wait_for_tasks_to_finish(task_mgr)

        # Verify the correct new driver is in place.
        assert isinstance(machine.driver, OtherDriver)
        assert id(machine.driver) != old_driver_id
        cleanup_spy.assert_called_once()
        assert machine.driver_name == "OtherDriver"
        assert machine.driver_args == {"port": "/dev/null"}

    @pytest.mark.asyncio
    async def test_send_job_calls_driver_run(
        self,
        doc: Doc,
        machine: Machine,
        doc_editor: DocEditor,
        mocker,
        context_initializer,
    ):
        """
        Verify that sending a job correctly calls the driver's run method
        with the expected arguments, including the `doc`.
        """
        # --- Arrange ---
        # Add a step to the workflow, which is required for job assembly.
        step = steps.create_contour_step(context_initializer)
        workflow = doc.active_layer.workflow
        assert workflow is not None
        workflow.add_step(step)

        # Add a workpiece to the document, which will trigger ops generation.
        workpiece, source = create_test_workpiece_and_source()
        doc.add_import_source(source)
        doc.active_layer.add_child(workpiece)

        # Wait for the background processing to finish.
        await doc_editor.wait_until_settled()

        run_spy = mocker.spy(machine.driver, "run")
        machine_cmd = MachineCmd(doc_editor)

        # --- Act ---
        # Run the full job assembly pipeline and send it to the driver.
        await machine_cmd.send_job(machine)

        # --- Assert ---
        run_spy.assert_called_once()
        ops, received_machine, received_doc = run_spy.call_args.args
        assert isinstance(ops, Ops)
        assert not ops.is_empty()
        assert received_machine is machine
        assert received_doc is doc

    @pytest.mark.asyncio
    async def test_frame_job_calls_driver_run(
        self,
        doc: Doc,
        machine: Machine,
        doc_editor: DocEditor,
        mocker,
        context_initializer,
    ):
        """Verify that framing a job calls the driver's run method."""
        # --- Arrange ---
        # Configure the machine to be capable of framing.
        head = machine.get_default_head()
        head.frame_power = 1
        assert machine.can_frame() is True

        # Add a step to the workflow, which is required for job assembly.
        step = steps.create_contour_step(context_initializer)
        workflow = doc.active_layer.workflow
        assert workflow is not None
        workflow.add_step(step)

        # Add a workpiece to the document.
        workpiece, source = create_test_workpiece_and_source()
        doc.add_import_source(source)
        doc.active_layer.add_child(workpiece)

        # Wait for background processing to complete.
        await doc_editor.wait_until_settled()

        run_spy = mocker.spy(machine.driver, "run")
        machine_cmd = MachineCmd(doc_editor)

        # --- Act ---
        await machine_cmd.frame_job(machine)

        # --- Assert ---
        run_spy.assert_called_once()
        ops, received_machine, received_doc = run_spy.call_args.args
        assert isinstance(ops, Ops)
        assert not ops.is_empty()
        assert received_machine is machine
        assert received_doc is doc

    @pytest.mark.asyncio
    async def test_simple_commands(
        self,
        machine: Machine,
        mocker,
        task_mgr: TaskManager,
        doc_editor: DocEditor,
    ):
        """
        Test simple fire-and-forget commands like home, cancel, etc.,
        ensuring they correctly delegate to the driver.
        """
        machine_cmd = MachineCmd(doc_editor)

        # Spy on the INSTANCE methods now that the fixture is stable
        home_spy = mocker.spy(machine.driver, "home")
        cancel_spy = mocker.spy(machine.driver, "cancel")
        set_hold_spy = mocker.spy(machine.driver, "set_hold")
        clear_alarm_spy = mocker.spy(machine.driver, "clear_alarm")
        select_tool_spy = mocker.spy(machine.driver, "select_tool")

        # Home
        machine_cmd.home_machine(machine)
        await wait_for_tasks_to_finish(task_mgr)
        home_spy.assert_called_once()

        # Cancel
        machine_cmd.cancel_job(machine)
        await wait_for_tasks_to_finish(task_mgr)
        cancel_spy.assert_called_once()

        # Hold
        machine_cmd.set_hold(machine, True)
        await wait_for_tasks_to_finish(task_mgr)
        set_hold_spy.assert_called_once_with(True)

        # Resume
        machine_cmd.set_hold(machine, False)
        await wait_for_tasks_to_finish(task_mgr)
        assert set_hold_spy.call_count == 2
        set_hold_spy.assert_called_with(False)

        # Clear Alarm
        machine_cmd.clear_alarm(machine)
        await wait_for_tasks_to_finish(task_mgr)
        clear_alarm_spy.assert_called_once()

        # Select Tool
        laser2 = Laser()
        laser2.tool_number = 5  # Give it a distinct tool number
        machine.add_head(laser2)
        assert len(machine.heads) == 2

        machine_cmd.select_tool(machine, 1)  # Select head at index 1
        await wait_for_tasks_to_finish(task_mgr)
        # Assert that the driver was called with the correct tool number (5)
        select_tool_spy.assert_called_once_with(5)

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up_driver(self, machine: Machine, mocker):
        """Verify that shutting down the machine calls driver.cleanup()."""
        cleanup_spy = mocker.spy(machine.driver, "cleanup")
        await machine.shutdown()
        cleanup_spy.assert_called_once()

    def test_acceleration_setter(self, machine: Machine, mocker):
        """Test that the acceleration setter works correctly."""
        # Test setting acceleration
        machine.set_acceleration(2000)
        assert machine.acceleration == 2000

        # Test that setting acceleration triggers changed signal
        changed_spy = mocker.spy(machine.changed, "send")
        machine.set_acceleration(1500)
        changed_spy.assert_called_once_with(machine)

    def test_machine_serialization_with_acceleration(self, machine: Machine):
        """Test that acceleration is properly serialized and deserialized."""
        # Set a specific acceleration value
        machine.set_acceleration(2500)

        # Serialize to dict
        machine_dict = machine.to_dict()

        # Check that acceleration is in the serialized data
        assert "acceleration" in machine_dict["machine"]["speeds"]
        assert machine_dict["machine"]["speeds"]["acceleration"] == 2500

        # Deserialize from dict
        new_machine = Machine.from_dict(machine_dict)

        # Check that acceleration is preserved
        assert new_machine.acceleration == 2500

    def test_get_default_head(self, machine: Machine):
        """Returns the first laser head, or raises an error if none exist."""
        default_head = machine.get_default_head()
        assert isinstance(default_head, Laser)
        assert len(machine.heads) == 1

        # Add another head
        laser2 = Laser()
        laser2.uid = "test-laser-2"
        machine.add_head(laser2)

        # Should still return the first head (the one from __init__)
        default_head = machine.get_default_head()
        assert default_head is machine.heads[0]
        assert len(machine.heads) == 2

        # Test with empty heads list - should raise ValueError
        machine.heads.clear()
        with pytest.raises(
            ValueError, match="Machine has no laser heads configured"
        ):
            machine.get_default_head()

    def test_home(self, machine: Machine):
        """Test that dummy driver has all jog features."""
        assert machine.can_home()
        assert machine.can_home(Axis.X)
        assert machine.can_home(Axis.Y)
        assert machine.can_home(Axis.Z)

    def test_jog(self, machine: Machine):
        """Test that dummy driver has all jog features."""
        assert machine.can_jog()
        assert machine.can_jog(Axis.X)
        assert machine.can_jog(Axis.Y)
        assert machine.can_jog(Axis.Z)

    def test_can_g0_with_speed(self, machine: Machine):
        assert machine.can_g0_with_speed()

    def test_new_driver_methods_smoothie(self, machine):
        """Test new driver methods for SmoothieDriver."""
        from rayforge.machine.driver.smoothie import SmoothieDriver

        machine.set_driver(SmoothieDriver, {"host": "test", "port": 23})

        # Test G0 with speed support
        assert machine.can_g0_with_speed()

        # Test homing support
        assert machine.can_home()
        assert machine.can_home(Axis.X)
        assert machine.can_home(Axis.Y)
        assert machine.can_home(Axis.Z)

        # Test jogging support
        assert machine.can_jog()
        assert machine.can_jog(Axis.X)
        assert machine.can_jog(Axis.Y)
        assert machine.can_jog(Axis.Z)

    def test_new_driver_methods_grbl_network(
        self, machine, context_initializer
    ):
        """Test new driver methods for GrblNetworkDriver."""
        from rayforge.machine.driver.grbl import GrblNetworkDriver

        # Create driver directly to avoid setup issues
        driver = GrblNetworkDriver(context_initializer)
        driver.setup(host="test")
        machine.driver = driver

        # Test G0 with speed support (GRBL doesn't support this)
        assert not machine.can_g0_with_speed()

        # Test homing support
        assert machine.can_home()
        assert machine.can_home(Axis.X)
        assert machine.can_home(Axis.Y)
        assert machine.can_home(Axis.Z)

        # Test jogging support
        assert machine.can_jog()
        assert machine.can_jog(Axis.X)
        assert machine.can_jog(Axis.Y)
        assert machine.can_jog(Axis.Z)

    def test_new_driver_methods_grbl_serial(
        self, machine, context_initializer
    ):
        """Test new driver methods for GrblSerialDriver."""
        from rayforge.machine.driver.grbl_serial import GrblSerialDriver

        # Create driver directly to avoid setup issues
        driver = GrblSerialDriver(context_initializer)
        driver.setup(port="/dev/test", baudrate=115200)
        machine.driver = driver

        # Test G0 with speed support (GRBL doesn't support this)
        assert not machine.can_g0_with_speed()

        # Test homing support
        assert machine.can_home()
        assert machine.can_home(Axis.X)
        assert machine.can_home(Axis.Y)
        assert machine.can_home(Axis.Z)

        # Test jogging support
        assert machine.can_jog()
        assert machine.can_jog(Axis.X)
        assert machine.can_jog(Axis.Y)
        assert machine.can_jog(Axis.Z)

    @pytest.mark.asyncio
    async def test_home_method_with_multiple_axes(
        self, machine, mocker, context_initializer
    ):
        """
        Test that home method accepts multiple axes using binary operators.
        """
        from rayforge.machine.driver.smoothie import SmoothieDriver

        # Mock the _send_and_wait method to avoid connection issues
        mock_send_and_wait = mocker.AsyncMock()

        # Create driver directly to avoid setup issues
        driver = SmoothieDriver(context_initializer)
        driver._send_and_wait = mock_send_and_wait
        machine.driver = driver

        # Test home with single axis
        await machine.home(Axis.X)
        await machine.home(Axis.Y)
        await machine.home(Axis.Z)

        # Test home with multiple axes using binary operators
        await machine.home(Axis.X | Axis.Y)
        await machine.home(Axis.X | Axis.Z)
        await machine.home(Axis.Y | Axis.Z)
        await machine.home(Axis.X | Axis.Y | Axis.Z)

        # Test home with no axes (should home all)
        await machine.home()
        await machine.home(None)

        # Verify that _send_and_wait was called for each home operation
        # 3 single axes + 3*2 for multiple axes (2 axes each)
        #  + 1*3 for all three axes + 2 for home all/home none
        assert mock_send_and_wait.call_count == 14

    @pytest.mark.asyncio
    async def test_machine_jog_methods(self, machine: Machine, mocker):
        """Test that machine jog methods delegate to driver."""
        # Mock the driver methods
        jog_mock = mocker.AsyncMock()
        home_mock = mocker.AsyncMock()
        machine.driver.jog = jog_mock
        machine.driver.home = home_mock

        # Test jog method with single axis
        await machine.jog(Axis.X, 1.0, 1000)
        jog_mock.assert_called_once_with(Axis.X, 1.0, 1000)

        # Reset the mock
        jog_mock.reset_mock()

        # Test jog method with multiple axes using bitmask
        await machine.jog(Axis.X | Axis.Y, 2.0, 1500)
        jog_mock.assert_called_once_with(Axis.X | Axis.Y, 2.0, 1500)

        # Reset the mock
        jog_mock.reset_mock()

        # Test jog method with all axes
        await machine.jog(Axis.X | Axis.Y | Axis.Z, 0.5, 2000)
        jog_mock.assert_called_once_with(Axis.X | Axis.Y | Axis.Z, 0.5, 2000)

        # Test home method
        await machine.home(Axis.Y)
        home_mock.assert_called_once_with(Axis.Y)

    def test_reports_granular_progress_no_device_driver(
        self, machine: Machine
    ):
        """Test reports_granular_progress returns True with NoDeviceDriver."""
        # NoDeviceDriver is the default driver and should return True
        assert machine.reports_granular_progress

    def test_reports_granular_progress_grbl_serial(
        self, machine: Machine, context_initializer
    ):
        """Test reports_granular_progress returns True for GrblSerialDriver."""
        from rayforge.machine.driver.grbl_serial import GrblSerialDriver

        # Create driver directly to avoid setup issues
        driver = GrblSerialDriver(context_initializer)
        driver.setup(port="/dev/test", baudrate=115200)
        machine.driver = driver

        # GrblSerialDriver should report granular progress
        assert machine.reports_granular_progress

    def test_reports_granular_progress_grbl_network(
        self, machine: Machine, context_initializer
    ):
        """
        Test reports_granular_progress returns False for GrblNetworkDriver.
        """
        from rayforge.machine.driver.grbl import GrblNetworkDriver

        # Create driver directly to avoid setup issues
        driver = GrblNetworkDriver(context_initializer)
        driver.setup(host="test")
        machine.driver = driver

        # GrblNetworkDriver should not report granular progress
        assert not machine.reports_granular_progress

    def test_reports_granular_progress_smoothie(
        self, machine: Machine, context_initializer
    ):
        """Test reports_granular_progress returns True for SmoothieDriver."""
        from rayforge.machine.driver.smoothie import SmoothieDriver

        # Create driver directly to avoid setup issues
        driver = SmoothieDriver(context_initializer)
        driver.setup(host="test", port=23)
        machine.driver = driver

        # SmoothieDriver should report granular progress
        assert machine.reports_granular_progress

from typing import Tuple
import pytest
import asyncio
from pathlib import Path
from rayforge.core.doc import Doc
from rayforge.core.geo import Geometry
from rayforge.core.matrix import Matrix
from rayforge.core.ops import Ops
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.editor import DocEditor
from rayforge.image import SVG_RENDERER
import rayforge.machine.driver as driver_module
from rayforge.machine.cmd import MachineCmd
from rayforge.machine.driver.dummy import NoDeviceDriver
from rayforge.machine.driver.driver import Axis
from rayforge.machine.models.laser import Laser
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.macro import MacroTrigger
from rayforge.pipeline import steps
from rayforge.pipeline.encoder.gcode import GcodeEncoder
from rayforge.shared.tasker.manager import TaskManager


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


def create_test_workpiece_and_source() -> Tuple[WorkPiece, SourceAsset]:
    """Creates a simple WorkPiece and its linked SourceAsset for testing."""
    svg_data = b'<svg><path d="M0,0 L10,10"/></svg>'
    source_file = Path("test.svg")
    source = SourceAsset(
        source_file=source_file,
        original_data=svg_data,
        renderer=SVG_RENDERER,
    )
    gen_config = SourceAssetSegment(
        source_asset_uid=source.uid,
        segment_mask_geometry=Geometry(),
        vectorization_spec=PassthroughSpec(),
    )
    workpiece = WorkPiece(name=source_file.name, source_segment=gen_config)
    workpiece.matrix = workpiece.matrix @ Matrix.scale(10, 10)
    return workpiece, source


async def wait_for_tasks_to_finish(task_mgr: TaskManager):
    """
    Asynchronously waits for the task manager to become idle.
    This is the correct way to wait inside an `async def` test.
    It will fail the test on timeout.
    """
    # Use the now-correct, thread-safe wait_until_settled in a non-blocking way
    if await asyncio.to_thread(task_mgr.wait_until_settled, 2000):
        return
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

    def test_encode_ops_delegates_to_driver(self, machine: Machine, mocker):
        """
        Verify that machine.encode_ops calls get_encoder on the active
        driver.
        """
        # --- Arrange ---
        # Create a mock encoder and spy on its encode method
        mock_encoder = GcodeEncoder(machine.dialect)
        encode_spy = mocker.spy(mock_encoder, "encode")

        # Patch the driver's get_encoder method to return our mock.
        # The return value of patch is the mock that replaced the original.
        get_encoder_mock = mocker.patch.object(
            machine.driver, "get_encoder", return_value=mock_encoder
        )

        ops_to_encode = Ops()
        doc_context = Doc()

        # --- Act ---
        machine_code, op_map = machine.encode_ops(ops_to_encode, doc_context)

        # --- Assert ---
        # 1. Verify that the patched get_encoder method was called
        get_encoder_mock.assert_called_once()

        # 2. Verify that the encoder's encode method was called with the
        # correct args
        encode_spy.assert_called_once()
        call_args = encode_spy.call_args.args
        assert call_args[0] is ops_to_encode
        assert call_args[1] is machine
        assert call_args[2] is doc_context

    @pytest.mark.asyncio
    async def test_send_job_calls_driver_run(
        self,
        doc: Doc,
        machine: Machine,
        doc_editor: DocEditor,
        mocker,
        context_initializer,
        task_mgr: TaskManager,
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
        doc.add_asset(source)
        doc.active_layer.add_child(workpiece)

        # Wait for the background processing to finish.
        await doc_editor.wait_until_settled()
        await wait_for_tasks_to_finish(task_mgr)

        run_spy = mocker.spy(machine.driver, "run")
        machine_cmd = MachineCmd(doc_editor)

        # --- Act ---
        # Run the full job assembly pipeline and send it to the driver.
        await machine_cmd.send_job(machine)
        await wait_for_tasks_to_finish(task_mgr)

        # --- Assert ---
        run_spy.assert_called_once()
        ops, received_doc = run_spy.call_args.args
        assert isinstance(ops, Ops)
        assert not ops.is_empty()
        assert received_doc is doc

    @pytest.mark.asyncio
    async def test_frame_job_calls_driver_run(
        self,
        doc: Doc,
        machine: Machine,
        doc_editor: DocEditor,
        mocker,
        context_initializer,
        task_mgr: TaskManager,
    ):
        """Verify that framing a job calls the driver's run method."""
        # --- Arrange ---
        # Configure the machine to be capable of framing.
        head = machine.get_default_head()
        head.set_frame_power(1)
        assert machine.can_frame() is True

        # Add a step to the workflow, which is required for job assembly.
        step = steps.create_contour_step(context_initializer)
        workflow = doc.active_layer.workflow
        assert workflow is not None
        workflow.add_step(step)

        # Add a workpiece to the document.
        workpiece, source = create_test_workpiece_and_source()
        doc.add_asset(source)
        doc.active_layer.add_child(workpiece)

        # Wait for background processing to complete.
        await doc_editor.wait_until_settled()
        await wait_for_tasks_to_finish(task_mgr)

        run_spy = mocker.spy(machine.driver, "run")
        machine_cmd = MachineCmd(doc_editor)

        # --- Act ---
        await machine_cmd.frame_job(machine)
        await wait_for_tasks_to_finish(task_mgr)

        # --- Assert ---
        run_spy.assert_called_once()
        ops, received_doc = run_spy.call_args.args
        assert isinstance(ops, Ops)
        assert not ops.is_empty()
        assert received_doc is doc

    def test_can_focus(self, machine: Machine):
        """Test that can_focus returns True when any head has focus power."""
        # Default state - focus power is 0
        assert machine.can_focus() is False

        # Set focus power on default head
        head = machine.get_default_head()
        head.set_focus_power(0.1)
        assert machine.can_focus() is True

        # Add another head with focus power
        laser2 = Laser()
        laser2.set_focus_power(0.05)
        machine.add_head(laser2)
        assert machine.can_focus() is True

        # Set focus power to 0 on all heads
        head.set_focus_power(0.0)
        laser2.set_focus_power(0.0)
        assert machine.can_focus() is False

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

    @pytest.mark.asyncio
    async def test_machine_serialization_with_acceleration(
        self, machine: Machine, task_mgr: TaskManager
    ):
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

        # Wait for the async driver rebuild scheduled by from_dict to finish
        await wait_for_tasks_to_finish(task_mgr)

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

    @pytest.mark.asyncio
    async def test_set_power_calls_driver(
        self,
        machine: Machine,
        mocker,
        task_mgr: TaskManager,
        context_initializer,
    ):
        """Test that set_power correctly calls driver with percentage."""
        # Mock the driver's set_power method
        mock_set_power = mocker.spy(machine.driver, "set_power")

        # Call set_power with 50%
        await machine.set_power(percent=50)

        # Verify the driver method was called with correct argument
        default_head = machine.get_default_head()
        mock_set_power.assert_called_once_with(default_head, 50)

    @pytest.mark.asyncio
    async def test_set_power_zero_percent(
        self,
        machine: Machine,
        mocker,
        task_mgr: TaskManager,
        context_initializer,
    ):
        """Test that set_power with 0% calls driver with disable command."""
        # Mock the driver's set_power method
        mock_set_power = mocker.spy(machine.driver, "set_power")

        # Call set_power with 0%
        await machine.set_power(percent=0)

        # Verify the driver method was called with 0
        default_head = machine.get_default_head()
        mock_set_power.assert_called_once_with(default_head, 0)

    @pytest.mark.asyncio
    async def test_set_power_full_power(
        self,
        machine: Machine,
        mocker,
        task_mgr: TaskManager,
        context_initializer,
    ):
        """Test that set_power with 100% calls driver with max power."""
        # Mock the driver's set_power method
        mock_set_power = mocker.spy(machine.driver, "set_power")

        # Call set_power with 100%
        await machine.set_power(percent=100)

        # Verify the driver method was called with 100
        default_head = machine.get_default_head()
        mock_set_power.assert_called_once_with(default_head, 100)

    @pytest.mark.asyncio
    async def test_set_power_with_specific_head(
        self,
        machine: Machine,
        mocker,
        task_mgr: TaskManager,
        context_initializer,
    ):
        """Test that set_power with specific head calls driver correctly."""
        # Mock driver's set_power method
        mock_set_power = mocker.spy(machine.driver, "set_power")

        # Get a specific head (second head)
        laser2 = Laser()
        laser2.uid = "test-laser-2"
        machine.add_head(laser2)

        # Call set_power with specific head
        await machine.set_power(head=laser2, percent=75)

        # Verify driver method was called with correct arguments
        mock_set_power.assert_called_once_with(laser2, 75)

    def test_dialect_property(self, machine: Machine):
        """Test that dialect property returns correct dialect instance."""
        from rayforge.machine.models.dialect import get_dialect

        # Get the dialect through the property
        dialect = machine.dialect

        # Verify it's the correct type
        from rayforge.machine.models.dialect import GcodeDialect

        assert isinstance(dialect, GcodeDialect)

        # Verify it matches what get_dialect would return
        expected_dialect = get_dialect(machine.dialect_uid)
        assert dialect == expected_dialect

    def test_dialect_property_changes_with_dialect_uid(self, machine: Machine):
        """Test that dialect property reflects changes to dialect_uid."""
        from rayforge.machine.models.dialect_builtins import (
            GRBL_DIALECT,
            SMOOTHIEWARE_DIALECT,
        )

        # Initial state
        assert machine.dialect_uid == "grbl"
        assert machine.dialect == GRBL_DIALECT

        # Change dialect uid
        machine.set_dialect_uid("smoothieware")

        # Verify dialect property returns the new dialect
        assert machine.dialect == SMOOTHIEWARE_DIALECT

        # Change back
        machine.set_dialect_uid("grbl")
        assert machine.dialect == GRBL_DIALECT

    def test_can_g0_with_speed(self, machine: Machine):
        assert machine.can_g0_with_speed()

    @pytest.mark.asyncio
    async def test_new_driver_methods_smoothie(
        self, machine: Machine, task_mgr: TaskManager
    ):
        """Test new driver methods for SmoothieDriver."""
        from rayforge.machine.driver.smoothie import SmoothieDriver

        try:
            machine.set_driver(SmoothieDriver, {"host": "test", "port": 23})
            # Wait for the async set_driver operation to complete
            await wait_for_tasks_to_finish(task_mgr)
        finally:
            # Ensure the driver is cleaned up to stop any pending tasks
            await machine.shutdown()

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

    @pytest.mark.asyncio
    async def test_new_driver_methods_grbl_network(
        self, machine: Machine, context_initializer
    ):
        """Test new driver methods for GrblNetworkDriver."""
        from rayforge.machine.driver.grbl import GrblNetworkDriver

        try:
            # Create driver directly to avoid setup issues
            driver = GrblNetworkDriver(context_initializer, machine)
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
        finally:
            await machine.shutdown()

    @pytest.mark.asyncio
    async def test_new_driver_methods_grbl_serial(
        self, machine: Machine, context_initializer
    ):
        """Test new driver methods for GrblSerialDriver."""
        from rayforge.machine.driver.grbl_serial import GrblSerialDriver

        try:
            # Create driver directly to avoid setup issues
            driver = GrblSerialDriver(context_initializer, machine)
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
        finally:
            await machine.shutdown()

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
        driver = SmoothieDriver(context_initializer, machine)
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

    @pytest.mark.asyncio
    async def test_run_raw_delegates_to_driver(self, machine: Machine, mocker):
        """Test that the machine's run_raw method delegates to the driver."""
        # Spy on the active driver's run_raw method
        run_raw_spy = mocker.spy(machine.driver, "run_raw")

        test_gcode = "G0 X10 Y10\nG1 F1000 X20"

        # Call the machine's run_raw method
        await machine.run_raw(test_gcode)

        # Verify that the driver's method was called with the correct argument
        run_raw_spy.assert_called_once_with(test_gcode)

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
        driver = GrblSerialDriver(context_initializer, machine)
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
        driver = GrblNetworkDriver(context_initializer, machine)
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
        driver = SmoothieDriver(context_initializer, machine)
        driver.setup(host="test", port=23)
        machine.driver = driver

        # SmoothieDriver should report granular progress
        assert machine.reports_granular_progress

    @pytest.mark.asyncio
    async def test_hook_migration_full(self, task_mgr: TaskManager):
        """
        Tests that legacy JOB_START and JOB_END hooks are migrated to a new
        custom dialect upon loading a machine.
        """
        from rayforge.machine.models.dialect import (
            get_dialect,
            _DIALECT_REGISTRY,
        )

        initial_dialect_count = len(_DIALECT_REGISTRY)
        start_code = ["G28 ; Home at start"]
        end_code = ["M2 ; Program End"]

        legacy_data = {
            "machine": {
                "name": "Legacy Machine",
                "dialect_uid": "grbl",
                "hookmacros": {
                    "JOB_START": {"code": start_code},
                    "JOB_END": {"code": end_code},
                    "LAYER_START": {"code": ["; Layer Start"]},
                },
            }
        }

        # Act
        new_machine = Machine.from_dict(legacy_data)
        await wait_for_tasks_to_finish(task_mgr)

        # Assert Migration
        assert len(_DIALECT_REGISTRY) == initial_dialect_count + 1
        assert new_machine.dialect_uid != "grbl"
        assert "JOB_START" not in new_machine.hookmacros
        assert "JOB_END" not in new_machine.hookmacros
        assert MacroTrigger.LAYER_START in new_machine.hookmacros

        # Assert New Dialect Content
        migrated_dialect = get_dialect(new_machine.dialect_uid)
        assert migrated_dialect.is_custom is True
        assert migrated_dialect.preamble == start_code
        assert migrated_dialect.postscript == end_code
        assert "Legacy Machine" in migrated_dialect.label

    @pytest.mark.asyncio
    async def test_hook_migration_partial(self, task_mgr: TaskManager):
        """
        Tests that migration works correctly if only one legacy hook is
        present.
        """
        from rayforge.machine.models.dialect import (
            get_dialect,
            _DIALECT_REGISTRY,
        )

        base_dialect = get_dialect("grbl")
        initial_dialect_count = len(_DIALECT_REGISTRY)
        start_code = ["G21 G90"]

        legacy_data = {
            "machine": {
                "name": "Partial Legacy",
                "dialect_uid": "grbl",
                "hookmacros": {"JOB_START": {"code": start_code}},
            }
        }

        # Act
        new_machine = Machine.from_dict(legacy_data)
        await wait_for_tasks_to_finish(task_mgr)

        # Assert Migration
        assert len(_DIALECT_REGISTRY) == initial_dialect_count + 1
        assert new_machine.dialect_uid != "grbl"
        assert not new_machine.hookmacros

        # Assert New Dialect Content
        migrated_dialect = get_dialect(new_machine.dialect_uid)
        assert migrated_dialect.is_custom is True
        assert migrated_dialect.preamble == start_code
        # Postscript should be inherited from the original dialect
        assert migrated_dialect.postscript == base_dialect.postscript

    @pytest.mark.asyncio
    async def test_hook_migration_not_needed(self, task_mgr: TaskManager):
        """
        Tests that no migration occurs for a modern machine configuration.
        """
        from rayforge.machine.models.dialect import _DIALECT_REGISTRY

        initial_dialect_count = len(_DIALECT_REGISTRY)

        modern_data = {
            "machine": {
                "name": "Modern Machine",
                "dialect_uid": "smoothieware",
                "hookmacros": {"LAYER_START": {"code": ["; Modern Hook"]}},
            }
        }

        # Act
        new_machine = Machine.from_dict(modern_data)
        await wait_for_tasks_to_finish(task_mgr)

        # Assert No Migration
        assert len(_DIALECT_REGISTRY) == initial_dialect_count
        assert new_machine.dialect_uid == "smoothieware"
        assert MacroTrigger.LAYER_START in new_machine.hookmacros

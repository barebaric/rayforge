import logging
from abc import ABC, abstractmethod
from typing import (
    List,
    Optional,
    Tuple,
    Any,
    TYPE_CHECKING,
    Callable,
    Union,
    Awaitable,
)
from blinker import Signal
from dataclasses import dataclass
from enum import Enum, auto, IntFlag
from ...core.ops import Ops
from ...debug import LogType
from ...context import RayforgeContext
from ..transport import TransportStatus

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...shared.varset import VarSet
    from ..models.machine import Machine
    from ..models.laser import Laser
    from ...pipeline.encoder.gcode import GcodeOpMap


logger = logging.getLogger(__name__)


class DriverPrecheckError(Exception):
    """Custom exception for non-fatal pre-flight check failures."""

    pass


class DriverSetupError(Exception):
    """Custom exception for driver setup failures."""

    pass


class DeviceConnectionError(Exception):
    """Custom exception for failures to communicate with a device."""

    pass


class DeviceStatus(Enum):
    UNKNOWN = auto()
    IDLE = auto()
    RUN = auto()
    HOLD = auto()
    JOG = auto()
    ALARM = auto()
    DOOR = auto()
    CHECK = auto()
    HOME = auto()
    SLEEP = auto()
    TOOL = auto()
    QUEUE = auto()
    LOCK = auto()
    UNLOCK = auto()
    CYCLE = auto()
    TEST = auto()


# Translatable labels for DeviceStatus enums
DEVICE_STATUS_LABELS = {
    DeviceStatus.UNKNOWN: _("Unknown"),
    DeviceStatus.IDLE: _("Idle"),
    DeviceStatus.RUN: _("Run"),
    DeviceStatus.HOLD: _("Hold"),
    DeviceStatus.JOG: _("Jog"),
    DeviceStatus.ALARM: _("Alarm"),
    DeviceStatus.DOOR: _("Door"),
    DeviceStatus.CHECK: _("Check"),
    DeviceStatus.HOME: _("Home"),
    DeviceStatus.SLEEP: _("Sleep"),
    DeviceStatus.TOOL: _("Tool"),
    DeviceStatus.QUEUE: _("Queue"),
    DeviceStatus.LOCK: _("Lock"),
    DeviceStatus.UNLOCK: _("Unlock"),
    DeviceStatus.CYCLE: _("Cycle"),
    DeviceStatus.TEST: _("Test"),
}


Pos = Tuple[Optional[float], Optional[float], Optional[float]]  # x, y, z in mm


class Axis(IntFlag):
    """Enum for machine axes"""

    X = 1
    Y = 2
    Z = 4


@dataclass
class DeviceState:
    status: DeviceStatus = DeviceStatus.UNKNOWN
    machine_pos: Pos = (None, None, None)
    work_pos: Pos = (None, None, None)
    feed_rate: Optional[int] = None


class Driver(ABC):
    """
    Abstract base class for all drivers.
    All drivers must provide the following methods:

       setup()
       cleanup()
       connect()
       run()
       move_to()

    All drivers provide the following signals:
       log_received: for log messages
       state_changed: emitted when the DeviceState changes
       command_status_changed: to monitor a command that was sent
       connection_status_changed: signals connectivity changes

    Subclasses of driver MUST NOT emit these signals directly;
    the should instead call self._log, self,_on_state_changed, etc.
    """

    label: str
    subtitle: str
    supports_settings: bool = False
    # Drivers that send files via the network may not be able to
    # report granular progress updates during the execution of a job.
    reports_granular_progress: bool = False

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        self._context = context
        self._machine = machine
        self.log_received = Signal()
        self.state_changed = Signal()
        self.command_status_changed = Signal()
        self.connection_status_changed = Signal()
        self.settings_read = Signal()
        self.job_finished = Signal()
        self.did_setup = False
        self.state: DeviceState = DeviceState()
        self.setup_error: Optional[str] = None

    @classmethod
    @abstractmethod
    def precheck(cls, **kwargs: Any) -> None:
        """
        A non-blocking, static check of the configuration that can be run
        before driver instantiation. It should raise DriverPrecheckError
        on failure. These failures are considered non-fatal warnings.
        """
        pass

    def setup(self, **kwargs: Any):
        """
        The method will be invoked with a dictionary of values gathered
        from the UI, based on the VarSet returned by get_setup_vars().
        """
        assert not self.did_setup
        self.did_setup = True
        self.setup_error = None

    async def cleanup(self):
        self.did_setup = False
        self.setup_error = None

    @classmethod
    @abstractmethod
    def get_setup_vars(cls) -> "VarSet":
        """
        Returns a VarSet defining the parameters needed for setup().
        This is used to dynamically generate the user interface.
        """
        pass

    @abstractmethod
    def get_setting_vars(self) -> List["VarSet"]:
        """
        Returns a VarSet defining the device's settings.
        The VarSet should define the settings but may have empty values.
        """
        pass

    @abstractmethod
    async def connect(self) -> None:
        """
        Establishes the connection and maintains it. i.e. auto reconnect.
        On errors or lost connection it should continue trying.
        """
        pass

    @abstractmethod
    async def run(
        self,
        ops: Ops,
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        """
        Converts the given Ops into commands for the machine, and executes
        them.

        Args:
            ops: The operations to execute
            doc: The document context
            on_command_done: Optional sync or async callback called when each
                           command is done. Called with the op_index.
        """
        pass

    @abstractmethod
    async def set_hold(self, hold: bool = True) -> None:
        """
        Sends a command to put the currently executing program on hold.
        If hold is False, sends the command to remove the hold.
        """
        pass

    @abstractmethod
    async def cancel(self) -> None:
        """
        Sends a command to cancel the currently executing program.
        """
        pass

    def can_home(self, axis: Optional["Axis"] = None) -> bool:
        """
        Check if this device supports homing for the given axis or axes.

        Args:
            axis: Optional axis to check. If None, checks if any homing
                  is supported.

        Returns:
            True if the device supports homing the specified axis/axes,
            False otherwise
        """
        return True

    @abstractmethod
    async def home(self, axes: Optional["Axis"] = None) -> None:
        """
        Sends a command to home machine.

        Args:
            axes: Optional axis or combination of axes to home. If None,
                homes all axes. Can be a single Axis or multiple axes
                using binary operators (e.g. Axis.X|Axis.Y)
        """
        pass

    @abstractmethod
    async def move_to(self, pos_x: float, pos_y: float) -> None:
        """
        Moves to the given position. Values are given mm.
        """
        pass

    @abstractmethod
    async def select_tool(self, tool_number: int) -> None:
        """
        Sends a command to select a new tool/laser head by its number.
        """
        pass

    @abstractmethod
    async def read_settings(self) -> None:
        """
        Reads the configuration settings from the device.
        Upon completion, it should emit the `settings_read` signal with the
        retrieved settings as a dictionary.
        """
        pass

    @abstractmethod
    async def write_setting(self, key: str, value: Any) -> None:
        """
        Writes a single configuration setting to the device.
        """
        pass

    @abstractmethod
    async def clear_alarm(self) -> None:
        """
        Sends a command to clear any active alarm state.
        """
        pass

    @abstractmethod
    async def set_power(self, head: "Laser", percent: int) -> None:
        """
        Sets the laser power to the specified percentage of max power.

        Args:
            head: The laser head to control.
            percent: Power percentage (0-100). 0 disables power.
        """
        pass

    def can_jog(self, axis: Optional["Axis"] = None) -> bool:
        """
        Check if this device supports jogging for the given axis or axes.

        Args:
            axis: Optional axis to check. If None, checks if any jogging
                  is supported.

        Returns:
            True if the device supports jogging the specified axis/axes,
            False otherwise
        """
        return False

    @abstractmethod
    async def jog(self, axis: Axis, distance: float, speed: int) -> None:
        """
        Jogs the machine along a specific axis or combination of axes.

        Args:
            axis: The axis or combination of axes to jog. Can be a single
                  Axis or multiple axes using binary operators
                  (e.g. Axis.X|Axis.Y)
            distance: The distance to jog in mm (positive or negative)
            speed: The jog speed in mm/min
        """
        pass

    def _log(self, message: str):
        self._context.debug_log_manager.add_entry(
            self.__class__.__name__, LogType.APP_INFO, message
        )
        self.log_received.send(self, message=message)

    def _on_state_changed(self):
        self._context.debug_log_manager.add_entry(
            self.__class__.__name__, LogType.STATE_CHANGE, self.state
        )
        self.state_changed.send(self, state=self.state)

    def _on_settings_read(self, settings: List["VarSet"]):
        num_settings = sum(len(vs) for vs in settings)
        logger.info(f"Driver settings read with {num_settings} settings.")

        all_values = {}
        for vs in settings:
            all_values.update(vs.get_values())

        self._context.debug_log_manager.add_entry(
            self.__class__.__name__,
            LogType.APP_INFO,
            f"Device settings read: {all_values}",
        )
        self.settings_read.send(self, settings=settings)

    def _on_command_status_changed(
        self, status: TransportStatus, message: Optional[str] = None
    ):
        log_data = f"Command status: {status.name}"
        if message:
            log_data += f" - {message}"
        self._context.debug_log_manager.add_entry(
            self.__class__.__name__, LogType.APP_INFO, log_data
        )
        self.command_status_changed.send(self, status=status, message=message)

    def _on_connection_status_changed(
        self, status: TransportStatus, message: Optional[str] = None
    ):
        log_data = f"Connection status: {status.name}"
        if message:
            log_data += f" - {message}"
        self._context.debug_log_manager.add_entry(
            self.__class__.__name__, LogType.APP_INFO, log_data
        )
        self.connection_status_changed.send(
            self, status=status, message=message
        )

    def can_g0_with_speed(self) -> bool:
        """
        Check if this device supports speed parameter in G0 commands.

        Returns:
            True if the device supports G0 with speed, False otherwise
        """
        return False

    def _track_command_execution(
        self,
        ops: Ops,
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> "GcodeOpMap":
        """
        Creates a GcodeOpMap for tracking command execution.

        This method should be called by driver implementations to get a
        GcodeOpMap that can be used to track which Ops commands correspond
        to which G-code lines. Drivers can then use this map to call the
        on_command_done callback at the appropriate times.

        Args:
            ops: The operations to execute
            doc: The document context
            on_command_done: Optional callback for command completion

        Returns:
            A GcodeOpMap for tracking command execution
        """
        from ...pipeline.encoder.gcode import GcodeEncoder

        encoder = GcodeEncoder.for_machine(self._machine)
        _, op_map = encoder.encode(ops, self._machine, doc)
        return op_map

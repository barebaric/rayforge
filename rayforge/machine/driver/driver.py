import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any, TYPE_CHECKING
from blinker import Signal
from dataclasses import dataclass
from enum import Enum, auto
from ...core.ops import Ops
from ...debug import debug_log_manager, LogType
from ..transport import TransportStatus

if TYPE_CHECKING:
    from ...shared.varset import VarSet
    from ..models.machine import Machine


logger = logging.getLogger(__name__)


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


Pos = Tuple[Optional[float], Optional[float], Optional[float]]  # x, y, z in mm


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

    label = None
    subtitle = None
    supports_settings = False

    def __init__(self):
        self.log_received = Signal()
        self.state_changed = Signal()
        self.command_status_changed = Signal()
        self.connection_status_changed = Signal()
        self.settings_read = Signal()
        self.did_setup = False
        self.state: DeviceState = DeviceState()
        self.setup_error: Optional[str] = None

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
    async def run(self, ops: Ops, machine: "Machine") -> None:
        """
        Converts the given Ops into commands for the machine, and executes
        them.
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

    @abstractmethod
    async def home(self) -> None:
        """
        Sends a command to home machine.
        """
        pass

    @abstractmethod
    async def move_to(self, pos_x: float, pos_y: float) -> None:
        """
        Moves to the given position. Values are given mm.
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

    def _log(self, message: str):
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.APP_INFO, message
        )
        self.log_received.send(self, message=message)

    def _on_state_changed(self):
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.STATE_CHANGE, self.state
        )
        self.state_changed.send(self, state=self.state)

    def _on_settings_read(self, settings: List["VarSet"]):
        num_settings = sum(len(vs) for vs in settings)
        logger.info(f"Driver settings read with {num_settings} settings.")

        all_values = {}
        for vs in settings:
            all_values.update(vs.get_values())

        debug_log_manager.add_entry(
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
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.APP_INFO, log_data
        )
        self.command_status_changed.send(self, status=status, message=message)

    def _on_connection_status_changed(
        self, status: TransportStatus, message: Optional[str] = None
    ):
        log_data = f"Connection status: {status.name}"
        if message:
            log_data += f" - {message}"
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.APP_INFO, log_data
        )
        self.connection_status_changed.send(
            self, status=status, message=message
        )

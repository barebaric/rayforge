from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
from blinker import Signal
from dataclasses import dataclass
from enum import Enum, auto
from ..transport import TransportStatus
from ..util.glib import idle_add
from ..models.ops import Ops
from ..models.machine import Machine


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

    def __init__(self):
        self.log_received = Signal()
        self.state_changed = Signal()
        self.command_status_changed = Signal()
        self.connection_status_changed = Signal()
        self.did_setup = False
        self.state: DeviceState = DeviceState()

    def setup(self, *args: Any, **kwargs: Any):
        """
        The type annotations of this method are used to generate a UI
        for the user! So if your driver requires any UI parameters,
        you should overload this function to ensure that a UI for the
        parameters is generated.

        The method will be invoked once the user has provided the arguments
        in the UI.
        """
        self.did_setup = True

    async def cleanup(self):
        self.did_setup = False

    @abstractmethod
    async def connect(self) -> None:
        """
        Establishes the connection and maintains it. i.e. auto reconnect.
        On errors or lost connection it should continue trying.
        """
        pass

    @abstractmethod
    async def run(self, ops: Ops, machine: Machine) -> None:
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

    def _log(self, message: str):
        idle_add(
            self.log_received.send,
            self,
            message=message
        )

    def _on_state_changed(self):
        idle_add(
            self.state_changed.send,
            self,
            state=self.state
        )

    def _on_command_status_changed(self,
                                   status: TransportStatus,
                                   message: Optional[str] = None):
        idle_add(
            self.command_status_changed.send,
            self,
            status=status,
            message=message
        )

    def _on_connection_status_changed(self,
                                      status: TransportStatus,
                                      message: Optional[str] = None):
        idle_add(
            self.connection_status_changed.send,
            self,
            status=status,
            message=message
        )


class DriverManager:
    def __init__(self):
        self.driver = None
        self.changed = Signal()

    async def _assign_driver(self, driver, **args):
        self.driver = driver
        self._on_driver_changed()
        self.driver.setup(**args)
        await self.driver.connect()

    async def _reconfigure_driver(self, **args):
        if not self.driver:
            return
        await self.driver.cleanup()
        self._on_driver_changed()
        self.driver.setup(**args)
        await self.driver.connect()

    async def _switch_driver(self, driver, **args):
        if self.driver:
            await self.driver.cleanup()
            del self.driver
        await self._assign_driver(driver, **args)

    def _on_driver_changed(self):
        idle_add(
            self.changed.send,
            self,
            driver=self.driver
        )

    async def select_by_cls(self, driver_cls, **args):
        if self.driver and self.driver.__class__ == driver_cls:
            await self._reconfigure_driver(**args)
        elif self.driver:
            await self._switch_driver(driver_cls(), **args)
        else:
            await self._assign_driver(driver_cls(), **args)


driver_mgr = DriverManager()

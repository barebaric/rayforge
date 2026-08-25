import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, auto
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Optional,
)

from blinker import Signal
from raygeo.ops.axis import Axis

from ...context import RayforgeContext
from ...core.varset import IntVar, VarSet
from ...shared.units.system import UnitSystem

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from ...core.doc import Doc
    from ...pipeline.encoder.base import EncodedOutput, OpsEncoder
    from ..device.profile import DeviceProfile
    from ..models.dialect import GcodeDialect
    from ..models.head import Head
    from ..models.laser import Laser
    from ..models.machine import Machine
    from .discovery import DeviceRecognizer


logger = logging.getLogger(__name__)


class DriverPrecheckError(Exception):
    """Custom exception for non-fatal pre-flight check failures."""


class DriverSetupError(Exception):
    """Custom exception for driver setup failures."""


class DeviceConnectionError(Exception):
    """Custom exception for failures to communicate with a device."""


class ResourceBusyError(DeviceConnectionError):
    """
    Raised when attempting to connect to a resource (e.g. serial port)
    that is already in use by another configured machine.
    """

    def __init__(self, resource: str, owner_name: str):
        self.resource = resource
        self.owner_name = owner_name
        super().__init__(
            _(
                "Resource '{resource}' is currently in use by '{owner}'."
            ).format(resource=resource, owner=owner_name)
        )


class DriverMaturity(Enum):
    STABLE = auto()
    UNTESTED = auto()
    EXPERIMENTAL = auto()
    KNOWN_BUGGY = auto()


DRIVER_MATURITY_LABELS = {
    DriverMaturity.STABLE: "",
    DriverMaturity.UNTESTED: _(
        "This driver has not been tested. It may or may not "
        "work. Use it at your own risk."
    ),
    DriverMaturity.EXPERIMENTAL: _(
        "This driver is experimental and may have "
        "unresolved issues. Use it with caution."
    ),
    DriverMaturity.KNOWN_BUGGY: _(
        "This driver is experimental and almost certainly buggy. It may not "
        "work reliably. Use it at your own risk."
    ),
}


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


@dataclass
class DeviceError:
    """Error with code, title and description."""

    code: int
    title: str
    description: str


Pos = tuple[float | None, ...]  # x, y, z[, a] in mm


@dataclass
class DeviceState:
    """Represents the complete state of a device at a moment in time."""

    status: DeviceStatus = DeviceStatus.UNKNOWN
    error: DeviceError | None = None
    machine_pos: Pos = (None, None, None)
    work_pos: Pos = (None, None, None)
    wco: Pos = (0.0, 0.0, 0.0)  # Work Coordinate Offset
    feed_rate: int | None = None
    spindle_speed: int | None = None
    buffer_available: int | None = None
    buffer_rx_available: int | None = None


@dataclass
class PWMParams:
    """PWM configuration reported by a driver for a laser head."""

    frequency: int
    max_frequency: int
    pulse_width: int
    min_pulse_width: int
    max_pulse_width: int


def pwm_varset(params: PWMParams) -> VarSet:
    """Build the PWM frequency / pulse-width settings VarSet."""
    return VarSet(
        vars=[
            IntVar(
                key="frequency",
                label=_("Frequency"),
                description=_("PWM frequency in Hz"),
                default=params.frequency,
                min_val=1,
                max_val=params.max_frequency,
            ),
            IntVar(
                key="pulse_width",
                label=_("Pulse Width"),
                description=_("Pulse width in microseconds"),
                default=params.pulse_width,
                min_val=params.min_pulse_width,
                max_val=params.max_pulse_width,
            ),
        ]
    )


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
       state_changed: emitted when the DeviceState changes
       command_status_changed: to monitor a command that was sent
       connection_status_changed: signals connectivity changes
       probe_status_changed: emits status during a probing cycle
       wcs_updated: emitted when work coordinate system data is updated
    """

    label: str
    subtitle: str
    supports_settings: bool = False
    # Drivers that send files via the network may not be able to
    # report granular progress updates during the execution of a job.
    reports_granular_progress: bool = False
    uses_gcode: bool = True
    maturity: DriverMaturity = DriverMaturity.STABLE
    supports_probing: bool = False
    # When True, the firmware applies its own overscan, so Rayforge's
    # OverscanTransformer would double it up and should be skipped.
    native_overscan: bool = False
    # When True, the driver can query the device to detect its
    # native unit system (metric vs imperial).
    supports_unit_detection: bool = False
    # Declarative description of what a device this driver can talk
    # to looks like on the wire. When set, the driver participates
    # in automatic device discovery: discovery scans each transport
    # once and evaluates the captured output against every
    # recognizer (see rayforge.machine.driver.discovery).
    DISCOVERY: ClassVar["DeviceRecognizer | None"] = None
    # mDNS service types announcing devices this driver can talk to
    # over the network. When non-empty, the driver participates in
    # network discovery (see rayforge.machine.driver.discovery).
    MDNS_SERVICES: ClassVar[tuple[str, ...]] = ()

    @property
    @abstractmethod
    def machine_space_wcs(self) -> str:
        """
        Returns the machine space coordinate system identifier.
        This is an immutable coordinate system with zero offset.
        """

    @property
    @abstractmethod
    def machine_space_wcs_display_name(self) -> str:
        """
        Returns a human-readable display name for the machine space
        coordinate system.
        """

    @property
    def supported_wcs(self) -> list[str]:
        """
        Returns the list of supported mutable Work Coordinate Systems.

        The first item should be the default WCS for this driver.
        Drivers may override this to provide driver-specific WCS names.
        """
        return ["G54", "G55", "G56", "G57", "G58", "G59"]

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        self._context = context
        self._machine = machine
        self.state_changed = Signal()
        self.command_status_changed = Signal()
        self.connection_status_changed = Signal()
        self.settings_read = Signal()
        self.job_finished = Signal()
        self.probe_status_changed = Signal()
        self.wcs_updated = Signal()
        self.config_changed = Signal()
        self.config: dict[str, Any] = {}
        self.did_setup = False
        self.state: DeviceState = DeviceState()

    @property
    def dialect(self) -> "GcodeDialect":
        assert self._machine.dialect is not None
        return self._machine.dialect

    def _log_extra(self, category: str) -> dict[str, str | None]:
        """Helper to create log extra dict with machine_id and category."""
        return {
            "log_category": category,
            "machine_id": self._machine.id if self._machine else None,
        }

    def _to_machine_length(self, mm: float) -> float:
        """
        Convert a length in millimeters to the machine's native units.

        Returns the value unchanged for metric machines, and inches
        rounded to four decimal places for imperial machines. Used when
        sending dimensional values to the device (e.g. jog distances,
        WCS offsets).
        """
        scale = self._machine.unit_system.scale_from_mm
        if scale == 1.0:
            return mm
        return round(mm * scale, 4)

    def _to_machine_speed(self, mm_per_min: float) -> float:
        """
        Convert a speed in mm/min to the machine's native units per minute.

        Returns the value unchanged for metric machines, and inches per
        minute for imperial machines.
        """
        scale = self._machine.unit_system.scale_from_mm
        if scale == 1.0:
            return mm_per_min
        return round(mm_per_min * scale, 4)

    def _from_machine_length(self, value: float) -> float:
        """
        Convert a length in the machine's native units back to millimeters.

        Used when interpreting positions reported by the device (e.g.
        status reports, probe results) which arrive in machine units.
        """
        return value / self._machine.unit_system.scale_from_mm

    @property
    def resource_uri(self) -> str | None:
        """
        Returns a unique identifier for the physical resource used by this
        driver (e.g. 'serial:///dev/ttyUSB0' or 'tcp://192.168.1.50:80').

        If multiple machines share this URI, the driver will prevent them
        from connecting simultaneously. Returns None if the driver does not
        lock a physical resource.
        """
        return None

    @classmethod
    @abstractmethod
    def precheck(cls, **kwargs: Any) -> None:
        """
        A non-blocking, static check of the configuration that can be run
        before driver instantiation. It should raise DriverPrecheckError
        on failure. These failures are considered non-fatal warnings.
        """

    @abstractmethod
    def _setup_implementation(self, **kwargs: Any) -> None:
        """
        Driver-specific setup implementation. Subclasses should override
        this method to perform their setup logic. If setup fails, this
        method should raise DriverSetupError.
        """

    def setup(self, **kwargs: Any):
        """
        The method will be invoked with a dictionary of values gathered
        from the UI, based on the VarSet returned by get_setup_vars().
        """
        assert not self.did_setup
        self.state.error = None
        try:
            self._setup_implementation(**kwargs)
        except DriverSetupError as e:
            logger.error(f"Setup failed: {e}")
            self.state.error = DeviceError(
                -999,
                str(e),
                _("Error during setup. You may need to edit device settings."),
            )
        self.did_setup = True

    async def cleanup(self):
        self.did_setup = False
        self.state.error = None

    @classmethod
    @abstractmethod
    def get_setup_vars(cls) -> "VarSet":
        """
        Returns a VarSet defining the parameters needed for setup().
        This is used to dynamically generate the user interface.
        """

    @classmethod
    @abstractmethod
    def create_encoder(cls, machine: "Machine") -> "OpsEncoder":
        """
        Factory method to return an OpsEncoder instance suitable for this
        driver class and the specific machine configuration.
        """

    @classmethod
    async def probe(
        cls, context: "RayforgeContext", **kwargs: Any
    ) -> tuple["DeviceProfile", list[str]]:
        """
        Probe a device at the given connection parameters and return
        an auto-populated ``(DeviceProfile, warnings)`` tuple.
        Only called if supports_probing is True.

        Raises on connection failure or timeout.
        """
        raise NotImplementedError

    def get_encoder(self) -> "OpsEncoder":
        """
        Convenience wrapper to get the encoder for this driver instance's
        machine. Delegates to the static factory method.
        """
        return self.create_encoder(self._machine)

    def supports_pwm(self, head: "Head") -> bool:
        """
        Returns whether the driver supports PWM for the given head.

        Subclasses may override this to report driver-specific support
        (e.g., PWM on Ruida CO2/fiber lasers but not diode lasers). The
        base implementation reports no PWM support.
        """
        return False

    def get_pwm_params(self, head: "Head") -> PWMParams | None:
        """
        Returns the PWM parameters reported by the driver for the given
        head, or None when the driver reports no PWM support.
        """
        return None

    async def detect_unit_system(self) -> UnitSystem | None:
        """
        Queries the device to detect its native unit system.

        Returns the detected ``UnitSystem``, or ``None`` when the
        driver cannot determine it (e.g. the device did not respond,
        or the firmware does not expose a unit-system setting).

        The base implementation always returns ``None``. Drivers that
        set ``supports_unit_detection = True`` should override this.

        This is called by the controller after a successful connection
        when ``machine.auto_detect_units`` is enabled.
        """
        return None

    @abstractmethod
    def get_setting_vars(self) -> list["VarSet"]:
        """
        Returns a VarSet defining the device's settings.
        The VarSet should define the settings but may have empty values.
        """

    async def connect(self) -> None:
        """
        Checks for resource conflicts with other machines, then establishes
        the connection via _connect_implementation().
        """
        my_uri = self.resource_uri
        if my_uri:
            # Check all other machines managed by the context
            # We access the internal dictionary to avoid overhead
            machines = self._context.machine_mgr.machines.values()
            for other_machine in machines:
                if other_machine is self._machine:
                    continue

                if (
                    other_machine.is_connected()
                    and other_machine.driver
                    and other_machine.driver.resource_uri == my_uri
                ):
                    raise ResourceBusyError(my_uri, other_machine.name)

        await self._connect_implementation()

    @abstractmethod
    async def _connect_implementation(self) -> None:
        """
        Establishes the connection and maintains it. i.e. auto reconnect.
        On errors or lost connection it should continue trying.
        """

    @abstractmethod
    async def run(
        self,
        encoded: "EncodedOutput",
        doc: "Doc",
        ops: "Ops",
        on_command_done: Callable[[int], None | Awaitable[None]] | None = None,
    ) -> None:
        """
        Executes the given encoded output.

        Args:
            encoded: The encoded output containing machine code and op map
            doc: The document context
            ops: The Ops object used to generate the encoded output.
            on_command_done: Optional sync or async callback called when each
                           command is done. Called with the op_index.
        """

    @abstractmethod
    async def run_raw(self, machine_code: str) -> None:
        """
        Executes a raw command (e.g. G-code if that is what the machine
        supports).

        Args:
            machine_code: The raw machine code to execute.
        """

    @abstractmethod
    async def set_hold(self, hold: bool = True) -> None:
        """
        Sends a command to put the currently executing program on hold.
        If hold is False, sends the command to remove the hold.
        """

    @abstractmethod
    async def cancel(self, emergency: bool = False) -> None:
        """
        Sends a command to cancel the currently executing program.

        When *emergency* is True, the cancellation is the result of an
        unexpected error rather than a user-initiated stop, and drivers
        should additionally send their dialect's failsafe command.
        """

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

    @abstractmethod
    async def move_to(self, pos_x: float, pos_y: float) -> None:
        """
        Moves to the given position. Values are given mm.
        """

    @abstractmethod
    async def select_tool(self, tool_number: int) -> None:
        """
        Sends a command to select a new tool/laser head by its number.
        """

    @abstractmethod
    async def read_settings(self) -> None:
        """
        Reads the configuration settings from the device.
        Upon completion, it should emit the `settings_read` signal with the
        retrieved settings as a dictionary.
        """

    @abstractmethod
    async def write_setting(self, key: str, value: Any) -> None:
        """
        Writes a single configuration setting to the device.
        """

    @abstractmethod
    async def clear_alarm(self) -> None:
        """
        Sends a command to clear any active alarm state.
        """

    @abstractmethod
    async def set_power(self, head: "Laser", percent: float) -> None:
        """
        Sets the laser power to the specified percentage of max power.

        Args:
            head: The laser head to control.
            percent: Power percentage (0-1.0). 0 disables power.
        """

    @abstractmethod
    async def set_focus_power(self, head: "Laser", percent: float) -> None:
        """
        Sets the laser power for focus mode.

        Some lasers use different commands for focusing vs cutting
        (e.g., M3 for constant power vs M4 for dynamic power).

        Args:
            head: The laser head to control.
            percent: Power percentage (0-1.0). 0 disables power.
        """

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
    async def jog(self, speed: int, **deltas: float) -> None:
        """
        Jogs the machine along specified axes.

        Args:
            speed: The jog speed in mm/min.
            **deltas: Keyword arguments where the key is the axis name
                      (e.g. 'x', 'y') and the value is the distance in mm.
        """

    @abstractmethod
    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None:
        """
        Sends a command to the controller to define the offset for a
        specific WCS slot (e.g. "G54").

        When *z* is ``None`` (a no-Z machine), the Z component is
        omitted from the emitted command.
        """

    @abstractmethod
    async def read_wcs_offsets(self) -> dict[str, Pos]:
        """
        Sends a command to query all current WCS offsets from the controller.

        Returns:
            A dictionary where keys are WCS slot names (e.g., "G54") and
            values are (x, y, z) offset tuples.
        """

    async def read_parser_state(self) -> str | None:
        """
        Sends a command to query the active G-code modal states, specifically
        to find the active coordinate system (e.g., "G54").

        Returns:
            The active WCS string if found, otherwise None.
        """
        return None

    async def select_wcs(self, wcs: str) -> None:
        """
        Selects the active Work Coordinate System on the controller.

        For G-code based controllers (GRBL, Smoothie), this is typically
        done via G-code commands during job execution. Drivers that require
        immediate selection should override this method.

        Args:
            wcs: The WCS slot to select (e.g., "G54", "REF0", "MACHINE")
        """

    @abstractmethod
    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Pos | None:
        """
        Initiates a single probing move along the specified axis. The move
        is performed in the negative direction if max_travel is negative.

        Args:
            axis: The axis to probe along.
            max_travel: The maximum distance to travel in mm. The sign
                        indicates direction.
            feed_rate: The speed of the probing move in mm/min.

        Returns:
            The absolute machine coordinates (x, y, z) of the trigger point,
            or None if the probe failed to trigger.
        """

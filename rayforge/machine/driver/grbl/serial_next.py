import asyncio
import logging
from collections.abc import Awaitable, Callable
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import serial
from raydriver.grbl import GrblSession

from ....context import RayforgeContext
from ....core.varset import (
    BaudrateVar,
    IntVar,
    SerialPortVar,
    Var,
    VarSet,
)
from ....pipeline.encoder.base import (
    EncodedOutput,
    OpsEncoder,
)
from ....pipeline.encoder.gcode import GcodeEncoder
from ....shared.units.system import UnitSystem
from ...discovery.spec import DiscoverySpec, SerialRecognizer
from ...transport import TransportStatus
from ...transport.serial import (
    SerialPortPermissionError,
    SerialTransport,
    resolve_serial_port,
)
from ..driver import (
    Axis,
    DeviceConnectionError,
    DeviceError,
    DeviceState,
    DeviceStatus,
    Driver,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    Pos,
)
from .grbl_probe import probe_grbl_device
from .grbl_util import (
    apply_setting_to_varset,
    extract_device_name_from_output,
    get_grbl_setting_varsets,
    is_grbl_output,
)

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from ....core.doc import Doc
    from ...device.profile import DeviceProfile
    from ...models.laser import Laser
    from ...models.machine import Machine

logger = logging.getLogger(__name__)


class GrblSerialNextDriver(Driver):
    """
    The GRBL serial driver with the entire protocol stack (flow
    control, streaming, stall detection and deadlock recovery)
    running in Rust via the ``raydriver`` package.

    This shell only translates between the Rayforge ``Driver``
    interface and the Rust session: dialects remain Rayforge data
    and are passed to the session as resolved command templates.
    """

    label = _("GRBL (Rust)")
    subtitle = _("GRBL-compatible serial connection (Rust driver)")
    supports_settings = True
    reports_granular_progress = True
    supports_probing = True
    supports_unit_detection = True
    maturity = DriverMaturity.EXPERIMENTAL
    DISCOVERY = DiscoverySpec(
        serial=SerialRecognizer(
            label=lambda: _("GRBL device"),
            firmware="grbl",
            matches=is_grbl_output,
            name=extract_device_name_from_output,
        )
    )

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self._session: GrblSession | None = None
        self._machine_wcs = "G53"

    @property
    def machine_space_wcs(self) -> str:
        return self._machine_wcs

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates (G53)")

    @property
    def resource_uri(self) -> str | None:
        if self._session is None:
            return None
        return self._session.resource_uri

    def _require_session(self) -> GrblSession:
        """Returns the session, raising a descriptive error when the
        driver was never set up (e.g. no port configured)."""
        if self._session is None:
            raise DeviceConnectionError(
                _("Driver is not set up. Check the port settings.")
            )
        return self._session

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        """Checks for systemic serial port issues before setup."""
        try:
            SerialTransport.check_serial_permissions_globally()
        except SerialPortPermissionError as e:
            raise DriverPrecheckError(str(e)) from e

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                SerialPortVar(
                    key="port",
                    label=_("Port"),
                    description=(
                        _("Serial port or USB VID:PID (e.g. 0403:6001)")
                    ),
                ),
                BaudrateVar(
                    "baudrate",
                    choices=SerialTransport.list_baud_rates(),
                ),
                Var(
                    key="poll_status_while_running",
                    label=_("Poll device status during jobs"),
                    description=_(
                        "Periodically query the device for position and "
                        "status while a job is running. Warning: Some "
                        "devices have trouble maintaining a stable "
                        "connection if this is used!"
                    ),
                    var_type=bool,
                    default=False,
                ),
                Var(
                    key="deadlock_detection",
                    label=_("Deadlock detection"),
                    description=_(
                        "Detect and recover from serial communication "
                        "deadlocks during jobs. If disabled, the driver "
                        "will simply wait for the machine to respond. "
                        "Disable if you experience false ALARM:3 errors."
                    ),
                    var_type=bool,
                    default=False,
                ),
                IntVar(
                    key="rx_buffer_size_override",
                    label=_("RX Buffer Size Override"),
                    description=_(
                        "Force a specific RX buffer size in bytes. "
                        "Set to 0 to auto-detect from the device."
                    ),
                    default=0,
                    min_val=0,
                    max_val=1024,
                ),
            ]
        )

    @classmethod
    def create_encoder(cls, machine: "Machine") -> "OpsEncoder":
        """Returns a GcodeEncoder configured for the machine's dialect."""
        assert machine.dialect is not None
        return GcodeEncoder(machine.dialect)

    @classmethod
    async def probe(
        cls, context: "RayforgeContext", **kwargs: Any
    ) -> tuple["DeviceProfile", list[str]]:
        return await probe_grbl_device(cls, context, **kwargs)

    def _dialect_templates(self) -> dict[str, Any]:
        """Resolve the interactive-command templates the Rust session
        needs from the machine's dialect."""
        dialect = self.dialect
        templates: dict[str, Any] = {
            "home_all": dialect.home_all,
            "home_axis": dialect.home_axis,
            "move_to": dialect.move_to,
            "jog": dialect.jog,
            "clear_alarm": dialect.clear_alarm,
            "laser_on": dialect.laser_on,
            "laser_off": dialect.laser_off,
            "focus_laser_on": dialect.focus_laser_on,
            "tool_change": dialect.tool_change,
            "set_wcs_offset": dialect.set_wcs_offset,
            "probe_cycle": dialect.probe_cycle,
            "safety_off_commands": dialect.get_safety_off_commands(),
        }
        if dialect.emergency_stop:
            templates["emergency_stop"] = dialect.emergency_stop
        return templates

    def _update_session_dialect(self) -> None:
        if self._session is not None:
            self._session.update_dialect(self._dialect_templates())

    def _setup_implementation(self, **kwargs: Any) -> None:
        port = cast(str, kwargs.get("port", ""))
        baudrate = kwargs.get("baudrate", 115200)
        if not port:
            raise DriverSetupError(_("Port must be configured."))
        if not baudrate:
            raise DriverSetupError(_("Baud rate must be configured."))

        # Note that we intentionally do not check if the serial port
        # exists, as a missing port is a common occurrence when e.g.
        # the USB cable is not plugged in, and not a sign of
        # misconfiguration.
        if port.startswith("/dev/ttyS"):
            logger.warning(
                f"Port {port} is a hardware serial port, which is "
                f"unlikely for USB-based GRBL devices."
            )

        # The Rust session retries its fixed port internally, so a
        # 'vid:pid' spec can only be resolved once, here at setup.
        try:
            port = resolve_serial_port(port)
        except serial.SerialException as e:
            raise DriverSetupError(str(e)) from e

        config = {
            "port": port,
            "baudrate": int(baudrate),
            "poll_status_while_running": bool(
                kwargs.get("poll_status_while_running", False)
            ),
            "deadlock_detection": bool(
                kwargs.get("deadlock_detection", False)
            ),
            "rx_buffer_size_override": int(
                kwargs.get("rx_buffer_size_override", 0) or 0
            ),
        }
        cached_rx_buffer_size = self.config.get("rx_buffer_size")
        if cached_rx_buffer_size is not None:
            config["cached_rx_buffer_size"] = int(cached_rx_buffer_size)

        self._session = GrblSession(
            config=config,
            dialect=self._dialect_templates(),
            event_callback=self._on_session_event,
        )

    def _on_session_event(self, name: str, payload: Any) -> None:
        """Re-emit Rust session events as Rayforge blinker signals."""

        print(f"SHELL EVENT {name} {type(payload).__name__}", flush=True)
        if name == "state_changed":
            state = self._convert_state(payload)
            old_status = self.state.status
            self.state = state
            if state.status != old_status:
                logger.info(
                    f"Device state changed: {state.status.name}",
                    extra=self._log_extra("STATE_CHANGE"),
                )
            self.state_changed.send(self, state=state)
        elif name == "connection_status_changed":
            status_str, message = payload
            logger.info(
                f"Connection status: {status_str}"
                + (f" - {message}" if message else ""),
                extra=self._log_extra("MACHINE_EVENT"),
            )
            self.connection_status_changed.send(
                self,
                status=TransportStatus[status_str],
                message=message,
            )
        elif name == "command_status_changed":
            status_str, message = payload
            self.command_status_changed.send(
                self,
                status=TransportStatus[status_str],
                message=message,
            )
        elif name == "job_finished":
            self.job_finished.send(self)
        elif name == "probe_status_changed":
            self.probe_status_changed.send(self, message=payload)
        elif name == "wcs_updated":
            self.wcs_updated.send(self, offsets=payload)
        elif name == "config_changed":
            key, value = payload
            self.config[key] = value
            self.config_changed.send(self)
        else:  # pragma: no cover - unknown events are ignored
            logger.debug(f"Ignoring session event: {name}")

    @staticmethod
    def _convert_state(rd_state: Any) -> DeviceState:
        """Convert a raydriver DeviceState into a Rayforge one."""
        error = None
        rd_error = rd_state.error
        if rd_error is not None:
            error = DeviceError(
                rd_error.code,
                rd_error.title,
                rd_error.description,
            )
        return DeviceState(
            status=DeviceStatus[rd_state.status.name],
            error=error,
            machine_pos=tuple(rd_state.machine_pos),
            work_pos=tuple(rd_state.work_pos),
            wco=tuple(rd_state.wco),
            feed_rate=rd_state.feed_rate,
            spindle_speed=rd_state.spindle_speed,
            buffer_available=rd_state.buffer_available,
            buffer_rx_available=rd_state.buffer_rx_available,
        )

    async def cleanup(self):
        logger.debug("Cleanup initiated.")
        if self._session is not None:
            await self._session.disconnect()
            self._session = None
        await super().cleanup()
        logger.debug("Cleanup completed.")

    async def _connect_implementation(self):
        """Launches the Rust connection loop and returns, allowing the
        UI to remain responsive."""
        if self._session is None:
            logger.error(
                "Cannot connect: session not initialized "
                "(check port settings)."
            )
            self.connection_status_changed.send(
                self,
                status=TransportStatus.ERROR,
                message=_("Port not configured"),
            )
            return
        self._update_session_dialect()
        await self._session.connect()

    async def execute_interactive_command(self, command: str) -> list[str]:
        """Send a command and await its full response, blocking other
        commands from interleaving; used by device probing."""
        return await self._require_session().execute_interactive_command(
            command
        )

    def get_setting_vars(self) -> list["VarSet"]:
        return get_grbl_setting_varsets()

    @staticmethod
    def _op_line_map(op_map: Any) -> dict[int, int]:
        """Extract the {line index: op index} mapping for the Rust
        session's per-line progress reporting."""
        result = {}
        for line_idx in range(op_map.line_count):
            op_index = op_map.op_for_line(line_idx)
            if op_index is not None:
                result[line_idx] = op_index
        return result

    async def run(
        self,
        encoded: EncodedOutput,
        doc: "Doc",
        ops: "Ops",
        on_command_done: Callable[[int], None | Awaitable[None]] | None = None,
    ) -> None:
        session = self._require_session()
        self._update_session_dialect()
        command_times = ops.estimate_command_times(
            default_feed_rate=self._machine.max_cut_speed,
            default_rapid_rate=self._machine.max_travel_speed,
            acceleration=self._machine.acceleration,
        )
        estimates = [float(t) for t in command_times]
        line_map = self._op_line_map(encoded.op_map)

        def _on_command_done(op_index: int):
            if on_command_done is None:
                return
            result = on_command_done(op_index)
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)

        try:
            await session.run(
                encoded.text,
                line_map,
                estimates,
                None if on_command_done is None else _on_command_done,
            )
        except DeviceConnectionError as e:
            logger.warning(
                f"Job terminated due to device error: {e}. "
                "Connection remains active."
            )
        except Exception:
            logger.exception("Job terminated with unexpected error")

    async def run_raw(self, machine_code: str) -> None:
        session = self._require_session()
        try:
            await session.run_raw(machine_code)
        except DeviceConnectionError as e:
            logger.warning(
                f"Raw G-code terminated due to device error: {e}. "
                "Connection remains active."
            )
        except Exception:
            logger.exception("Raw G-code terminated with unexpected error")

    async def cancel(self, emergency: bool = False) -> None:
        if self._session is None:
            raise ConnectionError("Serial transport not initialized")
        await self._session.cancel(emergency)

    async def set_hold(self, hold: bool = True) -> None:
        await self._require_session().set_hold(hold)

    def can_home(self, axis: Axis | None = None) -> bool:
        """GRBL supports homing for all axes."""
        return True

    async def home(self, axes: Axis | None = None) -> None:
        session = self._require_session()
        names = None
        if axes is not None:
            names = [axis.name for axis in axes]
        await session.home(names, self._machine.active_wcs)

    async def move_to(
        self,
        pos_x: float,
        pos_y: float,
        pos_z: float | None = None,
        speed: float | None = None,
    ) -> None:
        cmd = self._format_move_to(float(pos_x), float(pos_y), pos_z, speed)
        await self._require_session().execute_command(cmd)

    async def select_tool(self, tool_number: int) -> None:
        """Sends a tool change command for the given tool number."""
        await self._require_session().select_tool(tool_number)

    async def clear_alarm(self) -> None:
        session = self._require_session()
        dialect = self.dialect
        response = await session.execute_command(dialect.clear_alarm)
        has_error = any(line.startswith("error:") for line in response)
        if not has_error:
            self.state.error = None
            self.state_changed.send(self, state=self.state)

    async def set_power(self, head: "Laser", percent: float) -> None:
        """Sets the laser power (0.0-1.0 of max power)."""
        power = percent * head.max_power if percent > 0 else None
        await self._require_session().set_power(power)

    async def set_focus_power(self, head: "Laser", percent: float) -> None:
        """Sets the laser power for focus mode."""
        power = percent * head.max_power if percent > 0 else None
        await self._require_session().set_focus_power(power)

    def can_jog(self, axis: Axis | None = None) -> bool:
        """GRBL supports jogging for all axes."""
        return True

    async def jog(self, speed: int, **deltas: float) -> None:
        session = self._require_session()
        converted = [
            (name, self._to_machine_length(distance))
            for name, distance in deltas.items()
        ]
        await session.jog(self._to_machine_speed(speed), converted)

    async def detect_unit_system(self) -> UnitSystem | None:
        """Queries the device's ``$$`` settings and infers the unit
        system from the ``$13`` (Report in inches) flag."""
        result = await self._require_session().detect_unit_system()
        if result == "metric":
            return UnitSystem.METRIC
        if result == "imperial":
            return UnitSystem.IMPERIAL
        return None

    async def read_settings(self) -> None:
        pairs = await self._require_session().read_settings()

        known_varsets = self.get_setting_vars()
        key_to_varset_map = {
            var_key: varset
            for varset in known_varsets
            for var_key in varset.keys()  # noqa: SIM118
        }
        unknown_vars = VarSet(
            title=_("Unknown Settings"),
            description=_(
                "Settings reported by the device not in the standard list."
            ),
        )
        for key, value_str in pairs:
            target_varset = key_to_varset_map.get(key)
            if target_varset:
                apply_setting_to_varset(target_varset, key, value_str)
            else:
                unknown_vars.add(
                    Var(
                        key=key,
                        label=f"${key}",
                        var_type=str,
                        value=value_str,
                        description=_("Unknown setting from device"),
                    )
                )
        result = known_varsets
        if len(unknown_vars) > 0:
            result.append(unknown_vars)

        num_settings = sum(len(vs) for vs in result)
        logger.info(
            f"Driver settings read with {num_settings} settings.",
            extra={"log_category": "DRIVER_EVENT"},
        )
        self.settings_read.send(self, settings=result)

    async def write_setting(self, key: str, value: Any) -> None:
        if isinstance(value, bool):
            value = 1 if value else 0
        await self._require_session().write_setting(key, str(value))

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None:
        session = self._require_session()
        await session.set_wcs_offset(
            wcs_slot,
            self._to_machine_length(x),
            self._to_machine_length(y),
            self._to_machine_length(z) if z is not None else None,
        )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        offsets = await self._require_session().read_wcs_offsets()
        return {slot: tuple(pos) for slot, pos in offsets.items()}

    async def read_parser_state(self) -> str | None:
        """Reads the $G parser state to determine the active WCS."""
        try:
            return await self._require_session().read_parser_state()
        except DeviceConnectionError as e:
            logger.warning(f"Could not read parser state: {e}")
            return None

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Pos | None:
        session = self._require_session()
        assert axis.name, "Probing requires a single, named axis."
        return await session.run_probe_cycle(
            axis.name.upper(),
            self._to_machine_length(max_travel),
            self._to_machine_speed(feed_rate),
        )

    def get_error(self, error_code: str) -> DeviceError | None:
        """Returns error details for a given GRBL error code."""
        from .grbl_util import error_code_to_device_error

        return error_code_to_device_error(error_code)

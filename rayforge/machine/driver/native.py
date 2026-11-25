"""Generic base for drivers backed by a raydriver driver.

A raydriver driver — local Rust object or remote daemon proxy —
exposes one protocol-agnostic method and event surface. This base
translates that surface to the Rayforge ``Driver`` interface:
dialects stay Rayforge data and are passed to the driver as resolved
command templates. Subclasses bind the native class and supply the
connection config, presentation and protocol-specific capabilities
(settings, probing).
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from raydriver.core import DriverEvent

from ...context import RayforgeContext
from ...core.varset import VarSet
from ...pipeline.artifact.encoded import EncodedOutput
from ...shared.units.system import UnitSystem
from ..transport import TransportStatus
from .driver import (
    Axis,
    DeviceConnectionError,
    DeviceError,
    Driver,
    DriverSetupError,
    Pos,
)
from .session_state import from_session_state

if TYPE_CHECKING:
    from raydriver.core.types import DeviceState as NativeDeviceState
    from raygeo.ops import Ops

    from ...core.doc import Doc
    from ..models.laser import Laser
    from ..models.machine import Machine


#: Homing seek rates are typically well below the configured travel
#: speed, and pull-off moves add time; scale the predicted travel time
#: to keep the timeout a ceiling rather than an expected wait.
_HOMING_SAFETY_FACTOR = 8.0
_HOMING_MIN_TIMEOUT_S = 30.0


@runtime_checkable
class NativeSession(Protocol):
    """The uniform session surface every raydriver session exposes.

    The local Rust sessions and the remote daemon proxy implement
    the same methods with the same signatures; operations a
    protocol cannot perform raise ``NotImplementedError`` instead
    of being absent.
    """

    state: "NativeDeviceState"
    resource_uri: str | None

    def update_dialect(self, dialect: dict[str, Any]) -> None: ...

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    def probe_device(
        self, boot_lines: list[str] | None = None
    ) -> Awaitable[Any]: ...

    async def execute_interactive_command(self, command: str) -> list[str]: ...

    async def execute_command(self, command: str) -> list[str]: ...

    async def run(
        self,
        gcode: str,
        op_line_map: dict[int, int] | None = None,
        op_estimates: list[float] | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None: ...

    async def run_raw(self, machine_code: str) -> None: ...

    async def cancel(self, emergency: bool = False) -> None: ...

    async def set_hold(self, hold: bool = True) -> None: ...

    async def home(
        self, axes: list[str] | None = None, active_wcs: str | None = None
    ) -> None: ...

    async def move_to(
        self, speed: float, x: float, y: float, z: float | None = None
    ) -> None: ...

    async def jog(
        self, speed: int, deltas: list[tuple[str, float]] | None = None
    ) -> None: ...

    async def select_tool(self, tool_number: int) -> None: ...

    async def set_power(self, power: float | None = None) -> None: ...

    async def set_focus_power(self, power: float | None = None) -> None: ...

    async def clear_alarm(self) -> None: ...

    async def detect_unit_system(self) -> str | None: ...

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None: ...

    async def read_wcs_offsets(self) -> dict[str, tuple]: ...

    async def read_parser_state(self) -> str | None: ...

    async def run_probe_cycle(
        self, axis_letter: str, max_travel: float, feed_rate: int
    ) -> Any: ...


class NativeDriver(Driver):
    """Driver machinery shared by every native raydriver driver.

    Subclasses set :attr:`NATIVE` to the raydriver class and may
    override :meth:`build_config` / :meth:`_create_session` for
    protocol-specific connection parameters.
    """

    reports_granular_progress = True

    #: The raydriver class this Rayforge driver binds.
    NATIVE: type[Any] | None = None

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self._session: Any = None
        self._machine_wcs = "G53"

    def build_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Translate driver arguments into the native config dict."""
        return dict(kwargs)

    def _create_session(self, config: dict[str, Any]) -> NativeSession:
        """Construct the native driver."""
        if self.NATIVE is None:
            raise DriverSetupError(
                _("This driver has no native implementation.")
            )
        return self.NATIVE(
            config=config,
            dialect=self._dialect_templates(),
            event_callback=self._on_session_event,
        )

    def _setup_implementation(self, **kwargs: Any) -> None:
        self._session = self._create_session(self.build_config(kwargs))

    async def read_settings(self) -> None:
        await self._require_session().read_settings()
        self.settings_read.send(self, settings=[])

    async def write_setting(self, key: str, value: Any) -> None:
        if isinstance(value, bool):
            value = 1 if value else 0
        await self._require_session().write_setting(key, str(value))

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

    def _estimate_homing_timeout(self) -> float:
        """Predict a homing wait budget from axis length and speed.

        The longest axis travel is divided by the machine's configured
        travel speed; a safety factor covers the slower seek rate and
        the pull-off moves. The result is a ceiling for the session,
        not an expected wait.
        """
        machine = self._machine
        distances = list(machine.axis_extents)
        z_extents = machine.z_extents
        if z_extents is not None:
            distances.append(z_extents[1] - z_extents[0])
        distance = max(distances) if distances else 0.0
        speed = float(machine.max_travel_speed)
        if speed <= 0.0:
            speed = 3000.0
        seconds = distance / speed * 60.0
        return max(seconds * _HOMING_SAFETY_FACTOR, _HOMING_MIN_TIMEOUT_S)

    @property
    def session(self) -> NativeSession | None:
        """The underlying raydriver session, if set up."""
        return self._session

    def _require_session(self) -> Any:
        if self._session is None:
            raise DeviceConnectionError(
                _("Driver is not set up. Check the connection settings.")
            )
        return self._session

    @property
    def _logger(self) -> logging.Logger:
        """The concrete driver's module logger.

        Log records (raw I/O in particular) belong to the driver
        module, so per-driver log filters keep working as they did
        when every driver carried its own logging code.
        """
        return logging.getLogger(type(self).__module__)

    # -- dialect -------------------------------------------------------

    def _dialect_templates(self) -> dict[str, Any]:
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

    def refresh_dialect(self) -> None:
        self._update_session_dialect()

    # -- events --------------------------------------------------------

    def _on_session_event(self, event: DriverEvent) -> None:
        match event:
            case DriverEvent.StateChanged():
                self._on_state_changed(event.state)
            case DriverEvent.ConnectionStatusChanged():
                self._on_connection_status_changed(event.status, event.message)
            case DriverEvent.CommandStatusChanged():
                self._on_command_status_changed(event.status, event.message)
            case DriverEvent.JobFinished():
                self.job_finished.send(self)
            case DriverEvent.ProbeStatusChanged():
                self.probe_status_changed.send(self, message=event.message)
            case DriverEvent.WcsUpdated():
                self.wcs_updated.send(self, offsets=event.offsets)
            case DriverEvent.ConfigChanged():
                self._on_config_changed(event.key, event.value)
            case DriverEvent.RawRx():
                self._log_raw_line(event.line, received=True)
            case DriverEvent.RawTx():
                self._log_raw_line(event.line, received=False)
            case _:
                self._logger.debug(f"Ignoring session event: {event!r}")

    def _on_state_changed(self, session_state: "NativeDeviceState") -> None:
        state = from_session_state(session_state)
        old_status = self.state.status
        self.state = state
        if state.status != old_status:
            self._logger.info(
                f"Device state changed: {state.status.name}",
                extra=self._log_extra("STATE_CHANGE"),
            )
        self.state_changed.send(self, state=state)

    def _on_connection_status_changed(
        self, status_str: str, message: str | None
    ) -> None:
        self._logger.info(
            f"Connection status: {status_str}"
            + (f" - {message}" if message else ""),
            extra=self._log_extra("MACHINE_EVENT"),
        )
        self.connection_status_changed.send(
            self,
            status=TransportStatus[status_str],
            message=message,
        )

    def _on_command_status_changed(
        self, status_str: str, message: str | None
    ) -> None:
        self.command_status_changed.send(
            self,
            status=TransportStatus[status_str],
            message=message,
        )

    def _on_config_changed(self, key: str, value: int) -> None:
        self.config[key] = value
        self.config_changed.send(self)

    def _log_raw_line(self, line: str, received: bool) -> None:
        if received:
            if line.startswith("<") and line.endswith(">"):
                category = "STATUS_POLL"
            elif line == "ok" or line.startswith("error:"):
                category = "MACHINE_RESPONSE"
            else:
                category = "MACHINE_EVENT"
        else:
            category = "RAW_IO"
        self._logger.info(line, extra=self._log_extra(category))

    # -- lifecycle -----------------------------------------------------

    async def cleanup(self):
        self._logger.debug("Cleanup initiated.")
        if self._session is not None:
            await self._session.disconnect()
            self._session = None
        await super().cleanup()
        self._logger.debug("Cleanup completed.")

    async def _connect_implementation(self):
        if self._session is None:
            self._logger.error(
                "Cannot connect: session not initialized "
                "(check connection settings)."
            )
            self.connection_status_changed.send(
                self,
                status=TransportStatus.ERROR,
                message=_("Connection not configured"),
            )
            return
        self._update_session_dialect()
        await self._session.connect()

    async def execute_interactive_command(self, command: str) -> list[str]:
        return await self._require_session().execute_interactive_command(
            command
        )

    # -- jobs ----------------------------------------------------------

    @staticmethod
    def _op_line_map(op_map: Any) -> dict[int, int]:
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
            self._logger.warning(
                f"Job terminated due to device error: {e}. "
                "Connection remains active."
            )
        except Exception:
            self._logger.exception("Job terminated with unexpected error")

    async def run_raw(self, machine_code: str) -> None:
        session = self._require_session()
        try:
            await session.run_raw(machine_code)
        except DeviceConnectionError as e:
            self._logger.warning(
                f"Raw G-code terminated due to device error: {e}. "
                "Connection remains active."
            )
        except Exception:
            self._logger.exception(
                "Raw G-code terminated with unexpected error"
            )

    async def cancel(self, emergency: bool = False) -> None:
        if self._session is None:
            raise ConnectionError("Session not initialized")
        await self._session.cancel(emergency)

    async def set_hold(self, hold: bool = True) -> None:
        await self._require_session().set_hold(hold)

    # -- motion --------------------------------------------------------

    def can_home(self, axis: Axis | None = None) -> bool:
        return True

    async def home(self, axes: Axis | None = None) -> None:
        session = self._require_session()
        names = None
        if axes is not None:
            names = [axis.name for axis in axes]
        await session.home(names, self._machine.active_wcs)
        park = self._machine.park_position_after_homing
        if axes is None and park is not None:
            template = self.dialect.move_to_park
            if template:
                await session.execute_command(
                    template.format(x=park[0], y=park[1])
                )

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
        await self._require_session().select_tool(tool_number)

    async def clear_alarm(self) -> None:
        session = self._require_session()
        response = await session.execute_command(self.dialect.clear_alarm)
        if not any(line.startswith("error:") for line in response):
            self.state.error = None
            self.state_changed.send(self, state=self.state)

    async def set_power(self, head: "Laser", percent: float) -> None:
        power = percent * head.max_power if percent > 0 else None
        await self._require_session().set_power(power)

    async def set_focus_power(self, head: "Laser", percent: float) -> None:
        power = percent * head.max_power if percent > 0 else None
        await self._require_session().set_focus_power(power)

    def can_jog(self, axis: Axis | None = None) -> bool:
        return True

    async def jog(self, speed: int, **deltas: float) -> None:
        session = self._require_session()
        converted = [
            (name, self._to_machine_length(distance))
            for name, distance in deltas.items()
        ]
        await session.jog(self._to_machine_speed(speed), converted)

    async def detect_unit_system(self) -> UnitSystem | None:
        try:
            result = await self._require_session().detect_unit_system()
        except NotImplementedError:
            return None
        if result == "metric":
            return UnitSystem.METRIC
        if result == "imperial":
            return UnitSystem.IMPERIAL
        return None

    # -- WCS / probing -------------------------------------------------

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None:
        await self._require_session().set_wcs_offset(
            wcs_slot,
            self._to_machine_length(x),
            self._to_machine_length(y),
            self._to_machine_length(z) if z is not None else None,
        )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        offsets = await self._require_session().read_wcs_offsets()
        return {slot: tuple(pos) for slot, pos in offsets.items()}

    async def read_parser_state(self) -> str | None:
        try:
            return await self._require_session().read_parser_state()
        except NotImplementedError:
            return None
        except DeviceConnectionError as e:
            self._logger.warning(f"Could not read parser state: {e}")
            return None

    async def run_probe_cycle(
        self, axis: Axis, target: float, feed_rate: int
    ) -> Pos | None:
        session = self._require_session()
        assert axis.name, "Probing requires a single, named axis."
        return await session.run_probe_cycle(
            axis.name.upper(),
            self._to_machine_length(target),
            self._to_machine_speed(feed_rate),
        )

    # -- settings -----------------------------------------------------

    def get_setting_vars(self) -> list["VarSet"]:
        return []

    def get_error(self, error_code: str) -> DeviceError | None:
        return None

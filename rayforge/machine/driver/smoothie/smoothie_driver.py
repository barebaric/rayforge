import asyncio
import logging
from collections.abc import Awaitable, Callable
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from raydriver.core import SessionEvent
from raydriver.smoothie import SmoothieSession

from ....context import RayforgeContext
from ....core.varset import HostnameVar, PortVar, VarSet
from ....core.varset.hostnamevar import is_valid_hostname_or_ip
from ....pipeline.encoder.base import EncodedOutput, OpsEncoder
from ....pipeline.encoder.gcode import GcodeEncoder
from ...transport import TransportStatus
from ..driver import (
    Axis,
    DeviceConnectionError,
    DeviceStatus,
    Driver,
    DriverPrecheckError,
    DriverSetupError,
    Pos,
)
from ..session_state import from_session_state
from .smoothie_util import SmoothieProbeResult, build_smoothie_profile

if TYPE_CHECKING:
    from raydriver.core.types import DeviceState as SessionDeviceState
    from raygeo.ops import Ops

    from ....core.doc import Doc
    from ...device.profile import DeviceProfile
    from ...models.laser import Laser
    from ...models.machine import Machine

logger = logging.getLogger(__name__)


class SmoothieDriver(Driver):
    """
    Handles Smoothie-based devices via Telnet, with the entire
    protocol stack running in Rust via the ``raydriver`` package.

    This shell only translates between the Rayforge ``Driver``
    interface and the Rust session: dialects remain Rayforge data
    and are passed to the session as resolved command templates.
    """

    label = _("Smoothie")
    subtitle = _("Smoothieware via a Telnet connection")
    supports_settings = False
    reports_granular_progress = True
    supports_probing = True

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self._session: SmoothieSession | None = None

    @property
    def machine_space_wcs(self) -> str:
        return "G53"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates (G53)")

    @property
    def resource_uri(self) -> str | None:
        if self._session is None:
            return None
        return self._session.resource_uri

    def _require_session(self) -> SmoothieSession:
        """Returns the session, raising a descriptive error when the
        driver was never set up (e.g. no host configured)."""
        if self._session is None:
            raise DeviceConnectionError(
                _("Driver is not set up. Check the connection settings.")
            )
        return self._session

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        """Checks if the hostname is a valid format."""
        host = cast(str, kwargs.get("host", ""))
        if not is_valid_hostname_or_ip(host):
            raise DriverPrecheckError(
                _("Invalid hostname or IP address: '{host}'").format(host=host)
            )

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                HostnameVar(
                    key="host",
                    label=_("Hostname"),
                    description=_("The IP address or hostname of the device"),
                ),
                PortVar(
                    key="port",
                    label=_("Port"),
                    description=_("The Telnet port number"),
                    default=23,
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
        """Connect to a Smoothie device, query its configuration, and
        return an auto-populated ``(DeviceProfile, warnings)`` tuple.
        """
        from ...models.machine import Machine

        machine = Machine(context)
        driver = cls(context, machine)
        driver.setup(**kwargs)

        connected = asyncio.Event()

        def _on_status(sender, status=None, message=None, **kw):
            if status == TransportStatus.CONNECTED:
                connected.set()

        driver.connection_status_changed.connect(_on_status)
        try:
            await driver.connect()
            await asyncio.wait_for(connected.wait(), timeout=15.0)
            # Smoothieware needs a moment after the link comes up
            # before it accepts commands.
            await asyncio.sleep(0.2)
            query = driver.execute_interactive_command
            probe = SmoothieProbeResult(
                version=await query("version"),
                alpha_max=await query("config-get alpha_max"),
                beta_max=await query("config-get beta_max"),
                alpha_max_rate=await query("config-get alpha_max_rate"),
                beta_max_rate=await query("config-get beta_max_rate"),
                acceleration=await query("config-get acceleration"),
            )
        finally:
            driver.connection_status_changed.disconnect(_on_status)
            await driver.cleanup()
            context.dialect_mgr.dialects_changed.disconnect(
                machine._on_dialects_changed
            )

        profile, warnings = build_smoothie_profile(probe)
        profile.machine_config.driver = cls.__name__
        profile.machine_config.driver_args = kwargs
        return profile, warnings

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

    def refresh_dialect(self) -> None:
        self._update_session_dialect()

    def _setup_implementation(self, **kwargs: Any) -> None:
        host = cast(str, kwargs.get("host", ""))
        port = kwargs.get("port", 23)

        if not host:
            raise DriverSetupError(_("Hostname must be configured."))

        config = {
            "host": host,
            "tcp_port": int(port),
        }
        self._session = SmoothieSession(
            config=config,
            dialect=self._dialect_templates(),
            event_callback=self._on_session_event,
        )

    def _on_session_event(self, event: SessionEvent) -> None:
        """Re-emit Rust session events as Rayforge blinker signals."""
        match event:
            case SessionEvent.StateChanged():
                self._on_state_changed(event.state)
            case SessionEvent.ConnectionStatusChanged():
                self._on_connection_status_changed(event.status, event.message)
            case SessionEvent.CommandStatusChanged():
                self._on_command_status_changed(event.status, event.message)
            case SessionEvent.JobFinished():
                self.job_finished.send(self)
            case SessionEvent.ProbeStatusChanged():
                self.probe_status_changed.send(self, message=event.message)
            case SessionEvent.RawRx():
                self._log_raw_line(event.line, received=True)
            case SessionEvent.RawTx():
                self._log_raw_line(event.line, received=False)
            case _:
                logger.debug(f"Ignoring session event: {event!r}")

    def _on_state_changed(self, session_state: "SessionDeviceState") -> None:
        state = from_session_state(session_state)
        old_status = self.state.status
        self.state = state
        if state.status != old_status:
            logger.info(
                f"Device state changed: {state.status.name}",
                extra=self._log_extra("STATE_CHANGE"),
            )
        self.state_changed.send(self, state=state)

    def _on_connection_status_changed(
        self, status_str: str, message: str | None
    ) -> None:
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
        if status_str in ("ERROR", "DISCONNECTED") and (
            self.state.status != DeviceStatus.UNKNOWN
        ):
            self.state.status = DeviceStatus.UNKNOWN
            self.state_changed.send(self, state=self.state)

    def _on_command_status_changed(
        self, status_str: str, message: str | None
    ) -> None:
        self.command_status_changed.send(
            self,
            status=TransportStatus[status_str],
            message=message,
        )

    def _log_raw_line(self, line: str, received: bool) -> None:
        """Log a raw line from the session's communication so the
        console (with 'Show verbose' enabled) shows the full
        conversation with the device."""
        if received:
            if line.startswith("<") and line.endswith(">"):
                category = "STATUS_POLL"
            elif line == "ok" or line.startswith("error:"):
                category = "MACHINE_RESPONSE"
            else:
                category = "MACHINE_EVENT"
        else:
            category = "RAW_IO"
        logger.info(line, extra=self._log_extra(category))

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
                "(check host settings)."
            )
            self.connection_status_changed.send(
                self,
                status=TransportStatus.ERROR,
                message=_("Host not configured"),
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
        return [VarSet()]

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
            raise ConnectionError("Connection not initialized")
        await self._session.cancel(emergency)

    async def set_hold(self, hold: bool = True) -> None:
        await self._require_session().set_hold(hold)

    def can_home(self, axis: Axis | None = None) -> bool:
        """Smoothie supports homing for all axes."""
        return True

    async def home(self, axes: Axis | None = None) -> None:
        session = self._require_session()
        names = None
        if axes is not None:
            names = [axis.name for axis in axes]
        await session.home(names)

    async def move_to(
        self,
        pos_x: float,
        pos_y: float,
        pos_z: float | None = None,
        speed: float | None = None,
    ) -> None:
        cmd = self._format_move_to(float(pos_x), float(pos_y), pos_z, speed)
        await self._require_session().execute_command(cmd)

    def can_jog(self, axis: Axis | None = None) -> bool:
        """Smoothie supports jogging for all axes."""
        return True

    async def jog(self, speed: int, **deltas: float) -> None:
        session = self._require_session()
        converted = [
            (name, self._to_machine_length(distance))
            for name, distance in deltas.items()
        ]
        await session.jog(self._to_machine_speed(speed), converted)

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

    async def read_settings(self) -> None:
        raise NotImplementedError(
            "Device settings not implemented for this driver"
        )

    async def write_setting(self, key: str, value: Any) -> None:
        raise NotImplementedError(
            "Device settings not implemented for this driver"
        )

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None:
        """Sets a WCS offset using Smoothie's G10 L20 command."""
        session = self._require_session()
        await session.set_wcs_offset(
            wcs_slot,
            self._to_machine_length(x),
            self._to_machine_length(y),
            self._to_machine_length(z) if z is not None else None,
        )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        """Reading all WCS offsets is not supported by Smoothie."""
        raise NotImplementedError(
            "Reading all WCS offsets is not reliably supported "
            "by Smoothieware."
        )

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Pos | None:
        """
        Probing is not implemented due to difficulty in reliably capturing
        real-time probe position feedback over the standard Telnet protocol.
        """
        raise NotImplementedError(
            "Probing is not implemented for the Smoothie driver via Telnet."
        )

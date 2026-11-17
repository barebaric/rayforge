import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import aiohttp
from raydriver.core import SessionEvent
from raydriver.xtool import XtoolSession

from ....context import RayforgeContext
from ....core.varset import HostnameVar, PortVar, VarSet
from ....core.varset.hostnamevar import is_valid_hostname_or_ip
from ....pipeline.encoder.base import EncodedOutput, OpsEncoder
from ....pipeline.encoder.gcode import GcodeEncoder
from ....shared.units.system import UnitSystem
from ...transport import TransportStatus
from ..driver import (
    Axis,
    DeviceConnectionError,
    DeviceError,
    Driver,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    Pos,
)
from ..session_state import from_session_state

if TYPE_CHECKING:
    from raydriver.core.types import DeviceState as SessionDeviceState
    from raygeo.ops import Ops

    from ....core.doc import Doc
    from ...device.profile import DeviceProfile
    from ...models.laser import Laser
    from ...models.machine import Machine

logger = logging.getLogger(__name__)

# Default network ports of xTool D-series machines (proprietary WiFi
# API).
DEFAULT_HTTP_PORT = 8080
DEFAULT_WS_PORT = 8081

XTOOL_MACHINE_TYPE_URL = "/getmachinetype"


async def fetch_xtool_machine_type(
    host: str, port: int, timeout: float = 2.0
) -> str | None:
    """
    GETs the xTool machine-type endpoint and returns the reported
    machine type (e.g. ``"xTool D1Pro"``) when the device identifies
    itself as an xTool machine, ``None`` otherwise.  Never raises.
    """
    url = f"http://{host}:{port}{XTOOL_MACHINE_TYPE_URL}"
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                url, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response,
        ):
            if response.status != 200:
                return None
            text = await response.text()
    except Exception:
        logger.debug(
            "xTool machine-type query of %s:%s failed",
            host,
            port,
            exc_info=True,
        )
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    machine_type = str(payload.get("type") or "")
    if "xtool" in machine_type.lower():
        return machine_type
    return None


class XtoolNetworkDriver(Driver):
    """
    Driver for xTool D-series machines (D1, D1 Pro, …) over their
    proprietary WiFi API, with the entire protocol stack running in
    Rust via the ``raydriver`` package.

    xTool machines are GRBL-derived, but their WiFi module is neither
    ESP3D nor GRBL-over-TCP: commands are acknowledged through the
    HTTP result envelope, the WebSocket only carries job-state
    events, and the machine reports no position feedback at all.

    This shell only translates between the Rayforge ``Driver``
    interface and the Rust session: dialects remain Rayforge data
    and are passed to the session as resolved command templates.
    """

    label = _("xTool (Wi-Fi)")
    subtitle = _("xTool machines with Wi-Fi (D1, D1 Pro, …)")
    maturity = DriverMaturity.EXPERIMENTAL
    supports_settings = False
    reports_granular_progress = False
    supports_probing = True
    supports_unit_detection = True
    DISCOVERY = None

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self._session: XtoolSession | None = None

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

    def _require_session(self) -> XtoolSession:
        """Returns the session, raising a descriptive error when the
        driver was never set up (e.g. no host configured)."""
        if self._session is None:
            raise DeviceConnectionError(
                _("Driver is not set up. Check the network settings.")
            )
        return self._session

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
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
                    key="http_port",
                    label=_("HTTP Port"),
                    description=_("The HTTP API port of the device"),
                    default=DEFAULT_HTTP_PORT,
                ),
                PortVar(
                    key="ws_port",
                    label=_("WebSocket Port"),
                    description=_(
                        "The WebSocket notification port of the device"
                    ),
                    default=DEFAULT_WS_PORT,
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
        """
        Detects the machine type via /getmachinetype and builds a
        minimal profile from it.  The xTool WiFi API exposes no ``$$``
        settings, so nothing else can be probed.
        """
        from ...device.profile import (
            DeviceMeta,
            DeviceProfile,
            MachineConfig,
        )

        kwargs = dict(kwargs)
        host = cast(str, kwargs.get("host", ""))
        port = cast(int, kwargs.get("http_port", DEFAULT_HTTP_PORT))
        machine_type = await fetch_xtool_machine_type(host, port)
        if machine_type is None and port != DEFAULT_HTTP_PORT:
            machine_type = await fetch_xtool_machine_type(
                host, DEFAULT_HTTP_PORT
            )
            if machine_type is not None:
                kwargs["http_port"] = DEFAULT_HTTP_PORT
        if machine_type is None:
            raise DriverSetupError(
                _(
                    "No xTool machine answered at {host}:{port}. Check"
                    " the IP address and make sure the machine is"
                    " connected to the network."
                ).format(host=host, port=port)
            )
        kwargs.setdefault("http_port", DEFAULT_HTTP_PORT)
        kwargs.setdefault("ws_port", DEFAULT_WS_PORT)

        profile = DeviceProfile(
            meta=DeviceMeta(
                name=machine_type,
                description=_("Auto-configured via probe wizard"),
            ),
            machine_config=MachineConfig(unit_system=UnitSystem.METRIC),
            dialect_config={},
        )
        profile.machine_config.driver = cls.__name__
        profile.machine_config.driver_args = kwargs
        return profile, []

    def _dialect_templates(self) -> dict[str, Any]:
        """Resolve the command templates the Rust session needs from
        the machine's dialect."""
        dialect = self.dialect
        templates: dict[str, Any] = {
            "laser_on": dialect.laser_on,
            "laser_off": dialect.laser_off,
            "focus_laser_on": dialect.focus_laser_on,
            "safety_off_commands": dialect.get_safety_off_commands(),
        }
        if dialect.emergency_stop:
            templates["emergency_stop"] = dialect.emergency_stop
        return templates

    def _setup_implementation(self, **kwargs: Any) -> None:
        host = cast(str, kwargs.get("host", ""))
        if not host:
            raise DriverSetupError(_("Hostname must be configured."))
        http_port = cast(int, kwargs.get("http_port", DEFAULT_HTTP_PORT))
        ws_port = cast(int, kwargs.get("ws_port", DEFAULT_WS_PORT))

        config = {
            "host": host,
            "http_port": int(http_port),
            "ws_port": int(ws_port),
        }
        self._session = XtoolSession(
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
                self.command_status_changed.send(
                    self,
                    status=TransportStatus[event.status],
                    message=event.message,
                )
            case SessionEvent.JobFinished():
                self.job_finished.send(self)
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

    def _log_raw_line(self, line: str, received: bool) -> None:
        """Log a raw line from the session's communication so the
        console (with 'Show verbose' enabled) shows the full
        conversation with the device."""
        if received:
            category = (
                "MACHINE_RESPONSE"
                if line.startswith("ok:")
                else "MACHINE_EVENT"
            )
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
                "(check network settings)."
            )
            self.connection_status_changed.send(
                self,
                status=TransportStatus.ERROR,
                message=_("Host not configured"),
            )
            return
        await self._session.connect()

    async def execute_interactive_command(self, command: str) -> list[str]:
        """Send a command and await the device's HTTP acknowledgement.

        The xTool protocol has no per-line response text, so the
        result is always empty."""
        return await self._require_session().execute_interactive_command(
            command
        )

    def get_setting_vars(self) -> list["VarSet"]:
        return []

    async def run(
        self,
        encoded: EncodedOutput,
        doc: "Doc",
        ops: "Ops",
        on_command_done: Callable[[int], None | Awaitable[None]] | None = None,
    ) -> None:
        session = self._require_session()
        if on_command_done is not None:
            # No granular progress exists on this protocol; report
            # every op as done once the job is running on the device.
            num_ops = encoded.op_map.op_count if encoded.op_map else 0
            for op_index in range(num_ops):
                result = on_command_done(op_index)
                if asyncio.iscoroutine(result):
                    await result
        try:
            await session.run(encoded.text)
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
        await self._require_session().cancel(emergency)

    async def set_hold(self, hold: bool = True) -> None:
        await self._require_session().set_hold(hold)

    def can_home(self, axis: Axis | None = None) -> bool:
        """The xTool firmware only offers an all-axis homing cycle."""
        return axis is None

    async def home(self, axes: Axis | None = None) -> None:
        if axes is not None:
            raise DeviceConnectionError(
                _("This device only supports homing all axes.")
            )
        await self._require_session().home()

    async def move_to(
        self,
        pos_x: float,
        pos_y: float,
        pos_z: float | None = None,
        speed: float | None = None,
    ) -> None:
        session = self._require_session()
        z = self._to_machine_length(pos_z) if pos_z is not None else None
        await session.move_to(
            self._to_machine_length(float(pos_x)),
            self._to_machine_length(float(pos_y)),
            z,
            None if speed is None else self._to_machine_speed(speed),
        )

    def can_jog(self, axis: Axis | None = None) -> bool:
        """Jogging works in relative moves for all axes."""
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
        dialect = self.dialect
        cmd = dialect.tool_change.format(tool_number=tool_number)
        await self._require_session().execute_command(cmd)

    async def clear_alarm(self) -> None:
        # The firmware clears its safety alarms on its own; there is
        # no remote alarm-reset command, so only the locally mirrored
        # error state is cleared.
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

    async def detect_unit_system(self) -> UnitSystem | None:
        """The xTool dialect works in millimeters."""
        return UnitSystem.METRIC

    async def read_settings(self) -> None:
        logger.warning("xTool machines do not expose readable settings.")

    async def write_setting(self, key: str, value: Any) -> None:
        raise DeviceConnectionError(
            _("xTool machines do not support writing settings.")
        )

    @property
    def supported_wcs(self) -> list[str]:
        """The firmware has no work coordinate offset support."""
        return []

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None:
        raise DeviceConnectionError(
            _("xTool machines do not support work coordinate offsets.")
        )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        return {}

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Pos | None:
        raise DeviceConnectionError(_("This device does not support probing."))

    def get_error(self, error_code: str) -> DeviceError | None:
        return None

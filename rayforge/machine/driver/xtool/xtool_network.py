import asyncio
import logging
from collections.abc import Awaitable, Callable
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from raydriver.xtool import XtoolSession, query_machine_type

from ....context import RayforgeContext
from ....core.varset import HostnameVar, PortVar, VarSet
from ....core.varset.hostnamevar import is_valid_hostname_or_ip
from ....pipeline.encoder.base import EncodedOutput
from ....shared.units.system import UnitSystem
from ...transport import TransportStatus
from ..driver import (
    Axis,
    DeviceConnectionError,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    Pos,
)
from ..session_driver import SessionDriverBase

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from ....core.doc import Doc
    from ...device.profile import DeviceProfile

logger = logging.getLogger(__name__)

# Default network ports of xTool D-series machines (proprietary WiFi
# API).
DEFAULT_HTTP_PORT = 8080
DEFAULT_WS_PORT = 8081


class XtoolNetworkDriver(SessionDriverBase):
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
        machine_type = await query_machine_type(host, port)
        if machine_type is None and port != DEFAULT_HTTP_PORT:
            machine_type = await query_machine_type(host, DEFAULT_HTTP_PORT)
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

    async def read_parser_state(self) -> str | None:
        """The xTool API exposes no G-code parser state."""
        return None

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

    async def cancel(self, emergency: bool = False) -> None:
        await self._require_session().cancel(emergency)

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

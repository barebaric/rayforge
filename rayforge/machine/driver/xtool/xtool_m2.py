import logging
from collections.abc import Awaitable, Callable, Sequence
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from raydriver.core import SessionEvent
from raydriver.xtool.m2 import M2Session

from ....context import RayforgeContext
from ....core.varset import BoolVar, HostnameVar, PortVar, VarSet
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
    FrameCorner,
    Pos,
)
from ..session_state import from_session_state

if TYPE_CHECKING:
    from raydriver.core.types import DeviceState as SessionDeviceState
    from raygeo.ops import Ops

    from ....core.doc import Doc
    from ...models.laser import Laser
    from ...models.machine import Machine

logger = logging.getLogger(__name__)

# Default WS-V2 TLS port of the xTool M2.
DEFAULT_M2_PORT = 28900


class XtoolM2Driver(Driver):
    """
    Driver for the xTool M2 over its WS-V2 TLS service, with the
    entire protocol stack running in Rust via the ``raydriver``
    package.

    The M2 session supports state monitoring, motion and control of
    a job that is already staged on the machine — G-code job upload
    is not implemented, so :meth:`run` and :meth:`run_raw` raise a
    descriptive error instead of pretending to stream a job.  Use
    :meth:`start_staged_job` to start a job staged by xTool Studio.
    """

    label = _("xTool M2 (Wi-Fi)")
    subtitle = _("xTool M2 monitor and control (no job streaming)")
    maturity = DriverMaturity.EXPERIMENTAL
    uses_gcode = False
    supports_settings = False
    reports_granular_progress = False
    supports_probing = False
    supports_unit_detection = True
    DISCOVERY = None

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self._session: M2Session | None = None

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

    def _require_session(self) -> M2Session:
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
                    key="port",
                    label=_("Port"),
                    description=_("The WS-V2 port of the device"),
                    default=DEFAULT_M2_PORT,
                ),
                BoolVar(
                    key="tls_allow_invalid_certs",
                    label=_("Allow invalid TLS certificates"),
                    description=_(
                        "Skip certificate validation when connecting"
                        " to the machine. Only enable this if the"
                        " machine uses a self-signed certificate."
                    ),
                    value=False,
                ),
            ]
        )

    @classmethod
    def create_encoder(cls, machine: "Machine") -> "OpsEncoder":
        """Returns a GcodeEncoder configured for the machine's dialect.

        The M2 session cannot stream G-code jobs, so the encoder is
        currently unused; it keeps the frame() fallback working once
        job upload is implemented."""
        assert machine.dialect is not None
        return GcodeEncoder(machine.dialect)

    def _setup_implementation(self, **kwargs: Any) -> None:
        host = cast(str, kwargs.get("host", ""))
        if not host:
            raise DriverSetupError(_("Hostname must be configured."))
        port = cast(int, kwargs.get("port", DEFAULT_M2_PORT))

        config = {
            "host": host,
            "port": int(port),
            "tls_allow_invalid_certs": bool(
                kwargs.get("tls_allow_invalid_certs", False)
            ),
        }
        self._session = M2Session(
            config=config,
            event_callback=self._on_session_event,
        )

    def _on_session_event(self, event: SessionEvent) -> None:
        """Re-emit Rust session events as Rayforge blinker signals."""
        match event:
            case SessionEvent.StateChanged():
                self._on_state_changed(event.state)
            case SessionEvent.ConnectionStatusChanged():
                self._on_connection_status_changed(event.status, event.message)
            case SessionEvent.JobFinished():
                self.job_finished.send(self)
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

    async def run(
        self,
        encoded: EncodedOutput,
        doc: "Doc",
        ops: "Ops",
        on_command_done: Callable[[int], None | Awaitable[None]] | None = None,
    ) -> None:
        raise DeviceConnectionError(
            _(
                "The xTool M2 cannot receive G-code jobs over its"
                " network API yet. Stage the job in xTool Studio and"
                " start it from there."
            )
        )

    async def run_raw(self, machine_code: str) -> None:
        raise DeviceConnectionError(
            _(
                "The xTool M2 cannot receive G-code jobs over its"
                " network API yet. Stage the job in xTool Studio and"
                " start it from there."
            )
        )

    async def start_staged_job(self) -> None:
        """Starts a job already staged on the device, e.g. one sent
        by xTool Studio."""
        await self._require_session().start()

    async def frame(
        self,
        corners: Sequence[FrameCorner],
        speed_mm_per_min: float,
        doc: "Doc",
        repeat_count: int = 1,
        corner_pause_s: float = 0.0,
        power_fraction: float = 0.0,
        on_command_done: Callable[[int], None | Awaitable[None]] | None = None,
    ) -> None:
        raise DeviceConnectionError(
            _(
                "The xTool M2 cannot receive G-code jobs over its"
                " network API yet. Stage the job in xTool Studio and"
                " start it from there."
            )
        )

    async def cancel(self, emergency: bool = False) -> None:
        await self._require_session().cancel()

    async def set_hold(self, hold: bool = True) -> None:
        await self._require_session().set_hold(hold)

    def can_home(self, axis: Axis | None = None) -> bool:
        """The M2 homes all axes at once."""
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
        if pos_z is not None:
            raise DeviceConnectionError(
                _("This device does not support Z moves.")
            )
        if speed is None:
            speed = self._machine.max_travel_speed
        await self._require_session().move_to(
            self._to_machine_length(float(pos_x)),
            self._to_machine_length(float(pos_y)),
            self._to_machine_speed(speed),
        )

    def can_jog(self, axis: Axis | None = None) -> bool:
        """Jogging works for the X, Y and Z axes."""
        return True

    async def jog(self, speed: int, **deltas: float) -> None:
        await self._require_session().jog(
            dx=self._to_machine_length(deltas.get("x", 0.0)),
            dy=self._to_machine_length(deltas.get("y", 0.0)),
            dz=self._to_machine_length(deltas.get("z", 0.0)),
            speed=self._to_machine_speed(speed),
        )

    async def select_tool(self, tool_number: int) -> None:
        raise DeviceConnectionError(
            _("This device does not support tool changes.")
        )

    async def clear_alarm(self) -> None:
        # The M2 reports error states that clear on the device; only
        # the locally mirrored error state is reset here.
        self.state.error = None
        self.state_changed.send(self, state=self.state)

    async def set_power(self, head: "Laser", percent: float) -> None:
        raise DeviceConnectionError(
            _("This device does not support remote laser power control.")
        )

    async def set_focus_power(self, head: "Laser", percent: float) -> None:
        raise DeviceConnectionError(
            _("This device does not support remote laser power control.")
        )

    async def detect_unit_system(self) -> UnitSystem | None:
        """The M2 reports positions in millimeters."""
        return UnitSystem.METRIC

    def get_setting_vars(self) -> list["VarSet"]:
        return []

    async def read_settings(self) -> None:
        logger.warning("The xTool M2 does not expose readable settings.")

    async def write_setting(self, key: str, value: Any) -> None:
        raise DeviceConnectionError(
            _("The xTool M2 does not support writing settings.")
        )

    @property
    def supported_wcs(self) -> list[str]:
        """The M2 has no work coordinate offset support."""
        return []

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None:
        raise DeviceConnectionError(
            _("The xTool M2 does not support work coordinate offsets.")
        )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        return {}

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Pos | None:
        raise DeviceConnectionError(_("This device does not support probing."))

    def get_error(self, error_code: str) -> DeviceError | None:
        return None

import logging
from collections.abc import Awaitable, Callable, Sequence
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, cast

import raygeo.driver.xtool_m2

from ....core.varset import BoolVar, HostnameVar, PortVar, VarSet
from ....core.varset.hostnamevar import is_valid_hostname_or_ip
from ....pipeline.artifact.encoded import EncodedOutput
from ....shared.units.system import UnitSystem
from ..driver import (
    Axis,
    DeviceConnectionError,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    FrameCorner,
    Pos,
)
from ..native import NativeDriver

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from ....core.doc import Doc
    from ...models.laser import Laser

logger = logging.getLogger(__name__)

# Default WS-V2 TLS port of the xTool M2.
DEFAULT_M2_PORT = 28900


class XtoolM2Driver(NativeDriver):
    """
    Driver for the xTool M2 over its WS-V2 TLS service, with the
    entire protocol stack running in Rust via the ``raygeo``
    package.

    The M2 driver supports state monitoring, motion and control of a
    job that is already staged on the machine — G-code job upload is
    not implemented, so :meth:`run` and :meth:`run_raw` raise a
    descriptive error instead of pretending to stream a job.  Use
    :meth:`start_staged_job` to start a job staged by xTool Studio.
    """

    NATIVE = raygeo.driver.xtool_m2.XtoolM2Driver

    label = _("xTool M2 (Wi-Fi)")
    subtitle = _("xTool M2 monitor and control (no job streaming)")
    maturity = DriverMaturity.EXPERIMENTAL
    uses_gcode = False
    supports_settings = False
    reports_granular_progress = False
    supports_probing = False
    supports_unit_detection = True
    DISCOVERY = None

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

    def build_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        host = cast(str, kwargs.get("host", ""))
        if not host:
            raise DriverSetupError(_("Hostname must be configured."))
        port = cast(int, kwargs.get("port", DEFAULT_M2_PORT))

        return {
            "host": host,
            "port": int(port),
            "tls_allow_invalid_certs": bool(
                kwargs.get("tls_allow_invalid_certs", False)
            ),
        }

    def _create_session(self, config: dict[str, Any]) -> Any:
        return raygeo.driver.xtool_m2.XtoolM2Driver(
            config=config,
            event_callback=self._on_session_event,
        )

    def refresh_dialect(self) -> None:
        """The M2 driver takes no dialect templates."""

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

    async def read_parser_state(self) -> str | None:
        """The M2 API exposes no G-code parser state."""
        return None

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
            self._to_machine_speed(speed),
            self._to_machine_length(float(pos_x)),
            self._to_machine_length(float(pos_y)),
            None,
        )

    async def jog(self, speed: int, **deltas: float) -> None:
        await self._require_session().jog(
            self._to_machine_speed(speed),
            [
                ("x", self._to_machine_length(deltas.get("x", 0.0))),
                ("y", self._to_machine_length(deltas.get("y", 0.0))),
                ("z", self._to_machine_length(deltas.get("z", 0.0))),
            ],
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

    async def write_setting(self, key: str, value: Any) -> None:
        raise DeviceConnectionError(
            _("The xTool M2 does not support writing settings.")
        )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        return {}

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

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Pos | None:
        raise DeviceConnectionError(_("This device does not support probing."))

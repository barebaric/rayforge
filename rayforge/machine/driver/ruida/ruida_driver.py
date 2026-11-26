"""Native driver binding for Ruida laser controllers.

The protocol stack, job compiler and Ops encoder run in Rust via
``raydriver.ruida``. This binding supplies the connection form,
presentation, machine-speed defaults and the Ruida-specific WCS and
homing policy.
"""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, cast

import raydriver.ruida
from raydriver.encode import RuidaSpec

from rayforge.context import RayforgeContext
from rayforge.core.varset import (
    BoolVar,
    FloatVar,
    HostnameVar,
    LabeledChoiceVar,
    SerialPortVar,
    Var,
    VarSet,
)
from rayforge.machine.driver.driver import (
    Axis,
    DeviceConnectionError,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    FrameCorner,
    Pos,
    PWMParams,
    ZHomingMode,
)
from rayforge.machine.driver.native import NativeDriver
from rayforge.machine.models.laser import LaserHead
from rayforge.pipeline.artifact.encoded import MachineCodeOpMap

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from rayforge.core.doc import Doc
    from rayforge.machine.models.head import Head
    from rayforge.machine.models.laser import Laser
    from rayforge.machine.models.machine import Machine
    from rayforge.pipeline.artifact.encoded import EncodedOutput

logger = logging.getLogger(__name__)

DEFAULT_MOVE_TO_JOG_SPEED_MM_S = 600.0

CONNECTION_AUTO = "auto"
CONNECTION_NETWORK = "network"
CONNECTION_USB = "usb"

PROTOCOL_UDP = "udp"
PROTOCOL_TCP = "tcp"

DEFAULT_MAX_CUT_SPEED_MMPM = 24000
DEFAULT_MAX_TRAVEL_SPEED_MMPM = 36000

_UNCONFIGURED_MAX_CUT_SPEED_MMPM = 1000
_UNCONFIGURED_MAX_TRAVEL_SPEED_MMPM = 3000

DEFAULT_POWER_FLOOR = 8.0
DEFAULT_IMAGE_POWER_BIAS = 8.0


def _resolve_protocol(config: dict[str, object]) -> str:
    if not config.get("tcp_proto", False):
        return PROTOCOL_UDP
    mode = config.get("connection", CONNECTION_AUTO)
    if mode in (CONNECTION_NETWORK, CONNECTION_AUTO):
        return PROTOCOL_TCP
    return PROTOCOL_UDP


class RuidaUsbDeviceVar(SerialPortVar):
    """SerialPortVar whose value is optional (network mode needs no USB)."""

    def __init__(
        self,
        key: str,
        label: str,
        description: str | None = None,
        default: str | None = None,
        value: str | None = None,
        required_when: Callable[[dict[str, object]], bool] | None = None,
        *,
        visible_when: Callable[[dict[str, object]], bool] | None = None,
    ):
        super().__init__(
            key=key,
            label=label,
            description=description,
            default=default,
            value=value,
            visible_when=visible_when,
        )
        self.validator = None
        self.optional = True
        self.required_when = required_when


class RuidaDriver(NativeDriver):
    """Driver for Ruida laser controllers over USB, UDP or TCP.

    The native Rust stack handles the rpascript codec, the transport,
    the status monitor, the GlueScript compiler and the Ops encoder.
    """

    NATIVE = raydriver.ruida.RuidaDriver

    label = _("Ruida (Raydriver)")
    subtitle = _("Connect via Ruida Protocol")
    supports_settings = False
    reports_granular_progress = False
    uses_gcode = False
    maturity = DriverMaturity.UNTESTED
    supports_probing = False
    native_overscan = True
    supports_multi_depth_raster = False
    z_homing_modes = (ZHomingMode.PROBE, ZHomingMode.SWITCH)

    def __init__(self, context: RayforgeContext, machine: "Machine") -> None:
        super().__init__(context, machine)
        self._jog_speed_mm_s: float | None = None
        self._jog_z_speed_mm_s: float = 100.0
        self._jog_u_speed_mm_s: float = 100.0
        self._selected_wcs: str = self._initial_wcs(machine)

    def _initial_wcs(self, machine: "Machine") -> str:
        wcs = machine.active_wcs
        return wcs if wcs in self.supported_wcs else "MACHINE"

    # -- properties -----------------------------------------------------

    @property
    def machine_space_wcs(self) -> str:
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Ruida Coordinates")

    @property
    def supported_wcs(self) -> list[str]:
        return ["MACHINE", "ANCHOR", "CURRENT", "SET_POINT"]

    # -- setup ----------------------------------------------------------

    @staticmethod
    def _apply_connection_mode(
        config: dict[str, object],
    ) -> dict[str, object]:
        mode = config.get("connection", CONNECTION_AUTO)
        config = dict(config)
        if mode == CONNECTION_NETWORK:
            config.pop("usb_device", None)
        elif mode == CONNECTION_USB:
            config.pop("udp_host", None)
        return config

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        if kwargs.get("tui", False):
            raise DriverPrecheckError(
                _(
                    "TUI RPC mode is no longer supported. Use a direct "
                    "USB or network connection."
                )
            )
        connection = cls._apply_connection_mode(kwargs)
        mode = connection.get("connection", CONNECTION_AUTO)
        udp_host = connection.get("udp_host", "")
        usb_device = connection.get("usb_device", "")
        if not udp_host and not usb_device:
            if mode == CONNECTION_NETWORK:
                message = _("A Hostname must be configured.")
            elif mode == CONNECTION_USB:
                message = _("A USB device must be configured.")
            else:
                message = _(
                    "At least one of 'Hostname' or 'USB device' "
                    "must be configured."
                )
            raise DriverPrecheckError(message)

    @classmethod
    def get_setup_vars(cls) -> VarSet:
        return VarSet(
            vars=[
                LabeledChoiceVar(
                    key="connection",
                    label=_("Connection"),
                    choices=[
                        (_("Auto (USB preferred)"), CONNECTION_AUTO),
                        (_("Network"), CONNECTION_NETWORK),
                        (_("USB"), CONNECTION_USB),
                    ],
                    default=CONNECTION_AUTO,
                    allow_none=False,
                    description=_(
                        "Which transport to use. Automatic opens USB "
                        "when available and falls back to the network."
                    ),
                ),
                BoolVar(
                    key="tcp_proto",
                    label=_("TCP protocol"),
                    description=_(
                        "Use TCP instead of UDP for the network "
                        "connection. Needed by controllers that do not "
                        "respond over UDP."
                    ),
                    default=False,
                    visible_when=lambda v: (
                        v.get("connection", CONNECTION_AUTO)
                        in (CONNECTION_NETWORK, CONNECTION_AUTO)
                    ),
                ),
                HostnameVar(
                    key="udp_host",
                    label=_("Hostname or IP address"),
                    description=_(
                        "The IP address or hostname of the Ruida controller"
                    ),
                    visible_when=lambda v: (
                        v.get("connection", CONNECTION_AUTO)
                        in (CONNECTION_NETWORK, CONNECTION_AUTO)
                    ),
                ),
                RuidaUsbDeviceVar(
                    key="usb_device",
                    label=_("USB"),
                    description=_(
                        "USB device path or VID:PID (e.g. 0403:6001)"
                    ),
                    required_when=lambda v: (
                        v.get("connection", CONNECTION_AUTO) == CONNECTION_USB
                    ),
                    visible_when=lambda v: (
                        v.get("connection", CONNECTION_AUTO)
                        in (CONNECTION_USB, CONNECTION_AUTO)
                    ),
                ),
                Var(
                    key="magic_number",
                    label=_("Magic"),
                    var_type=str,
                    description=_(
                        "Controller magic number in hex. The default "
                        "0x88 works for most controllers."
                    ),
                    default="0x88",
                ),
                BoolVar(
                    key="power_scaling_enabled",
                    label=_("Power scaling"),
                    description=_(
                        "When enabled, power_range() raises the emitted "
                        "minimum as the layer's cut speed decreases "
                        "(effective-min power scaling); when disabled, the "
                        "resolved minimum is emitted unchanged."
                    ),
                    default=True,
                ),
                BoolVar(
                    key="power_floor_enabled",
                    label=_("Enable VECTOR power floor"),
                    description=_(
                        "When disabled, the emitted power range always "
                        "has min == max (constant power). When enabled, "
                        "the VECTOR power floor value below acts as the "
                        "minimum power for VECTOR cut compensation."
                    ),
                    default=False,
                ),
                FloatVar(
                    key="power_floor",
                    label=_("VECTOR power floor"),
                    description=_(
                        "Minimum power percentage for VECTOR cut "
                        "compensation (e.g. 8 = 8%). Applies only when "
                        "the VECTOR power floor toggle is enabled."
                    ),
                    default=DEFAULT_POWER_FLOOR,
                    min_val=0.0,
                    max_val=100.0,
                    digits=1,
                ),
                FloatVar(
                    key="image_power_bias",
                    label=_("IMAGE power bias"),
                    description=_(
                        "Power bias added to IMAGE/raster "
                        "scan-line power (SET_POWER), % of max "
                        "(e.g. 8 = 8%)."
                    ),
                    default=DEFAULT_IMAGE_POWER_BIAS,
                    min_val=0.0,
                    max_val=100.0,
                    digits=1,
                ),
            ]
        )

    @staticmethod
    def _parse_magic_number(raw_magic: object) -> int:
        if raw_magic is None:
            return 0x88
        magic_str = str(raw_magic).strip()
        if not magic_str:
            return 0x88
        try:
            magic = int(magic_str, 0)
        except (TypeError, ValueError):
            raise DriverSetupError(
                "Magic number must be a valid hex value (e.g. 0x88 or 88)"
            ) from None
        if not 0x00 <= magic <= 0xFF:
            raise DriverSetupError(
                "Magic number must be between 0x00 and 0xFF"
            )
        return magic

    def build_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.precheck(**kwargs)
        connection = self._apply_connection_mode(kwargs)
        host = cast(str, connection.get("udp_host", "") or "")
        usb_device = cast(str, connection.get("usb_device", "") or "")
        magic = self._parse_magic_number(kwargs.get("magic_number"))
        config = {
            "host": host,
            "usb_device": usb_device,
            "protocol": _resolve_protocol(connection),
            "magic_number": f"0x{magic:02X}",
            "protect": True,
        }
        for key in (
            "port",
            "query_interval",
            "ping_retries",
            "ping_delay",
        ):
            if key in kwargs:
                config[key] = kwargs[key]
        return config

    @classmethod
    def create_encoder_spec(cls, machine: "Machine", doc: "Doc") -> RuidaSpec:
        from rayforge.machine.driver.ruida.ruida_encoder import (
            build_ruida_spec,
        )

        return build_ruida_spec(machine, doc)

    def _create_session(
        self, config: dict[str, Any]
    ) -> raydriver.ruida.RuidaDriver:
        return raydriver.ruida.RuidaDriver(
            config=config,
            event_callback=self._on_session_event,
        )

    def _update_session_dialect(self) -> None:
        """Ruida jobs are rpascript, not dialect-templated G-code."""

    def refresh_dialect(self) -> None:
        """Ruida jobs are rpascript, not dialect-templated G-code."""

    def _seed_machine_speed_defaults(self) -> None:
        if self._machine.max_cut_speed == _UNCONFIGURED_MAX_CUT_SPEED_MMPM:
            self._machine.set_max_cut_speed(DEFAULT_MAX_CUT_SPEED_MMPM)
        if (
            self._machine.max_travel_speed
            == _UNCONFIGURED_MAX_TRAVEL_SPEED_MMPM
        ):
            self._machine.set_max_travel_speed(DEFAULT_MAX_TRAVEL_SPEED_MMPM)

    def _setup_implementation(self, **kwargs: Any) -> None:
        self._seed_machine_speed_defaults()
        super()._setup_implementation(**kwargs)

    # -- protect --------------------------------------------------------

    async def get_protect(self) -> bool:
        session = cast(raydriver.ruida.RuidaDriver, self.session)
        if session is None:
            return False
        return bool(session.protect_enabled)

    async def set_protect(self, enabled: bool) -> None:
        session = cast(raydriver.ruida.RuidaDriver, self.session)
        if session is None:
            return
        session.set_protect(bool(enabled))

    # -- jobs -----------------------------------------------------------

    @staticmethod
    def _op_line_map(op_map: MachineCodeOpMap | None) -> dict[int, int]:
        result = {}
        if op_map is None:
            return result
        for line_idx in range(op_map.line_count):
            op_index = op_map.op_for_line(line_idx)
            if op_index is not None:
                result[line_idx] = op_index
        return result

    async def run(
        self,
        encoded: "EncodedOutput",
        doc: "Doc",
        ops: "Ops",
        on_command_done: Callable[[int], None | Awaitable[None]] | None = None,
    ) -> None:
        op_map = encoded.op_map
        if on_command_done is not None:
            count = op_map.op_count if op_map else 0
            for op_index in range(count):
                result = on_command_done(op_index)
                if inspect.isawaitable(result):
                    await result

        payload = encoded.driver_data.get("payload")
        if isinstance(payload, (bytes, bytearray)):
            script = bytes(payload).decode("utf-8")
        else:
            script = encoded.text
        if not script.strip():
            logger.debug("No rpascript commands to execute")
        session = self._require_session()
        try:
            await session.run(script, self._op_line_map(op_map), [], None)
        except DeviceConnectionError as error:
            logger.warning(
                "Job terminated due to device error: %s. "
                "Connection remains active.",
                error,
            )
        except Exception:
            logger.exception("Job terminated with unexpected error")

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
        for _repeat in range(max(1, repeat_count)):
            for pos_x, pos_y, _extra in corners:
                await self.move_to(pos_x, pos_y, speed=speed_mm_per_min)
                if corner_pause_s > 0:
                    await asyncio.sleep(corner_pause_s)
        self.job_finished.send(self)

    # -- movement -------------------------------------------------------

    async def home(self, axes: Axis | None = None) -> None:
        session = self._require_session()
        xy_axes = Axis.X | Axis.Y
        if axes is None or (axes & xy_axes) == xy_axes:
            await session.home_xy()
        elif axes & Axis.X:
            await session.jog_x_to(0.0)
        elif axes & Axis.Y:
            await session.jog_y_to(0.0)
        if axes is not None and (axes & Axis.Z):
            if self.get_z_homing_mode() is ZHomingMode.SWITCH:
                await session.home_z()
            else:
                await session.focus_z()

    async def move_to(
        self,
        pos_x: float,
        pos_y: float,
        pos_z: float | None = None,
        speed: float | None = None,
    ) -> None:
        session = self._require_session()
        if speed is not None:
            speed_mm_s = speed / 60.0
        else:
            speed_mm_s = (
                self._jog_speed_mm_s
                if self._jog_speed_mm_s is not None
                else DEFAULT_MOVE_TO_JOG_SPEED_MM_S
            )
        self._jog_speed_mm_s = speed_mm_s
        session.set_jog_speeds(
            speed_mm_s,
            self._jog_z_speed_mm_s,
            self._jog_u_speed_mm_s,
        )
        await session.jog_xy_to(pos_x, pos_y)

    async def jog(self, speed: int, **deltas: float) -> None:
        speed_mm_s = speed / 60.0
        self._jog_speed_mm_s = speed_mm_s
        if not deltas:
            return
        if "z" in deltas:
            self._jog_z_speed_mm_s = speed_mm_s
        if "u" in deltas:
            self._jog_u_speed_mm_s = speed_mm_s
        session = self._require_session()
        session.set_jog_speeds(
            speed_mm_s,
            self._jog_z_speed_mm_s,
            self._jog_u_speed_mm_s,
        )
        if "x" in deltas or "y" in deltas:
            await session.jog_xy_rel(deltas.get("x"), deltas.get("y"))
        if "z" in deltas:
            await session.execute(f"SPEED_AXIS {self._jog_z_speed_mm_s}")
            await session.execute(f"JOG_Z Rel:CURRENT Z={deltas['z']}mm")
        if "u" in deltas:
            await session.execute(f"SPEED_AXIS {self._jog_u_speed_mm_s}")
            await session.execute(f"JOG_U Rel:CURRENT U={deltas['u']}mm")

    # -- job control ----------------------------------------------------

    async def set_hold(self, hold: bool = True) -> None:
        await self._require_session().set_hold(hold)

    async def cancel(self, emergency: bool = False) -> None:
        await self._require_session().cancel(emergency)

    async def clear_alarm(self) -> None:
        await self._require_session().clear_alarm()

    # -- power / probing ------------------------------------------------

    async def set_power(self, head: "Laser", percent: float) -> None:
        logger.warning(
            _(
                "set_power is not supported outside of a job with a "
                "Ruida controller — the correct handling is to be "
                "determined; no command was sent."
            )
        )

    async def set_focus_power(self, head: "Laser", percent: float) -> None:
        logger.warning(
            _(
                "set_focus_power is not supported outside of a job with "
                "a Ruida controller — the correct handling is to be "
                "determined; no command was sent."
            )
        )

    async def run_probe_cycle(
        self, axis: Axis, target: float, feed_rate: int
    ) -> Pos | None:
        self.probe_status_changed.send(
            self, message=_("Probe not supported by the Ruida driver")
        )
        return None

    # -- WCS ------------------------------------------------------------

    async def select_wcs(self, wcs: str) -> None:
        if wcs not in self.supported_wcs:
            raise ValueError(
                f"Unsupported WCS {wcs!r} — valid names: "
                f"{', '.join(self.supported_wcs)}"
            )
        self._selected_wcs = wcs

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None:
        raise NotImplementedError(
            _(
                "set_wcs_offset is not supported: protect mode blocks "
                "SET_SETTING, and the feature was dropped by user "
                "decision. Use select_wcs to pick a reference point."
            )
        )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        offsets = await super().read_wcs_offsets()
        self.wcs_updated.send(self, offsets=offsets)
        return offsets

    async def read_parser_state(self) -> str | None:
        return self._selected_wcs

    # -- settings -------------------------------------------------------

    def get_setting_vars(self) -> list[VarSet]:
        return [VarSet(title=_("No settings"))]

    # -- capabilities ---------------------------------------------------

    def can_jog(self, axis: Axis | None = None) -> bool:
        return True

    def supports_pwm(self, head: "Head") -> bool:
        return isinstance(head, LaserHead) and head.laser_type.supports_pwm

    def get_pwm_params(self, head: "Head") -> PWMParams | None:
        if not isinstance(head, LaserHead) or not self.supports_pwm(head):
            return None
        return PWMParams(
            frequency=head.pwm_frequency,
            max_frequency=head.max_pwm_frequency,
            pulse_width=head.pulse_width,
            min_pulse_width=head.min_pulse_width,
            max_pulse_width=head.max_pulse_width,
        )

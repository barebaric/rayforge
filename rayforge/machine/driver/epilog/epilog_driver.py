"""Epilog laser driver backed by the raydriver LPD session.

Epilog laser cutters are print devices: Rayforge encodes the job as
an Epilog print job (the reverse-engineered reference format is PCL5
carrying HPGL vector output) and uploads it to the machine's network
print queue, and the operator starts it at the control panel. The
machine exposes no status channel and cannot be homed, jogged or
moved remotely.

The encoder itself lives in raygeo (``raygeo.ops.convert.EpilogSpec``)
and is selected by :class:`~rayforge.pipeline.intent_builder.IntentBuilder`.
That encoder returns both the printable listing Rayforge shows and
the exact byte stream this driver uploads.
"""

import logging
from collections.abc import Awaitable, Callable
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, cast

from raydriver.epilog import EpilogSession

from ....context import RayforgeContext
from ....core.varset import (
    ChoiceVar,
    HostnameVar,
    IntVar,
    PortVar,
    SpeedVar,
    TextAreaVar,
    VarSet,
)
from ....core.varset.hostnamevar import hostname_validator
from ....pipeline.encoder.base import EncodedOutput
from ...models.origin import Origin
from ..driver import (
    Axis,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    Pos,
)
from ..session_driver import SessionDriverBase

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from ....core.doc import Doc
    from ...models.laser import Laser
    from ...models.machine import Machine

logger = logging.getLogger(__name__)

#: LPD port of an Epilog printer.
DEFAULT_PORT = 515

#: Print queue name Epilog's setup instructions use.
DEFAULT_QUEUE = "Laser"

#: Supported raster and vector resolutions in dots per inch.
RESOLUTIONS = ("100", "200", "250", "400", "500", "1000")
DEFAULT_RESOLUTION = "500"

#: Cutting speed the machine reaches at 100 %, in mm/min.
#:
#: The machine's speed control is a percentage of its own reference,
#: which is a property of the model: the reference driver calibrates a
#: Helix at 21.3 mm/s (1278 mm/min). It is not the machine's maximum
#: travel speed. Zing-specific calibration is still owed; this default
#: matches the 600 mm/min = 10 % figure the profile notes used.
DEFAULT_SPEED_REFERENCE = 6000

DEFAULT_FREQUENCY = 1000
MIN_FREQUENCY = 10
MAX_FREQUENCY = 5000


class EpilogZingDriver(SessionDriverBase):
    """
    Driver for Epilog laser cutters (Zing 16, Zing 24/6030, …) over
    the machine's LPD print queue.

    The entire transport stack runs in Rust via ``raydriver``. This
    shell only translates between the Rayforge ``Driver`` interface
    and the session, and uploads the encoder's byte stream.
    """

    label = _("Epilog")
    subtitle = _("Upload print jobs to an Epilog laser over Ethernet")
    maturity = DriverMaturity.UNTESTED
    supports_settings = False
    reports_granular_progress = False
    uses_gcode = False
    supports_probing = False
    supports_unit_detection = False
    supports_multi_depth_raster = False
    DISCOVERY = None
    # The print job always addresses a top-left page: Y grows towards
    # the front of the machine and the job language cannot express
    # another corner. The machine's origin is constrained to match.
    supported_origins = (Origin.TOP_LEFT,)

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self._machine_wcs = "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Epilog Bed Coordinates")

    @property
    def supported_wcs(self) -> list[str]:
        return ["MACHINE"]

    @classmethod
    def supports_travel_speed(cls, dialect: Any) -> bool:
        return False

    @property
    def session(self) -> EpilogSession | None:
        """The underlying raydriver session, if set up."""
        return self._session

    def _require_session(self) -> EpilogSession:
        if self._session is None:
            raise DriverSetupError(
                _("Driver is not set up. Check the connection settings.")
            )
        return self._session

    # -- setup ---------------------------------------------------------

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        try:
            hostname_validator(kwargs.get("host"))
            port = kwargs.get("port", DEFAULT_PORT)
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError(_("Port must be between 1 and 65535"))
            resolution = str(kwargs.get("resolution", DEFAULT_RESOLUTION))
            if resolution not in RESOLUTIONS:
                raise ValueError(
                    _("Unsupported resolution: {resolution}").format(
                        resolution=resolution
                    )
                )
            frequency = kwargs.get("frequency", DEFAULT_FREQUENCY)
            if not MIN_FREQUENCY <= frequency <= MAX_FREQUENCY:
                raise ValueError(
                    _("Frequency must be between {min} and {max} Hz").format(
                        min=MIN_FREQUENCY, max=MAX_FREQUENCY
                    )
                )
        except (ValueError, TypeError) as exc:
            raise DriverPrecheckError(str(exc)) from exc

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            title=_("Epilog"),
            description=_(
                "Jobs are uploaded to the laser and started at its "
                "control panel. Focus and air assist are manual."
            ),
            vars=[
                HostnameVar(
                    key="host",
                    label=_("Hostname / IP"),
                    description=_("The IP address or hostname of the laser"),
                ),
                PortVar(
                    key="port",
                    label=_("Port"),
                    description=_("The LPD print port of the laser"),
                    default=DEFAULT_PORT,
                ),
                SpeedVar(
                    key="speed_reference",
                    label=_("100 % Speed"),
                    description=_(
                        "The cutting speed the machine reaches at 100 %"
                    ),
                    default=DEFAULT_SPEED_REFERENCE,
                ),
                TextAreaVar(
                    key="queue",
                    label=_("Print Queue"),
                    description=_(
                        "The print queue name the laser expects. "
                        "Epilog's setup names it 'Laser'"
                    ),
                    default=DEFAULT_QUEUE,
                ),
                ChoiceVar(
                    key="resolution",
                    label=_("Resolution (DPI)"),
                    choices=list(RESOLUTIONS),
                    default=DEFAULT_RESOLUTION,
                    allow_none=False,
                ),
                IntVar(
                    key="frequency",
                    label=_("Vector Frequency"),
                    description=_("Epilog vector frequency setting (Hz)"),
                    default=DEFAULT_FREQUENCY,
                    min_val=MIN_FREQUENCY,
                    max_val=MAX_FREQUENCY,
                ),
            ],
        )

    def _setup_implementation(self, **kwargs: Any) -> None:
        try:
            self.precheck(**kwargs)
        except DriverPrecheckError as exc:
            raise DriverSetupError(str(exc)) from exc
        host = cast(str, kwargs["host"])
        port = cast(int, kwargs.get("port", DEFAULT_PORT))
        self._speed_reference = float(
            kwargs.get("speed_reference") or DEFAULT_SPEED_REFERENCE
        )
        self._session = EpilogSession(
            config={
                "host": host,
                "port": int(port),
                "queue": str(kwargs.get("queue") or DEFAULT_QUEUE),
            },
            event_callback=self._on_session_event,
        )

    def get_setting_vars(self) -> list[VarSet]:
        return []

    @property
    def speed_reference(self) -> float:
        """Cutting speed the machine reaches at 100 %, in mm/min."""
        return self._speed_reference

    # -- capabilities --------------------------------------------------

    def can_home(self, axis: Axis | None = None) -> bool:
        return False

    def can_jog(self, axis: Axis | None = None) -> bool:
        return False

    def can_frame(self) -> bool:
        """A frame is a beam-off outline job uploaded like any job."""
        return True

    async def select_tool(self, tool_number: int) -> None:
        if tool_number != self._machine.get_default_head().tool_number:
            raise NotImplementedError(
                _("Use the Epilog control panel for this operation")
            )

    async def run_raw(self, machine_code: str) -> None:
        raise NotImplementedError(
            _("Use the Epilog control panel for this operation")
        )

    async def set_hold(self, hold: bool = True) -> None:
        raise NotImplementedError(
            _("Use the Epilog control panel for this operation")
        )

    async def home(self, axes: Axis | None = None) -> None:
        raise NotImplementedError(
            _("Use the Epilog control panel for this operation")
        )

    async def move_to(
        self,
        pos_x: float,
        pos_y: float,
        pos_z: float | None = None,
        speed: float | None = None,
    ) -> None:
        raise NotImplementedError(
            _("Use the Epilog control panel for this operation")
        )

    async def read_settings(self) -> None:
        raise NotImplementedError(
            _("Use the Epilog control panel for this operation")
        )

    async def write_setting(self, key: str, value: Any) -> None:
        raise NotImplementedError(
            _("Use the Epilog control panel for this operation")
        )

    async def clear_alarm(self) -> None:
        raise NotImplementedError(
            _("Use the Epilog control panel for this operation")
        )

    async def set_power(self, head: "Laser", percent: float) -> None:
        raise NotImplementedError(
            _("Use the Epilog control panel for this operation")
        )

    async def set_focus_power(self, head: "Laser", percent: float) -> None:
        raise NotImplementedError(
            _("Use the Epilog control panel for this operation")
        )

    async def jog(self, speed: int, **deltas: float) -> None:
        raise NotImplementedError(
            _("Use the Epilog control panel for this operation")
        )

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None:
        raise NotImplementedError(
            _("Use the Epilog control panel for this operation")
        )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        return {"MACHINE": (0.0, 0.0, 0.0)}

    async def read_parser_state(self) -> str | None:
        return "MACHINE"

    async def select_wcs(self, wcs: str) -> None:
        if wcs != "MACHINE":
            raise NotImplementedError(
                _("Use the Epilog control panel for this operation")
            )

    async def run_probe_cycle(
        self, axis: Axis, max_travel: int, feed_rate: int
    ) -> Pos | None:
        raise NotImplementedError(
            _("Use the Epilog control panel for this operation")
        )

    # -- jobs ----------------------------------------------------------

    async def run(
        self,
        encoded: EncodedOutput,
        doc: "Doc",
        ops: "Ops",
        on_command_done: Callable[[int], None | Awaitable[None]] | None = None,
    ) -> None:
        """Upload the encoded job's byte stream.

        *on_command_done* is unused: the machine gives no progress
        feedback after accepting a job.
        """
        payload = encoded.driver_data.get("payload")
        if not payload:
            raise DriverSetupError(_("The job has no machine code to upload."))
        title = doc.name or "Rayforge"
        try:
            await self._require_session().upload(payload, title)
        except ConnectionError as exc:
            self._logger.warning(
                f"Job terminated due to device error: {exc}. "
                "Connection remains active."
            )
            raise
        except Exception as exc:
            self._logger.error(f"Epilog upload failed: {exc}")
            raise

    async def cancel(self, emergency: bool = False) -> None:
        await self._require_session().cancel(emergency)

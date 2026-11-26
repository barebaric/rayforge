"""Epilog laser driver backed by the raydriver LPD driver.

Epilog laser cutters are print devices: Rayforge encodes the job as
an Epilog print job (the reverse-engineered reference format is PCL5
carrying HPGL vector output) and uploads it to the machine's network
print queue, and the operator starts it at the control panel. The
machine exposes no status channel and cannot be homed, jogged or
moved remotely.

The encoder itself lives in raydriver (``raydriver.encode.EpilogSpec``)
and its inputs are built by this package's :mod:`.encoder`; the
pipeline only asks the driver for the spec. The Zing and Helix
families share the print language and differ only in the raster
resolutions they accept.
"""

from collections.abc import Awaitable, Callable
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

import raydriver.epilog

from ....context import RayforgeContext
from ....core.varset import (
    ChoiceVar,
    HostnameVar,
    IntVar,
    PortVar,
    SpeedVar,
    Var,
    VarSet,
)
from ....core.varset.hostnamevar import hostname_validator
from ....pipeline.artifact.encoded import EncodedOutput
from ...models.origin import Origin
from ..driver import (
    Axis,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    Pos,
)
from ..native import NativeDriver
from .encoder import (
    DEFAULT_FREQUENCY,
    DEFAULT_SPEED_REFERENCE,
    HELIX_DEFAULT_RESOLUTION,
    HELIX_RESOLUTIONS,
    MAX_FREQUENCY,
    MIN_FREQUENCY,
    ZING_RESOLUTIONS,
    build_epilog_spec,
)
from .encoder import (
    DEFAULT_RESOLUTION as ZING_DEFAULT_RESOLUTION,
)

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from ....core.doc import Doc
    from ...models.laser import Laser
    from ...models.machine import Machine

#: LPD port of an Epilog printer.
DEFAULT_PORT = 515

#: The receive-job operand is empty unless a print server needs a name.
DEFAULT_QUEUE = ""


def _panel_only() -> NotImplementedError:
    return NotImplementedError(
        _("Use the Epilog control panel for this operation")
    )


class EpilogLaserDriver(NativeDriver):
    """
    Shared implementation for Epilog laser cutters over the machine's
    LPD print queue.

    The entire transport stack runs in Rust via ``raydriver``. This
    shell only supplies the setup vars and uploads the encoder's byte
    stream.
    """

    NATIVE = raydriver.epilog.EpilogDriver

    label = _("Epilog")
    subtitle = _("Upload print jobs to an Epilog laser over Ethernet")
    maturity = DriverMaturity.EXPERIMENTAL
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

    #: Encoder model family (``zing`` or ``helix``).
    model = "zing"
    #: Raster resolutions this model accepts, in DPI.
    resolutions = ZING_RESOLUTIONS
    default_resolution = ZING_DEFAULT_RESOLUTION
    default_queue = DEFAULT_QUEUE

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

    @classmethod
    def create_encoder_spec(
        cls, machine: "Machine", doc: "Doc", ops: "Ops"
    ) -> Any:
        return build_epilog_spec(machine, doc, model=cls.model)

    # -- setup ---------------------------------------------------------

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        try:
            hostname_validator(kwargs.get("host"))
            port = kwargs.get("port", DEFAULT_PORT)
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError(_("Port must be between 1 and 65535"))
            resolution = str(kwargs.get("resolution", cls.default_resolution))
            if resolution not in cls.resolutions:
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
                Var(
                    key="queue",
                    label=_("Print Queue"),
                    description=_(
                        "The print queue name. Leave empty unless your "
                        "print server requires one; Epilog lasers usually "
                        "accept an empty queue"
                    ),
                    var_type=str,
                    default=cls.default_queue,
                    optional=True,
                ),
                ChoiceVar(
                    key="resolution",
                    label=_("Resolution (DPI)"),
                    choices=list(cls.resolutions),
                    default=cls.default_resolution,
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

    def build_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        host = kwargs.get("host")
        if not isinstance(host, str) or not host.strip():
            raise DriverSetupError(_("Host must be configured."))
        queue = kwargs.get("queue")
        if queue is None:
            queue = self.default_queue
        return {
            "host": host.strip(),
            "port": int(kwargs.get("port", DEFAULT_PORT)),
            "queue": str(queue),
        }

    def get_setting_vars(self) -> list[VarSet]:
        return []

    # -- capabilities --------------------------------------------------

    def can_home(self, axis: Axis | None = None) -> bool:
        return False

    def can_jog(self, axis: Axis | None = None) -> bool:
        return False

    def can_frame(self) -> bool:
        """A frame is a beam-off outline job uploaded like any job."""
        return True

    async def select_tool(self, tool_number: int) -> None:
        head = self._machine.get_default_head()
        if head is None or tool_number != head.tool_number:
            raise _panel_only()

    async def select_wcs(self, wcs: str) -> None:
        if wcs != "MACHINE":
            raise _panel_only()

    # -- panel-only operations -----------------------------------------

    async def run_raw(self, machine_code: str) -> None:
        raise _panel_only()

    async def set_hold(self, hold: bool = True) -> None:
        raise _panel_only()

    async def home(self, axes: Axis | None = None) -> None:
        raise _panel_only()

    async def move_to(
        self,
        pos_x: float,
        pos_y: float,
        pos_z: float | None = None,
        speed: float | None = None,
    ) -> None:
        raise _panel_only()

    async def read_settings(self) -> None:
        raise _panel_only()

    async def write_setting(self, key: str, value: Any) -> None:
        raise _panel_only()

    async def clear_alarm(self) -> None:
        raise _panel_only()

    async def set_power(self, head: "Laser", percent: float) -> None:
        raise _panel_only()

    async def set_focus_power(self, head: "Laser", percent: float) -> None:
        raise _panel_only()

    async def jog(self, speed: int, **deltas: float) -> None:
        raise _panel_only()

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float | None
    ) -> None:
        raise _panel_only()

    async def run_probe_cycle(
        self, axis: Axis, target: float, feed_rate: int
    ) -> Pos | None:
        raise _panel_only()

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
        title = (doc.name if doc else "") or "Rayforge"
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


class EpilogZingDriver(EpilogLaserDriver):
    """Driver for the Epilog Zing 16 and Zing 24/6030."""

    label = _("Epilog Zing")
    subtitle = _("Upload print jobs to an Epilog Zing over Ethernet")
    model = "zing"
    resolutions = ZING_RESOLUTIONS
    default_resolution = ZING_DEFAULT_RESOLUTION


class EpilogHelixDriver(EpilogLaserDriver):
    """Driver for the Epilog Helix family."""

    label = _("Epilog Helix")
    subtitle = _("Upload print jobs to an Epilog Helix over Ethernet")
    model = "helix"
    resolutions = HELIX_RESOLUTIONS
    default_resolution = HELIX_DEFAULT_RESOLUTION

import asyncio
import logging
from collections.abc import Awaitable
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

from blinker import Signal
from raygeo.driver.grbl.probe import GrblDeviceInfo, GrblFlavor
from raygeo.driver.marlin.probe import MarlinDeviceInfo
from raygeo.driver.octoprint.probe import OctoPrintDeviceInfo
from raygeo.driver.smoothie.probe import SmoothieDeviceInfo

from ....shared.units.system import UnitSystem
from ...models.dialect.grbl import GRBL_DIALECT
from ...models.dialect.grbl_dynamic import GRBL_DYNAMIC_DIALECT
from ...transport import TransportStatus
from ..driver import DeviceConnectionError

if TYPE_CHECKING:
    from ....context import RayforgeContext
    from ...device.profile import DeviceProfile
    from ...models.machine import Machine

logger = logging.getLogger(__name__)


@runtime_checkable
class _ProbeSession(Protocol):
    """The session surface the device census needs; the concrete
    census type depends on the session's protocol."""

    def probe_device(
        self,
    ) -> Awaitable[
        GrblDeviceInfo
        | MarlinDeviceInfo
        | SmoothieDeviceInfo
        | OctoPrintDeviceInfo
    ]: ...


@runtime_checkable
class _GrblProbeDriver(Protocol):
    """
    Protocol describing the interface ``probe_grbl_device`` needs
    from any Grbl driver.  Any driver implementing these methods
    can be probed — serial, telnet, network, or otherwise.
    """

    connection_status_changed: Signal

    @property
    def session(self) -> _ProbeSession | None: ...

    def __init__(
        self,
        context: "RayforgeContext",
        machine: "Machine",
    ) -> None: ...

    def setup(self, **kwargs: Any) -> None: ...

    async def connect(self) -> None: ...

    async def cleanup(self) -> None: ...


async def probe_grbl_device(
    driver_cls: type[_GrblProbeDriver],
    context: "RayforgeContext",
    **kwargs: Any,
) -> tuple["DeviceProfile", list[str]]:
    """
    Shared probe orchestration for all Grbl drivers.

    Creates a temporary machine + driver, connects, runs the device
    census over the driver's session, disconnects, and returns a
    ``(DeviceProfile, warnings)`` tuple.
    """
    from ...models.machine import Machine

    machine = Machine(context)
    driver = driver_cls(context, machine)
    driver.setup(**kwargs)

    connected = asyncio.Event()

    def _on_status(sender, status=None, message=None, **kw):
        if status == TransportStatus.CONNECTED:
            connected.set()

    driver.connection_status_changed.connect(_on_status)
    try:
        await driver.connect()
        # Bounds the whole connect attempt, including a possible
        # bootloader-delayed first handshake plus one retry cycle
        # (HANDSHAKE_TIMEOUT + reconnect sleep + HANDSHAKE_TIMEOUT).
        await asyncio.wait_for(connected.wait(), timeout=30.0)
        session = driver.session
        if session is None:
            raise DeviceConnectionError(_("Driver is not connected."))
        info = await session.probe_device()
        if not isinstance(info, GrblDeviceInfo):
            raise DeviceConnectionError(_("Device is not a GRBL device."))
    finally:
        driver.connection_status_changed.disconnect(_on_status)
        await driver.cleanup()
        context.dialect_mgr.dialects_changed.disconnect(
            machine._on_dialects_changed
        )

    profile, warnings = build_grbl_profile(info)
    profile.machine_config.driver = driver_cls.__name__
    profile.machine_config.driver_args = kwargs
    return profile, warnings


def build_grbl_profile(
    info: GrblDeviceInfo,
) -> tuple["DeviceProfile", list[str]]:
    """
    Build a ``DeviceProfile`` from a Grbl device census.

    This is a pure data-transformation function with no I/O; the
    census itself comes from ``raygeo.driver.grbl.probe``. Maps the
    protocol-neutral values onto Rayforge's profile model and adds
    user-facing warnings.
    """
    from ...device.profile import (
        DeviceMeta,
        DeviceProfile,
        MachineConfig,
    )

    warnings: list[str] = []

    driver_config: dict[str, Any] = {}
    if info.rx_buffer_size is not None:
        driver_config["rx_buffer_size"] = info.rx_buffer_size
    if info.firmware_version is not None:
        driver_config["firmware_version"] = info.firmware_version
    if info.arc_tolerance is not None:
        driver_config["arc_tolerance"] = info.arc_tolerance

    heads: list[dict[str, Any]] | None = None
    if info.max_spindle is not None:
        heads = [{"max_power": info.max_spindle}]

    if info.report_inches:
        warnings.append(
            _(
                "Device is configured to report in inches "
                "($13=1). All values shown are in machine "
                "units."
            )
        )

    detected_unit_system = UnitSystem.METRIC
    if info.report_inches:
        detected_unit_system = UnitSystem.IMPERIAL

    if info.laser_mode is False:
        warnings.append(
            _(
                "Laser mode is not enabled ($32=0). "
                "Enable it for best results with laser "
                "cutters."
            )
        )

    dialect_config: dict[str, Any] = {}
    if info.flavor == GrblFlavor.STOCK:
        dialect_config = GRBL_DIALECT.to_template_dict()
    elif info.flavor in (GrblFlavor.GRBLHAL, GrblFlavor.FLUIDNC):
        dialect_config = GRBL_DYNAMIC_DIALECT.to_template_dict()

    return (
        DeviceProfile(
            meta=DeviceMeta(
                name=info.name,
                description=_("Auto-configured via probe wizard"),
            ),
            machine_config=MachineConfig(
                driver_config=driver_config or None,
                axis_extents=info.max_travel,
                max_travel_speed=info.max_speed,
                max_cut_speed=info.max_speed,
                acceleration=info.acceleration,
                home_on_start=info.homing_enabled,
                single_axis_homing_enabled=(info.single_axis_homing or None),
                unit_system=detected_unit_system,
                heads=heads,
            ),
            dialect_config=dialect_config,
        ),
        warnings,
    )

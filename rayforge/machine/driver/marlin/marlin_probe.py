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
from raygeo.driver.grbl.probe import GrblDeviceInfo
from raygeo.driver.marlin.probe import MarlinDeviceInfo
from raygeo.driver.octoprint.probe import OctoPrintDeviceInfo
from raygeo.driver.smoothie.probe import SmoothieDeviceInfo

from ....shared.units.system import UnitSystem
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
        boot_lines: list[str] | None = None,
    ) -> Awaitable[
        GrblDeviceInfo
        | MarlinDeviceInfo
        | SmoothieDeviceInfo
        | OctoPrintDeviceInfo
    ]: ...


@runtime_checkable
class _MarlinProbeDriver(Protocol):
    """
    Protocol describing the interface ``probe_marlin_device`` needs
    from any Marlin driver.
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

    @property
    def boot_lines(self) -> list[str]: ...


async def probe_marlin_device(
    driver_cls: type[_MarlinProbeDriver],
    context: "RayforgeContext",
    **kwargs: Any,
) -> tuple["DeviceProfile", list[str]]:
    """
    Shared probe orchestration for all Marlin drivers.

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
        await asyncio.wait_for(connected.wait(), timeout=15.0)
        session = driver.session
        if session is None:
            raise DeviceConnectionError(_("Driver is not connected."))
        info = await session.probe_device(driver.boot_lines)
        if not isinstance(info, MarlinDeviceInfo):
            raise DeviceConnectionError(_("Device is not a Marlin device."))
    finally:
        driver.connection_status_changed.disconnect(_on_status)
        await driver.cleanup()
        context.dialect_mgr.dialects_changed.disconnect(
            machine._on_dialects_changed
        )

    profile, warnings = build_marlin_profile(info)
    profile.machine_config.driver = driver_cls.__name__
    profile.machine_config.driver_args = kwargs
    return profile, warnings


def build_marlin_profile(
    info: MarlinDeviceInfo,
) -> tuple["DeviceProfile", list[str]]:
    """
    Build a ``DeviceProfile`` from a Marlin device census.

    This is a pure data-transformation function with no I/O; the
    census itself comes from ``raygeo.driver.marlin.probe``.
    """
    from ...device.profile import (
        DeviceMeta,
        DeviceProfile,
        MachineConfig,
    )

    warnings: list[str] = []

    driver_config: dict[str, Any] = {}
    if info.firmware_version is not None:
        driver_config["firmware_version"] = info.firmware_version

    detected_unit_system = UnitSystem.METRIC
    if info.units == "imperial":
        detected_unit_system = UnitSystem.IMPERIAL

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
                home_on_start=None,
                single_axis_homing_enabled=True,
                supports_arcs=True,
                unit_system=detected_unit_system,
                heads=None,
            ),
            dialect_config={},
        ),
        warnings,
    )

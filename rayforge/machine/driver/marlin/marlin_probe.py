import asyncio
import logging
from gettext import gettext as _
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TYPE_CHECKING,
    runtime_checkable,
)

from blinker import Signal

from ...transport import TransportStatus
from .marlin_util import (
    parse_m115_firmware_info,
    parse_m211_endstops,
    parse_m503_settings,
    extract_marlin_device_name,
    parse_marlin_version,
)

if TYPE_CHECKING:
    from ....context import RayforgeContext
    from ...device.profile import DeviceProfile
    from ...models.machine import Machine

logger = logging.getLogger(__name__)


@runtime_checkable
class _MarlinProbeDriver(Protocol):
    """
    Protocol describing the interface ``probe_marlin_device`` needs
    from any Marlin driver.
    """

    connection_status_changed: Signal

    def __init__(
        self,
        context: "RayforgeContext",
        machine: "Machine",
    ) -> None: ...

    def setup(self, **kwargs: Any) -> None: ...

    async def connect(self) -> None: ...

    async def cleanup(self) -> None: ...

    async def execute_interactive_command(self, command: str) -> List[str]: ...

    @property
    def boot_lines(self) -> List[str]: ...


async def probe_marlin_device(
    driver_cls: Type[_MarlinProbeDriver],
    context: "RayforgeContext",
    **kwargs: Any,
) -> Tuple["DeviceProfile", List[str]]:
    """
    Shared probe orchestration for all Marlin drivers.

    Creates a temporary machine + driver, connects, queries M115,
    M211, and M503, disconnects, and returns a
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
        m115_lines = await driver.execute_interactive_command("M115")
        m211_lines = await driver.execute_interactive_command("M211")
        m503_lines = await driver.execute_interactive_command("M503")
    finally:
        driver.connection_status_changed.disconnect(_on_status)
        await driver.cleanup()
        context.dialect_mgr.dialects_changed.disconnect(
            machine._on_dialects_changed
        )

    profile, warnings = build_marlin_profile(
        m115_lines, m211_lines, m503_lines, driver.boot_lines
    )
    profile.machine_config.driver = driver_cls.__name__
    profile.machine_config.driver_args = kwargs
    return profile, warnings


def build_marlin_profile(
    m115_lines: List[str],
    m211_lines: List[str],
    m503_lines: List[str],
    boot_lines: Optional[List[str]] = None,
) -> Tuple["DeviceProfile", List[str]]:
    """
    Build a ``DeviceProfile`` from raw Marlin M115, M211, and M503
    response lines.

    This is a pure data-transformation function with no I/O.
    The caller is responsible for communicating with the device
    and passing the collected response lines.

    Returns a ``(DeviceProfile, warnings)`` tuple where *warnings*
    is a list of human-readable strings about potential issues
    detected in the device configuration.
    """
    from ...device.profile import (
        DeviceMeta,
        DeviceProfile,
        MachineConfig,
    )

    boot = boot_lines or []
    warnings: List[str] = []

    name = extract_marlin_device_name(m115_lines, boot)

    fw_info = parse_m115_firmware_info(m115_lines)
    driver_config: Dict[str, Any] = {}
    fw_name = fw_info.get("firmware_name", "")
    if fw_name:
        for line in boot:
            ver = parse_marlin_version(line)
            if ver:
                fw_name = ver
                break
        driver_config["firmware_version"] = fw_name

    extents: Optional[Tuple[float, float]] = None
    endstops = parse_m211_endstops(m211_lines)
    if endstops is not None:
        x_max, y_max = endstops
        if x_max > 0 and y_max > 0:
            extents = (x_max, y_max)

    m503 = parse_m503_settings(m503_lines)

    max_speed: Optional[int] = None
    max_feed_x = m503.get("max_feedrate_x")
    max_feed_y = m503.get("max_feedrate_y")
    if max_feed_x is not None and max_feed_y is not None:
        max_speed_mm_s = min(max_feed_x, max_feed_y)
        max_speed = int(max_speed_mm_s * 60)

    accel: Optional[int] = None
    accel_val = m503.get("acceleration")
    if accel_val is not None:
        accel = int(accel_val)

    return (
        DeviceProfile(
            meta=DeviceMeta(
                name=name,
                description=_("Auto-configured via probe wizard"),
            ),
            machine_config=MachineConfig(
                driver_config=driver_config or None,
                axis_extents=extents,
                max_travel_speed=max_speed,
                max_cut_speed=max_speed,
                acceleration=accel,
                home_on_start=None,
                single_axis_homing_enabled=True,
                supports_arcs=True,
                heads=None,
            ),
            dialect_config={},
        ),
        warnings,
    )

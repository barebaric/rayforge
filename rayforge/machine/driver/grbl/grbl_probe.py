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

from .grbl_util import (
    grbl_opt_re,
    parse_opt_info,
    parse_grbl_settings,
    extract_device_name,
    parse_version,
)
from ...transport import TransportStatus

if TYPE_CHECKING:
    from ....context import RayforgeContext
    from ...device.profile import DeviceProfile
    from ...models.machine import Machine

logger = logging.getLogger(__name__)


@runtime_checkable
class _GrblProbeDriver(Protocol):
    """
    Protocol describing the interface ``probe_grbl_device`` needs
    from any Grbl driver.  Any driver implementing these methods
    can be probed — serial, telnet, network, or otherwise.
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


async def probe_grbl_device(
    driver_cls: Type[_GrblProbeDriver],
    context: "RayforgeContext",
    **kwargs: Any,
) -> Tuple["DeviceProfile", List[str]]:
    """
    Shared probe orchestration for all Grbl drivers.

    Creates a temporary machine + driver, connects, queries $I and $$,
    disconnects, and returns a ``(DeviceProfile, warnings)`` tuple.
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
        build_info = await driver.execute_interactive_command("$I")
        settings_lines = await driver.execute_interactive_command("$$")
    finally:
        driver.connection_status_changed.disconnect(_on_status)
        await driver.cleanup()
        context.dialect_mgr.dialects_changed.disconnect(
            machine._on_dialects_changed
        )

    profile, warnings = build_grbl_profile(build_info, settings_lines)
    profile.machine_config.driver = driver_cls.__name__
    profile.machine_config.driver_args = kwargs
    return profile, warnings


def build_grbl_profile(
    build_info: List[str],
    settings_lines: List[str],
) -> Tuple["DeviceProfile", List[str]]:
    """
    Build a ``DeviceProfile`` from raw Grbl ``$I`` and ``$``
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

    rx_buffer_size: Optional[int] = None
    compile_flags: str = ""
    for line in build_info:
        rx = parse_opt_info(line)
        if rx is not None:
            rx_buffer_size = rx
        match = grbl_opt_re.search(line)
        if match:
            compile_flags = match.group(1)

    settings = parse_grbl_settings(settings_lines)
    warnings: List[str] = []

    name = extract_device_name(build_info)
    axis_x = settings.get("130")
    axis_y = settings.get("131")
    max_x_rate = settings.get("110")
    max_y_rate = settings.get("111")
    x_accel = settings.get("120")
    y_accel = settings.get("121")
    max_spindle = settings.get("30")
    laser_mode = settings.get("32")
    homing_enabled = settings.get("22")
    report_inches = settings.get("13")

    max_speed: Optional[int] = None
    if max_x_rate is not None and max_y_rate is not None:
        max_speed = int(min(max_x_rate, max_y_rate))

    accel: Optional[int] = None
    if x_accel is not None and y_accel is not None:
        accel = int(min(x_accel, y_accel))

    single_axis_homing = "H" in compile_flags

    driver_config: Dict[str, Any] = {}
    if rx_buffer_size is not None:
        driver_config["rx_buffer_size"] = rx_buffer_size

    fw_version = parse_version(build_info)
    if fw_version is not None:
        driver_config["firmware_version"] = fw_version

    arc_tol = settings.get("12")
    if arc_tol is not None:
        driver_config["arc_tolerance"] = arc_tol

    extents: Optional[Tuple[float, float]] = None
    if axis_x is not None and axis_y is not None:
        extents = (float(axis_x), float(axis_y))

    heads: Optional[List[Dict[str, Any]]] = None
    if max_spindle is not None:
        heads = [{"max_power": int(max_spindle)}]

    home_on_start: Optional[bool] = None
    if homing_enabled is not None:
        home_on_start = bool(int(homing_enabled))

    if report_inches is not None and int(report_inches):
        warnings.append(
            _(
                "Device is configured to report in inches "
                "($13=1). All values shown are in machine "
                "units."
            )
        )

    if laser_mode is not None and not int(laser_mode):
        warnings.append(
            _(
                "Laser mode is not enabled ($32=0). "
                "Enable it for best results with laser "
                "cutters."
            )
        )

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
                home_on_start=home_on_start,
                single_axis_homing_enabled=(single_axis_homing or None),
                heads=heads,
            ),
            dialect_config={},
        ),
        warnings,
    )

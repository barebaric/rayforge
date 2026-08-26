import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...device.profile import DeviceProfile

logger = logging.getLogger(__name__)


def first_value(lines: list[str]) -> str | None:
    """Return the first non-empty reply line, if any."""
    for line in lines:
        text = line.strip()
        if text:
            return text
    return None


def parse_float(lines: list[str]) -> float | None:
    """Parse a config-get numeric reply into a float.

    Smoothieware replies to ``config-get <key>`` with the bare value
    (e.g. ``200.000``) followed by ``ok``; the ``ok`` line is filtered
    out by the driver before these lines reach us.
    """
    raw = first_value(lines)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Could not parse Smoothie config value: {raw!r}")
        return None


def build_smoothie_profile(
    version_lines: list[str],
    alpha_max_lines: list[str],
    beta_max_lines: list[str],
    alpha_max_rate_lines: list[str],
    beta_max_rate_lines: list[str],
    acceleration_lines: list[str],
) -> tuple["DeviceProfile", list[str]]:
    """
    Build a ``DeviceProfile`` from raw Smoothieware ``version`` and
    ``config-get`` response lines.

    This is a pure data-transformation function with no I/O. The caller
    is responsible for communicating with the device and passing the
    collected response lines.

    Returns a ``(DeviceProfile, warnings)`` tuple where *warnings* is a
    list of human-readable strings about potential issues. All
    failures become warnings; this function never raises.
    """
    from ...device.profile import (
        DeviceMeta,
        DeviceProfile,
        MachineConfig,
    )

    warnings: list[str] = []

    name = "Smoothieware"
    fw_version = first_value(version_lines)
    driver_config: dict[str, Any] = {}
    if fw_version:
        driver_config["firmware_version"] = fw_version

    alpha_max = parse_float(alpha_max_lines)
    beta_max = parse_float(beta_max_lines)
    extents: tuple[float, float] | None = None
    if alpha_max is not None and beta_max is not None:
        if alpha_max > 0 and beta_max > 0:
            extents = (alpha_max, beta_max)
        else:
            warnings.append(
                _(
                    "Smoothie reported a non-positive work-area "
                    "dimension (alpha_max={x}, beta_max={y})."
                ).format(x=alpha_max, y=beta_max)
            )
    else:
        warnings.append(_("Could not read the Smoothie work-area dimensions."))

    alpha_rate = parse_float(alpha_max_rate_lines)
    beta_rate = parse_float(beta_max_rate_lines)
    max_speed: int | None = None
    if alpha_rate is not None and beta_rate is not None:
        # config-get rates are in mm/s; machine config expects mm/min.
        max_speed = int(min(alpha_rate, beta_rate) * 60)
    else:
        warnings.append(_("Could not read the Smoothie maximum feed rates."))

    accel_val = parse_float(acceleration_lines)
    accel: int | None = None
    if accel_val is not None:
        accel = int(accel_val)
    else:
        warnings.append(_("Could not read the Smoothie acceleration."))

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

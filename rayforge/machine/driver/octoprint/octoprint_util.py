import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...device.profile import DeviceProfile

logger = logging.getLogger(__name__)


def build_octoprint_profile(
    version_info: dict[str, Any] | None,
    printer_info: dict[str, Any] | None,
) -> tuple["DeviceProfile", list[str]]:
    """
    Build a ``DeviceProfile`` from OctoPrint API responses.

    This is a pure data-transformation function with no I/O. The
    caller is responsible for making the HTTP requests and passing
    the parsed JSON payloads.

    Args:
        version_info: Parsed ``GET /api/version`` response (contains
            ``server`` and ``version`` keys, or ``None`` if the
            request failed).
        printer_info: Parsed ``GET /api/printer`` response (contains
            ``state`` and ``printer`` data, or ``None``).

    Returns a ``(DeviceProfile, warnings)`` tuple where *warnings*
    is a list of human-readable strings. All failures become
    warnings; this function never raises.
    """
    from ...device.profile import (
        DeviceMeta,
        DeviceProfile,
        MachineConfig,
    )

    warnings: list[str] = []
    driver_config: dict[str, Any] = {}

    if version_info is not None:
        server_version = version_info.get("version")
        if server_version:
            driver_config["server_version"] = server_version

    name = "OctoPrint"
    if printer_info is not None:
        printer_name = printer_info.get("printer", {}).get("name")
        if not printer_name:
            state = printer_info.get("state", {})
            text = state.get("text", "") if isinstance(state, dict) else ""
            if text and text != "Operational":
                name = f"OctoPrint ({text})"
        else:
            name = printer_name

    if not version_info:
        warnings.append(_("Could not read the OctoPrint server version."))

    return (
        DeviceProfile(
            meta=DeviceMeta(
                name=name,
                description=_("Auto-configured via probe wizard"),
            ),
            machine_config=MachineConfig(
                driver_config=driver_config or None,
                heads=None,
            ),
            dialect_config={},
        ),
        warnings,
    )

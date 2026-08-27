import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...device.profile import DeviceProfile

logger = logging.getLogger(__name__)


def _positive_length(value: Any) -> float | None:
    """A positive number from API JSON, else ``None`` (booleans and
    non-numeric values are rejected)."""
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    length = float(value)
    return length if length > 0 else None


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
            ``state``, ``printer`` and, when configured on the server,
            ``dimensions`` data, or ``None``).

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

    if isinstance(version_info, dict):
        server_version = version_info.get("version")
        if server_version:
            driver_config["server_version"] = server_version

    name = "OctoPrint"
    axis_extents: tuple[float, float] | None = None
    if isinstance(printer_info, dict):
        # Only the static machine name feeds the profile name; baking
        # transient state text ("Heating bed...") into it would make
        # the persisted name drift with printer state.
        printer = printer_info.get("printer")
        if isinstance(printer, dict):
            printer_name = printer.get("name")
            if isinstance(printer_name, str) and printer_name.strip():
                name = printer_name

        dimensions = printer_info.get("dimensions")
        if isinstance(dimensions, dict):
            x_length = _positive_length(dimensions.get("x_length"))
            y_length = _positive_length(dimensions.get("y_length"))
            if x_length is not None and y_length is not None:
                axis_extents = (x_length, y_length)

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
                axis_extents=axis_extents,
                heads=None,
            ),
            dialect_config={},
        ),
        warnings,
    )

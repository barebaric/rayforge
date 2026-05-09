"""Parsing utilities for the Marlin firmware driver."""

import re
from typing import Dict, List, Optional, Tuple

MARLIN_HANDSHAKE_TIMEOUT = 5.0
MARLIN_COMMAND_TIMEOUT = 30.0
MARLIN_RECONNECT_DELAY = 5.0
MARLIN_STATUS_POLL_INTERVAL = 2.0

m114_pos_re = re.compile(
    r"X:([+-]?\d+\.?\d*)\s+Y:([+-]?\d+\.?\d*)\s+Z:([+-]?\d+\.?\d*)"
)
marlin_version_re = re.compile(r"Marlin\s+([\d.]+)")
marlin_error_re = re.compile(r"Error:(.+)")
marlin_echo_re = re.compile(r"echo:(.+)")
m115_firmware_re = re.compile(r"FIRMWARE_NAME:\s*(\S+)")
m115_machine_type_re = re.compile(r"MACHINE_TYPE:(.+?)(?:\s+[A-Z_]+:|$)")
m203_feedrate_re = re.compile(
    r"echo:\s*M203\s+"
    r"X([+-]?\d+\.?\d*)\s+"
    r"Y([+-]?\d+\.?\d*)"
)
m204_accel_re = re.compile(r"echo:\s*M204\s+S([+-]?\d+\.?\d*)")
m211_max_re = re.compile(
    r"X:?\s*([+-]?\d+\.?\d*)\s+"
    r"Y:?\s*([+-]?\d+\.?\d*)"
)


def parse_m114_position(
    response_lines: List[str],
) -> Optional[Tuple[float, float, float]]:
    """
    Parse M114 output to extract the (X, Y, Z) position.

    Args:
        response_lines: List of response lines from an M114 command.

    Returns:
        A (x, y, z) float tuple, or None if no position line is found.
    """
    for line in response_lines:
        match = m114_pos_re.search(line)
        if match:
            return (
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
            )
    return None


def parse_marlin_version(line: str) -> Optional[str]:
    """
    Extract the Marlin version string from a boot line.

    Args:
        line: A single line containing the firmware identifier,
              e.g. ``"Marlin 2.1.2.7"``.

    Returns:
        The version string (e.g. ``"2.1.2.7"``), or None if not found.
    """
    match = marlin_version_re.search(line)
    if match:
        return match.group(1)
    return None


def is_ok_response(line: str) -> bool:
    """
    Check whether a line is a Marlin ``ok`` acknowledgment.

    Handles plain ``ok`` as well as temperature-augmented forms such
    as ``ok T:19.5 /200.0 B:60.0 /60.0`` and
    ``ok P:15 B:3``.
    """
    stripped = line.strip()
    return stripped.startswith("ok") and (
        len(stripped) == 2 or not stripped[2].isalnum()
    )


def is_error_response(line: str) -> bool:
    """
    Check whether a line is a Marlin error response.

    Args:
        line: A single response line.

    Returns:
        True if the line starts with ``Error:``.
    """
    return line.strip().startswith("Error:")


def parse_error_message(line: str) -> str:
    """
    Extract the error message from an ``Error:...`` line.

    Args:
        line: A single response line starting with ``Error:``.

    Returns:
        The message portion after ``Error:`` (whitespace-stripped).
    """
    match = marlin_error_re.search(line)
    if match:
        return match.group(1).strip()
    return ""


def is_boot_message(line: str) -> bool:
    """
    Check whether a line is a Marlin boot/startup message.

    Recognised prefixes: ``start``, ``Marlin``, ``echo:``,
    ``External Reset``.
    """
    stripped = line.strip()
    prefixes = ("start", "Marlin", "echo:", "External Reset")
    return any(stripped.startswith(p) for p in prefixes)


def parse_m115_firmware_info(
    response_lines: List[str],
) -> Dict[str, str]:
    """
    Parse M115 response to extract firmware name and machine type.

    Returns a dict with optional keys ``firmware_name`` and
    ``machine_type``.
    """
    info: Dict[str, str] = {}
    for line in response_lines:
        match = m115_firmware_re.search(line)
        if match:
            info["firmware_name"] = match.group(1).strip()
        match = m115_machine_type_re.search(line)
        if match:
            info["machine_type"] = match.group(1).strip()
    return info


def parse_m211_endstops(
    response_lines: List[str],
) -> Optional[Tuple[float, float]]:
    """
    Parse M211 output to extract X/Y max travel from software endstops.

    Returns (x_max, y_max) or None if not found.
    """
    for line in response_lines:
        match = m211_max_re.search(line)
        if match:
            return (float(match.group(1)), float(match.group(2)))
    return None


def parse_m503_settings(
    response_lines: List[str],
) -> Dict[str, float]:
    """
    Parse M503 output to extract key motion settings.

    Returns a dict with optional keys:
      - ``max_feedrate_x``, ``max_feedrate_y`` (mm/s from M203)
      - ``acceleration`` (mm/s^2 from M204 S)
    """
    settings: Dict[str, float] = {}
    for line in response_lines:
        match = m203_feedrate_re.search(line)
        if match:
            settings["max_feedrate_x"] = float(match.group(1))
            settings["max_feedrate_y"] = float(match.group(2))
        match = m204_accel_re.search(line)
        if match:
            settings["acceleration"] = float(match.group(1))
    return settings


def extract_marlin_device_name(
    m115_lines: List[str],
    boot_lines: Optional[List[str]] = None,
) -> str:
    """
    Extract a human-readable device name from M115 and boot output.

    Uses MACHINE_TYPE from M115 first, then falls back to the
    firmware name.
    """
    info = parse_m115_firmware_info(m115_lines)
    machine_type = info.get("machine_type")
    if machine_type and machine_type.lower() not in (
        "3d printer",
        "default",
    ):
        return machine_type

    fw_name = info.get("firmware_name", "")
    if fw_name:
        return fw_name

    if boot_lines:
        for line in boot_lines:
            ver = parse_marlin_version(line)
            if ver:
                return f"Marlin {ver}"

    return "Unknown Marlin Device"


def gcode_to_p_number(wcs_slot: str) -> Optional[int]:
    """Converts a G-code WCS name (e.g., "G54") to its P-number."""
    try:
        if not wcs_slot.startswith("G"):
            return None
        p_num = int(wcs_slot[1:]) - 53
        if 1 <= p_num <= 6:
            return p_num
    except (ValueError, IndexError):
        pass
    return None

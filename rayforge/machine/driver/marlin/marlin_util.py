"""Parsing utilities for the Marlin firmware driver."""

import re
from typing import List, Optional, Tuple
from ..grbl.grbl_util import strip_gcode_comments

__all__ = [
    "strip_gcode_comments",
]

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

"""GRBL firmware dialect detection from ``$I`` and ``$$`` output.

Determines which built-in G-code dialect best matches a discovered
GRBL device based on its build-info version string, compile-option
flags, and settings-key inventory. Pure functions only — no I/O.
"""

import logging
import re

from .grbl_util import parse_version

logger = logging.getLogger(__name__)

#: Regex matching grblHAL / FluidNC identifiers in a ``[VER:]`` line.
_HAL_RE = re.compile(r"grblhal|fluidnc", re.IGNORECASE)

#: grblHAL exposes per-axis settings under the ``$4xx`` group
#: (e.g. ``$400`` for axis steps/mm). Stock Grbl uses ``$0``–``$255``.
_HAL_SETTING_RE = re.compile(r"\$4\d{2}")


def _has_hal_settings(settings_lines: list[str]) -> bool:
    """True when ``$$`` output includes grblHAL-specific ``$4xx`` keys."""
    return any(_HAL_SETTING_RE.search(line) for line in settings_lines)


def detect_grbl_dialect(
    build_info: list[str],
    settings_lines: list[str] | None = None,
) -> str | None:
    """Inspect ``$I`` and ``$$`` output and return the uid of the
    best-matching built-in dialect, or ``None`` to keep the current
    default.

    Signals:
    - ``[VER:]`` contains ``grblHAL`` or ``FluidNC`` → ``grbl_dynamic``
    - ``$$`` includes ``$4xx`` grouped axis settings (grblHAL) →
      ``grbl_dynamic``
    - Stock Grbl 1.1 (no HAL signals) → ``grbl``
    - Unrecognized / empty → ``None``
    """
    settings_lines = settings_lines or []

    for line in build_info:
        if _HAL_RE.search(line):
            return "grbl_dynamic"

    if _has_hal_settings(settings_lines):
        return "grbl_dynamic"

    version = parse_version(build_info)
    if version is not None:
        return "grbl"

    return None


__all__ = ["detect_grbl_dialect"]

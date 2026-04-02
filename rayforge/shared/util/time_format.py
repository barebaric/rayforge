"""Utility functions for formatting time values."""

from gettext import gettext as _


def format_hours_to_hm(hours: float) -> str:
    """
    Format a fractional hours value to hours and minutes string.

    Args:
        hours: Fractional hours value (e.g., 10.5 for 10h 30m).

    Returns:
        Formatted string like "10h 30m", "10h", or "30m".
        Zero values are omitted (e.g., 0.5h -> "30m", 10h -> "10h").
    """
    h = int(hours)
    m = int((hours - h) * 60)
    if h > 0 and m > 0:
        return f"{h}h {m}m"
    if h > 0:
        return f"{h}h"
    return f"{m}m"


def format_seconds(seconds: float, compact: bool = False) -> str:
    """
    Format a number of seconds into a human-readable time string.

    Args:
        seconds: Positive number of seconds.
        compact: If True, omit spaces and zero-valued components
            (e.g. "3s", "2m5s", "1h30m" vs "3s", "2m 5s", "1h 30m").

    Returns:
        Formatted time string.
    """
    sep = "" if compact else " "

    if seconds < 60:
        return _("{:.0f}s").format(seconds)
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if compact and secs == 0:
            return _("{}m").format(minutes)
        return _(f"{{}}m{sep}{{}}s").format(minutes, secs)
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        if compact and mins == 0:
            return _("{}h").format(hours)
        return _(f"{{}}h{sep}{{}}m").format(hours, mins)

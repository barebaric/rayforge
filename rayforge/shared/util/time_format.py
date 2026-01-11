"""Utility functions for formatting time values."""


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

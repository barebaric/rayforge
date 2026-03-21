"""
Size comparison utilities for pipeline artifacts.
"""

import math
from gettext import gettext as _
from typing import Optional, Tuple


def sizes_are_close(
    size1: Optional[Tuple[float, float]],
    size2: Optional[Tuple[float, float]],
) -> bool:
    """Compares two size tuples with a safe tolerance for float errors."""
    if size1 is None or size2 is None:
        return False
    return math.isclose(size1[0], size2[0], abs_tol=1e-6) and math.isclose(
        size1[1], size2[1], abs_tol=1e-6
    )


def format_byte_size(size_bytes: int) -> str:
    """
    Format a byte size as a human-readable string.

    Args:
        size_bytes: The size in bytes.

    Returns:
        A formatted string like "512 B", "1.5 KB", or "2.3 MB".
    """
    if size_bytes < 1024:
        return _("{size} B").format(size=size_bytes)
    elif size_bytes < 1024 * 1024:
        return _("{size:.1f} KB").format(size=size_bytes / 1024)
    else:
        return _("{size:.1f} MB").format(size=size_bytes / (1024 * 1024))

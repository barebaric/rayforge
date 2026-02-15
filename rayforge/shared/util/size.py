"""
Size comparison utilities for pipeline artifacts.
"""

import math
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

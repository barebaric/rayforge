"""Speed-dependent bidirectional scan offset lookup."""

from itertools import pairwise


def interpolate_bidir_offset(
    table: list[tuple[float, float]], speed: float
) -> float:
    """
    Bidirectional scan offset for ``speed`` from a speed table.

    Args:
        table: ``(speed mm/min, offset mm)`` rows sorted by speed.
        speed: The scan speed in mm/min.

    Returns:
        The offset in mm, interpolated linearly between rows and
        clamped to the first and last row; 0.0 for an empty table.
    """
    if not table:
        return 0.0
    if speed <= table[0][0]:
        return table[0][1]
    for (lo_speed, lo_offset), (hi_speed, hi_offset) in pairwise(table):
        if speed <= hi_speed:
            t = (speed - lo_speed) / (hi_speed - lo_speed)
            return lo_offset + t * (hi_offset - lo_offset)
    return table[-1][1]

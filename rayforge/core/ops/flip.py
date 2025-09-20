from copy import copy
from typing import List, cast
from .commands import ArcToCommand, MovingCommand


def flip_segment(segment: List[MovingCommand]) -> List[MovingCommand]:
    """
    Reverses a segment of path commands, correctly adjusting states and
    ArcToCommand parameters.

    The states attached to each point describe the intended machine state
    while traveling TO that point. When flipping, states must be shifted
    to maintain this relationship. Arcs must also have their parameters
    recalculated relative to their new start point.
    """
    length = len(segment)
    if length <= 1:
        return segment

    new_segment = []
    for i in range(length - 1, -1, -1):
        cmd = segment[i]
        prev_cmd = segment[(i + 1) % length]
        new_cmd = copy(prev_cmd)
        new_cmd.end = cmd.end

        # Fix arc_to parameters
        if isinstance(new_cmd, ArcToCommand):
            # Get original arc (prev op in original segment)
            orig_cmd = cast(ArcToCommand, segment[i + 1])
            x_end, y_end, _ = orig_cmd.end
            i_orig, j_orig = orig_cmd.center_offset

            # Calculate center and new offsets.
            # new_cmd.end holds the start point of the original arc.
            assert new_cmd.end is not None, "Arc must have an endpoint"
            x_start, y_start, z_start = new_cmd.end
            center_x = x_start + i_orig
            center_y = y_start + j_orig
            new_i = center_x - x_end
            new_j = center_y - y_end

            # Update arc parameters
            new_cmd.end = (x_start, y_start, z_start)
            new_cmd.center_offset = (new_i, new_j)
            new_cmd.clockwise = not orig_cmd.clockwise

        new_segment.append(new_cmd)

    return new_segment

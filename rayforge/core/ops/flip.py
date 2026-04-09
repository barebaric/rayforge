from copy import copy
from typing import List
from .commands import (
    ArcToCommand,
    BezierToCommand,
    MovingCommand,
    MoveToCommand,
    ScanLinePowerCommand,
)


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

    new_segment: List[MovingCommand] = []

    # The first command of the new segment is a MoveTo, created from the
    # original segment's first MoveTo command.
    first_cmd = MoveToCommand(end=segment[-1].end)
    first_cmd.state = segment[0].state
    new_segment.append(first_cmd)

    # Process the rest of the commands in reverse
    for i in range(length - 2, -1, -1):
        original_cmd = segment[i + 1]
        new_cmd = copy(original_cmd)  # Copies type (LineTo, ArcTo) and state
        new_cmd.end = segment[i].end

        if isinstance(new_cmd, ScanLinePowerCommand):
            # For a reversed scanline, we only need to reverse the power data.
            # The geometry (start/end) is handled like any other MovingCommand.
            new_cmd.power_values = new_cmd.power_values[::-1]
        elif isinstance(new_cmd, BezierToCommand):
            # A reversed cubic bezier P0→C1→C2→P3 becomes P3→C2→C1→P0,
            # so the control points must be swapped.
            new_cmd.control1, new_cmd.control2 = (
                new_cmd.control2,
                new_cmd.control1,
            )
        elif isinstance(new_cmd, ArcToCommand):
            # The original arc's start point is the endpoint of the command
            # before it in the original segment.
            original_start = segment[i].end
            assert original_start is not None
            # Delegate the complex recalculation to the command itself.
            new_cmd.reverse_geometry(
                original_start=original_start,
                original_end=original_cmd.end,
            )
        new_segment.append(new_cmd)

    return new_segment

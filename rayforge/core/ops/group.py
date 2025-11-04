from typing import List, cast
from .commands import (
    Command,
    MovingCommand,
    LineToCommand,
    ArcToCommand,
    MoveToCommand,
)


def group_by_state_continuity(
    operations: List[Command],
) -> List[List[Command]]:
    """
    Splits a command list into segments where state changes (like toggling
    air assist) or marker commands create boundaries. This is used to find
    sub-lists of commands that can be safely reordered.
    """
    if not operations:
        return []

    segments: List[List[Command]] = []
    current_segment: List[Command] = []

    for op in operations:
        if op.is_marker():
            if current_segment:
                segments.append(current_segment)
            segments.append([op])
            current_segment = []
            continue

        if not current_segment:
            current_segment.append(op)
            continue

        last_state = current_segment[-1].state
        if last_state and op.state and last_state.allow_rapid_change(op.state):
            current_segment.append(op)
        else:
            segments.append(current_segment)
            current_segment = [op]

    if current_segment:
        segments.append(current_segment)

    return segments


def group_by_path_continuity(
    commands: List[Command],
) -> List[List[MovingCommand]]:
    """
    Splits a command list into continuous path segments. Each segment starts
    with a travel move and is followed by one or more cutting moves.
    """
    segments: List[List[MovingCommand]] = []
    current_segment: List[MovingCommand] = []
    for cmd in commands:
        if cmd.is_travel_command():
            if current_segment:
                segments.append(current_segment)
            current_segment = [cast(MovingCommand, cmd)]
        elif cmd.is_cutting_command():
            current_segment.append(cast(MovingCommand, cmd))
        else:
            # This function assumes a pre-filtered list of only moving commands
            raise ValueError(f"Unexpected non-moving command: {cmd}")

    if current_segment:
        segments.append(current_segment)
    return segments


def group_by_command_type(commands: List[Command]) -> List[List[Command]]:
    """
    Splits commands into segments based on command type transitions, preserving
    logical groupings for processes like arc welding. State commands form
    their own segments.
    """
    segments: List[List[Command]] = []
    current_segment: List[Command] = []
    current_pos = None

    for cmd in commands:
        if isinstance(cmd, MoveToCommand):
            if current_segment:
                segments.append(current_segment)
            current_segment = [cmd]
            current_pos = cmd.end
        elif isinstance(cmd, ArcToCommand):
            if any(isinstance(c, LineToCommand) for c in current_segment):
                segments.append(current_segment)
                current_segment = []
            if not current_segment:
                if current_pos is None:
                    # An arc needs a start point, create an implicit MoveTo
                    current_segment.append(MoveToCommand((0.0, 0.0, 0.0)))
                else:
                    current_segment.append(MoveToCommand(current_pos))
            current_segment.append(cmd)
            current_pos = cmd.end
        elif isinstance(cmd, LineToCommand):
            if any(isinstance(c, ArcToCommand) for c in current_segment):
                segments.append(current_segment)
                current_segment = []
            if not current_segment:
                if current_pos is None:
                    raise ValueError(
                        "LineToCommand requires a starting position."
                    )
                current_segment.append(MoveToCommand(current_pos))
            current_segment.append(cmd)
            current_pos = cmd.end
        elif cmd.is_state_command() or cmd.is_marker():
            if current_segment:
                segments.append(current_segment)
            segments.append([cmd])
            current_segment = []
        else:
            # Should not be reached if all command types are handled
            raise ValueError(f"Unsupported command type for grouping: {cmd}")

    if current_segment:
        segments.append(current_segment)

    return segments

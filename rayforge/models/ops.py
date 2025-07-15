from __future__ import annotations
import math
from copy import copy, deepcopy
from typing import List
from dataclasses import dataclass


@dataclass
class State:
    power: int = 0
    air_assist: bool = False
    cut_speed: int = None
    travel_speed: int = None

    def allow_rapid_change(self, target_state):
        """
        Returns True if a change to the target state should be allowed
        in a rapid manner, i.e. for each gcode instruction. For example,
        changing air-assist should not be done too frequently, because
        it could damage the air pump.

        Changing the laser power rapidly is unproblematic.
        """
        return self.air_assist == target_state.air_assist


class Command:
    """
    Note that the state attribute is not set by default. It is later
    filled during the pre-processing stage, where state commands are
    removed.
    """
    def __init__(self, end=None, state=None):
        self.end: tuple = end  # x, y of end position, None for state command
        self.state: State = state  # Intended state during execution
        self._state_ref_for_pyreverse: State

    def __repr__(self):
        return f"<{super().__repr__()} {self.__dict__}"

    def apply_to_state(self, state):
        pass

    def is_state_command(self):
        return False

    def is_cutting_command(self):
        """Whether it is a cutting movement"""
        return False

    def is_travel_command(self):
        """Whether it is a non-cutting movement"""
        return False


class MoveToCommand(Command):
    def is_travel_command(self):
        return True


class LineToCommand(Command):
    def is_cutting_command(self):
        return True


class ArcToCommand(Command):
    def __init__(self, end, center_offset, clockwise):
        super().__init__(end)
        self.center_offset: tuple = center_offset
        self.clockwise: bool = clockwise

    def is_cutting_command(self):
        return True


class SetPowerCommand(Command):
    def __init__(self, power):
        super().__init__()
        self.power: int = power

    def is_state_command(self):
        return True

    def apply_to_state(self, state):
        state.power = self.power


class SetCutSpeedCommand(Command):
    def __init__(self, speed):
        super().__init__()
        self.speed: int = speed

    def is_state_command(self):
        return True

    def apply_to_state(self, state):
        state.cut_speed = self.speed


class SetTravelSpeedCommand(Command):
    def __init__(self, speed):
        super().__init__()
        self.speed: int = speed

    def is_state_command(self):
        return True

    def apply_to_state(self, state):
        state.travel_speed = self.speed


class EnableAirAssistCommand(Command):
    def is_state_command(self):
        return True

    def apply_to_state(self, state):
        state.air_assist = True


class DisableAirAssistCommand(Command):
    def is_state_command(self):
        return True

    def apply_to_state(self, state):
        state.air_assist = False


class Ops:
    """
    Represents a set of generated path segments and instructions that
    are used for making gcode, but also to generate vector graphics
    for display.
    """
    def __init__(self):
        self.commands: List[Command] = []
        self._commands_ref_for_pyreverse: Command
        self.last_move_to = 0.0, 0.0

    def __iter__(self):
        return iter(self.commands)

    def __add__(self, ops):
        result = Ops()
        result.commands = self.commands + ops.commands
        return result

    def __mul__(self, count):
        result = Ops()
        result.commands = count*self.commands
        return result

    def __len__(self):
        return len(self.commands)

    def copy(self) -> Ops:
        """Creates a deep copy of the Ops object."""
        new_ops = Ops()
        new_ops.commands = deepcopy(self.commands)
        new_ops.last_move_to = self.last_move_to
        return new_ops

    def preload_state(self):
        """
        Walks through all commands, enriching each by the indended
        state of the machine. The state is useful for some post-processors
        that need to re-order commands without changing the intended
        state during each command.

        Returns a list of Command objects. Any state commands are wipe out,
        as the state is now included in every operation.
        """
        state = State()
        for cmd in self.commands:
            if cmd.is_state_command():
                cmd.apply_to_state(state)
            else:
                cmd.state = copy(state)

    def clear(self):
        self.commands = []

    def add(self, command):
        self.commands.append(command)

    def move_to(self, x, y):
        self.last_move_to = float(x), float(y)
        cmd = MoveToCommand(self.last_move_to)
        self.commands.append(cmd)

    def line_to(self, x, y):
        cmd = LineToCommand((float(x), float(y)))
        self.commands.append(cmd)

    def close_path(self):
        """
        Convenience method that wraps line_to(). Makes a line to
        the last move_to point.
        """
        self.line_to(*self.last_move_to)

    def arc_to(self, x, y, i, j, clockwise=True):
        """
        Adds an arc command with specified endpoint, center offsets,
        and direction (cw/ccw).
        """
        self.commands.append(ArcToCommand(
            (float(x), float(y)), (float(i), float(j)), bool(clockwise)
        ))

    def set_power(self, power: float):
        """Laser power (0-1000 for GRBL)"""
        cmd = SetPowerCommand(float(power))
        self.commands.append(cmd)

    def set_cut_speed(self, speed: float):
        """Cutting speed (mm/min)"""
        cmd = SetCutSpeedCommand(float(speed))
        self.commands.append(cmd)

    def set_travel_speed(self, speed: float):
        """Rapid movement speed (mm/min)"""
        cmd = SetTravelSpeedCommand(float(speed))
        self.commands.append(cmd)

    def enable_air_assist(self, enable=True):
        if enable:
            self.commands.append(EnableAirAssistCommand())
        else:
            self.disable_air_assist()

    def disable_air_assist(self):
        self.commands.append(DisableAirAssistCommand())

    def rect(self):
        """
        Returns a rectangle (x1, y1, x2, y2) that encloses the
        occupied area.
        """
        occupied_points = []
        last_point = None
        for cmd in self.commands:
            if cmd.is_travel_command():
                last_point = cmd.end
            elif cmd.is_cutting_command():
                occupied_points.append(last_point)
                occupied_points.append(cmd.end)
                last_point = cmd.end

        if not occupied_points:
            return 0, 0, 0, 0

        xs = [p[0] for p in occupied_points]
        ys = [p[1] for p in occupied_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return min_x, min_y, max_x, max_y

    def get_frame(self, power=None, speed=None):
        """
        Returns a new Ops object containing four move_to operations forming
        a frame around the occupied area of the original Ops. The occupied
        area includes all points from line_to and close_path commands.
        """
        min_x, min_y, max_x, max_y = self.rect()
        if (min_x, min_y, max_x, max_y) == (0, 0, 0, 0):
            return Ops()

        frame_ops = Ops()
        if power is not None:
            frame_ops.set_power(power)
        if speed is not None:
            frame_ops.set_cut_speed(speed)
        frame_ops.move_to(min_x, min_y)
        frame_ops.line_to(min_x, max_y)
        frame_ops.line_to(max_x, max_y)
        frame_ops.line_to(max_x, min_y)
        frame_ops.line_to(min_x, min_y)
        return frame_ops

    def distance(self):
        """
        Calculates the total distance of all moves. Mostly exists to help
        debug the optimize() method.
        """
        total = 0.0

        last = None
        for cmd in self.commands:
            if cmd.is_travel_command():
                if last is not None:
                    total += math.dist(cmd.end, last)
                last = cmd.end
            elif cmd.is_cutting_command():
                # treating arcs as lines is probably good enough
                if last is not None:
                    total += math.dist(cmd.end, last)
                last = cmd.end
        return total

    def cut_distance(self):
        """
        Like distance(), but only counts cut distance.
        """
        total = 0.0

        last = None
        for cmd in self.commands:
            if cmd.is_travel_command():
                last = cmd.end
            elif cmd.is_cutting_command():
                # treating arcs as lines is probably good enough
                if last is not None:
                    total += math.dist(cmd.end, last)
                last = cmd.end
        return total

    def segments(self):
        segment = []
        for command in self.commands:
            if not segment:
                segment.append(command)
                continue

            if command.is_travel_command():
                yield segment
                segment = [command]

            elif command.is_cutting_command():
                segment.append(command)

            elif command.is_state_command():
                yield segment
                yield [command]
                segment = []

        if segment:
            yield segment

    def translate(self, dx: float, dy: float) -> Ops:
        """Translate geometric commands while preserving relative offsets"""
        for cmd in self.commands:
            if cmd.end is not None:
                # Translate endpoint only.
                # Arcs need no offset adjustment needed because
                # I/J are relative to start point
                x, y = cmd.end
                cmd.end = (x + dx, y + dy)

        # Update last known position
        last_x, last_y = self.last_move_to
        self.last_move_to = (last_x + dx, last_y + dy)
        return self

    def scale(self, sx: float, sy: float) -> Ops:
        """Scale both absolute positions and relative offsets"""
        for cmd in self.commands:
            if cmd.end is not None:
                x, y = cmd.end
                cmd.end = (x * sx, y * sy)

            if isinstance(cmd, ArcToCommand):
                # Scale relative offsets
                i, j = cmd.center_offset
                cmd.center_offset = (i * sx, j * sy)

        # Scale last known position
        last_x, last_y = self.last_move_to
        self.last_move_to = last_x * sx, last_y * sy
        return self

    def dump(self):
        for segment in self.segments():
            print(segment)

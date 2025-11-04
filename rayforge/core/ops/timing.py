"""
Timing estimation functionality for operations.

This module provides time estimation capabilities for laser cutting operations,
taking into account different speeds for cutting and travel movements,
as well as acceleration considerations.
"""

from typing import Optional, Tuple, Sequence
import math
from copy import copy
from .commands import MovingCommand, Command
from .container import State


def estimate_time(
    commands: Sequence[Command],
    default_cut_speed: float = 1000.0,
    default_travel_speed: float = 3000.0,
    acceleration: float = 1000.0,
) -> float:
    """
    Estimates the execution time of the operations in seconds.

    This function calculates the time required to execute all commands,
    taking into account different speeds for cutting and travel movements,
    as well as acceleration considerations.

    Args:
        commands: List of commands to estimate time for.
        default_cut_speed: Default cutting speed in mm/min if not specified
                           by state commands.
        default_travel_speed: Default travel speed in mm/min if not
                             specified by state commands.
        acceleration: Machine acceleration in mm/s² for more accurate
                     time estimation.

    Returns:
        The estimated execution time in seconds.
    """
    if not commands:
        return 0.0

    # Preload state for accurate time estimation
    state = State()
    for cmd in commands:
        if cmd.is_state_command():
            cmd.apply_to_state(state)
        elif not cmd.is_marker():
            cmd.state = copy(state)

    total_time = 0.0
    last_point: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    current_state = State()
    current_state.cut_speed = int(default_cut_speed)
    current_state.travel_speed = int(default_travel_speed)

    for cmd in commands:
        if cmd.is_state_command():
            cmd.apply_to_state(current_state)
            continue

        if not isinstance(cmd, MovingCommand) or cmd.end is None:
            continue

        # Calculate distance for this movement using the command's
        # distance method
        distance = cmd.distance(last_point)

        if distance < 1e-9:  # Skip negligible movements
            last_point = cmd.end
            continue

        # Determine speed based on movement type
        if cmd.is_cutting_command():
            speed = current_state.cut_speed or default_cut_speed
        else:  # Travel movement
            speed = current_state.travel_speed or default_travel_speed

        # Convert speed from mm/min to mm/s
        speed_mm_per_sec = speed / 60.0

        # Calculate time with acceleration consideration
        # Using a simple trapezoidal velocity profile
        if acceleration > 0:
            # Time to reach full speed
            accel_time = speed_mm_per_sec / acceleration
            # Distance covered during acceleration
            accel_distance = 0.5 * acceleration * accel_time**2

            if distance < 2 * accel_distance:
                # Can't reach full speed, triangular profile
                # t = 2 * sqrt(d / a)
                move_time = 2 * math.sqrt(distance / acceleration)
            else:
                # Trapezoidal profile with constant speed phase
                # t = 2 * t_accel + (d - 2 * d_accel) / v
                cruise_distance = distance - 2 * accel_distance
                cruise_time = cruise_distance / speed_mm_per_sec
                move_time = 2 * accel_time + cruise_time
        else:
            # Simple calculation without acceleration
            move_time = distance / speed_mm_per_sec

        total_time += move_time
        last_point = cmd.end

    return total_time

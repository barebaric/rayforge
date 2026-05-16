"""
Timing estimation functionality for operations.

This module provides time estimation capabilities for laser cutting operations,
taking into account different speeds for cutting and travel movements,
as well as acceleration considerations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .container import Ops

from .enums import CommandType, CommandCategory


def estimate_time(
    ops: Ops,
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
        ops: An Ops object whose commands to estimate time for.
        default_cut_speed: Default cutting speed in mm/min if not specified
                           by state commands.
        default_travel_speed: Default travel speed in mm/min if not
                             specified by state commands.
        acceleration: Machine acceleration in mm/s² for more accurate
                     time estimation.

    Returns:
        The estimated execution time in seconds.
    """
    if ops.len() == 0:
        return 0.0

    total_time = 0.0
    last_point = (0.0, 0.0, 0.0)
    cut_speed = default_cut_speed
    travel_speed = default_travel_speed

    for i in range(ops.len()):
        if ops.is_state(i):
            ct = ops.command_type(i)
            if ct == CommandType.SET_CUT_SPEED:
                cut_speed = ops.speed(i)
            elif ct == CommandType.SET_TRAVEL_SPEED:
                travel_speed = ops.speed(i)
            continue

        if ops.category(i) != CommandCategory.MOVING:
            continue

        end = ops.endpoint(i)
        distance = math.hypot(end[0] - last_point[0], end[1] - last_point[1])

        if distance < 1e-9:  # Skip negligible movements
            last_point = end
            continue

        # Determine speed based on movement type
        if ops.is_cutting(i):
            speed = cut_speed
        else:  # Travel movement
            speed = travel_speed

        # Convert speed from mm/min to mm/s
        speed_mm_per_sec = speed / 60.0

        # Calculate time with acceleration consideration
        # Using a simple trapezoidal velocity profile
        if acceleration > 0:
            # Time to reach full speed
            accel_time = speed_mm_per_sec / acceleration
            # Distance covered during acceleration
            accel_distance = 0.5 * acceleration * accel_time ** 2

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
        last_point = end

    return total_time

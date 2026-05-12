from __future__ import annotations
from enum import IntEnum, auto


class CommandType(IntEnum):
    MOVE_TO = 1
    LINE_TO = 2
    ARC_TO = 3
    BEZIER_TO = 6
    QUADRATIC_BEZIER_TO = 7
    SCAN_LINE = 4
    DWELL = 5
    SET_POWER = 10
    SET_CUT_SPEED = 11
    SET_TRAVEL_SPEED = 12
    SET_FREQUENCY = 16
    SET_PULSE_WIDTH = 17
    ENABLE_AIR_ASSIST = 13
    DISABLE_AIR_ASSIST = 14
    SET_LASER = 15
    JOB_START = 100
    JOB_END = 101
    LAYER_START = 102
    LAYER_END = 103
    WORKPIECE_START = 104
    WORKPIECE_END = 105
    OPS_SECTION_START = 106
    OPS_SECTION_END = 107


class CommandCategory(IntEnum):
    MOVING = auto()
    STATE = auto()
    MARKER = auto()

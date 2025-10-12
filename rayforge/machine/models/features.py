from enum import Enum, auto


class DriverFeature(Enum):
    """Feature flags for driver capabilities"""

    G0_WITH_SPEED = auto()
    """Driver supports speed parameter in G0 (rapid move) commands"""

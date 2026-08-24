from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SketchArrayMode(Enum):
    """Supported sketch pattern types."""

    CIRCULAR = "circular"


@dataclass
class CircularPatternParams:
    """User-facing parameters for a circular (polar) array."""

    count: int = 6
    total_angle_deg: float = 360.0
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    rotate_copies: bool = True

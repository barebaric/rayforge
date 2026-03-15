from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...geo import Point


@dataclass
class DimensionData:
    label: str
    position: Point
    leader_end: Optional[Point] = None

    @staticmethod
    def format_length(value: float) -> str:
        if abs(value) < 0.01:
            return "0.00"
        return f"{value:.2f}"

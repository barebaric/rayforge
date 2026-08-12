from dataclasses import dataclass

from raygeo.geo.types import Point


@dataclass
class DimensionData:
    label: str
    position: Point
    leader_end: Point | None = None

    @staticmethod
    def format_length(value: float) -> str:
        if abs(value) < 0.01:
            return "0.00"
        return f"{value:.2f}"

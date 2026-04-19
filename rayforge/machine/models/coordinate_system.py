from dataclasses import dataclass
from typing import Any, Dict

from ...core.geo import Point3D

_DEFAULT_WCS = ["G54", "G55", "G56", "G57", "G58", "G59"]
ZERO_OFFSET: Point3D = (0.0, 0.0, 0.0)


@dataclass
class CoordinateSystem:
    name: str
    label: str = ""
    offset: Point3D = ZERO_OFFSET

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name}
        if self.label:
            d["label"] = self.label
        d["offset"] = list(self.offset)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoordinateSystem":
        offset = tuple(data.get("offset", list(ZERO_OFFSET)))
        return cls(
            name=data["name"],
            label=data.get("label", ""),
            offset=offset,
        )

    @staticmethod
    def defaults() -> Dict[str, "CoordinateSystem"]:
        return {n: CoordinateSystem(name=n) for n in _DEFAULT_WCS}

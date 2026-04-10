from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from rayforge.core.ops.axis import Axis


class AxisType(Enum):
    LINEAR = "linear"
    ROTARY = "rotary"


class AxisDirection(Enum):
    NORMAL = "normal"
    REVERSED = "reversed"


@dataclass
class AxisConfig:
    letter: Axis
    axis_type: AxisType
    extents: Tuple[float, float]
    direction: AxisDirection = AxisDirection.NORMAL
    gcode_letter: Optional[str] = None
    resolution: float = 0.01
    rotary_diameter: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "letter": self.letter.name,
            "axis_type": self.axis_type.value,
            "extents": list(self.extents),
            "direction": self.direction.value,
            "resolution": self.resolution,
        }
        if self.gcode_letter is not None:
            result["gcode_letter"] = self.gcode_letter
        if self.rotary_diameter is not None:
            result["rotary_diameter"] = self.rotary_diameter
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AxisConfig":
        return cls(
            letter=Axis[data["letter"]],
            axis_type=AxisType(data["axis_type"]),
            extents=tuple(data["extents"]),
            direction=AxisDirection(
                data.get("direction", AxisDirection.NORMAL.value)
            ),
            gcode_letter=data.get("gcode_letter"),
            resolution=data.get("resolution", 0.01),
            rotary_diameter=data.get("rotary_diameter"),
        )


class AxisSet:
    def __init__(self, configs: List[AxisConfig]):
        self.configs = configs
        self.linear_axes = [
            c for c in configs if c.axis_type == AxisType.LINEAR
        ]
        self.rotary_axes = [
            c for c in configs if c.axis_type == AxisType.ROTARY
        ]

    def get(self, axis: Axis) -> Optional[AxisConfig]:
        for c in self.configs:
            if c.letter == axis:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configs": [c.to_dict() for c in self.configs],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AxisSet":
        configs = [AxisConfig.from_dict(c) for c in data["configs"]]
        return cls(configs)

    @classmethod
    def from_legacy(
        cls,
        axis_extents: Tuple[float, float],
        reverse_x: bool,
        reverse_y: bool,
        reverse_z: bool,
        rotary_modules=None,
    ) -> "AxisSet":
        width, height = axis_extents
        configs: List[AxisConfig] = [
            AxisConfig(
                letter=Axis.X,
                axis_type=AxisType.LINEAR,
                extents=(0, width),
                direction=(
                    AxisDirection.REVERSED
                    if reverse_x
                    else AxisDirection.NORMAL
                ),
            ),
            AxisConfig(
                letter=Axis.Y,
                axis_type=AxisType.LINEAR,
                extents=(0, height),
                direction=(
                    AxisDirection.REVERSED
                    if reverse_y
                    else AxisDirection.NORMAL
                ),
            ),
            AxisConfig(
                letter=Axis.Z,
                axis_type=AxisType.LINEAR,
                extents=(-50, 50),
                direction=(
                    AxisDirection.REVERSED
                    if reverse_z
                    else AxisDirection.NORMAL
                ),
            ),
        ]
        if rotary_modules:
            for module in rotary_modules.values():
                configs.append(
                    AxisConfig(
                        letter=module.axis,
                        axis_type=AxisType.ROTARY,
                        extents=(0, 360),
                        rotary_diameter=module.default_diameter,
                    )
                )
        return cls(configs)

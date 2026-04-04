from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class StepRenderConfig:
    rotary_enabled: bool
    rotary_diameter: float
    laser_uid: str

    def to_dict(self) -> dict:
        return {
            "rotary_enabled": self.rotary_enabled,
            "rotary_diameter": self.rotary_diameter,
            "laser_uid": self.laser_uid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepRenderConfig":
        return cls(
            rotary_enabled=data["rotary_enabled"],
            rotary_diameter=data["rotary_diameter"],
            laser_uid=data["laser_uid"],
        )


@dataclass
class RenderConfig3D:
    world_to_visual: np.ndarray
    world_to_cyl_local: np.ndarray
    step_configs: Dict[str, StepRenderConfig]
    default_color_lut_cut: bytes
    default_color_lut_engrave: bytes
    laser_color_luts: Dict[str, dict]
    zero_power_rgba: Tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "world_to_visual": self.world_to_visual.astype(
                np.float32
            ).tobytes(),
            "world_to_cyl_local": self.world_to_cyl_local.astype(
                np.float32
            ).tobytes(),
            "step_configs": {
                k: v.to_dict() for k, v in self.step_configs.items()
            },
            "default_color_lut_cut": self.default_color_lut_cut,
            "default_color_lut_engrave": self.default_color_lut_engrave,
            "laser_color_luts": self.laser_color_luts,
            "zero_power_rgba": list(self.zero_power_rgba),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RenderConfig3D":
        w2v = (
            np.frombuffer(data["world_to_visual"], dtype=np.float32)
            .reshape(4, 4)
            .copy()
        )
        w2c = (
            np.frombuffer(data["world_to_cyl_local"], dtype=np.float32)
            .reshape(4, 4)
            .copy()
        )
        step_configs = {
            k: StepRenderConfig.from_dict(v)
            for k, v in data["step_configs"].items()
        }
        return cls(
            world_to_visual=w2v,
            world_to_cyl_local=w2c,
            step_configs=step_configs,
            default_color_lut_cut=data["default_color_lut_cut"],
            default_color_lut_engrave=data["default_color_lut_engrave"],
            laser_color_luts=data["laser_color_luts"],
            zero_power_rgba=tuple(data["zero_power_rgba"]),
        )

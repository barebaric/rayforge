from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class LayerRenderConfig:
    rotary_enabled: bool
    rotary_diameter: float
    axis_position: float = 0.0
    gear_ratio: float = 1.0
    reverse: bool = False
    axis_position_3d: Optional[Tuple[float, ...]] = None
    cylinder_dir: Optional[Tuple[float, ...]] = None

    def to_dict(self) -> dict:
        d = {
            "rotary_enabled": self.rotary_enabled,
            "rotary_diameter": self.rotary_diameter,
            "axis_position": self.axis_position,
            "gear_ratio": self.gear_ratio,
            "reverse": self.reverse,
        }
        if self.axis_position_3d is not None:
            d["axis_position_3d"] = list(self.axis_position_3d)
        if self.cylinder_dir is not None:
            d["cylinder_dir"] = list(self.cylinder_dir)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "LayerRenderConfig":
        ap3d = data.get("axis_position_3d")
        if ap3d is not None:
            ap3d = tuple(ap3d)
        cdir = data.get("cylinder_dir")
        if cdir is not None:
            cdir = tuple(cdir)
        return cls(
            rotary_enabled=data["rotary_enabled"],
            rotary_diameter=data["rotary_diameter"],
            axis_position=data.get("axis_position", 0.0),
            gear_ratio=data.get("gear_ratio", 1.0),
            reverse=data.get("reverse", False),
            axis_position_3d=ap3d,
            cylinder_dir=cdir,
        )


@dataclass
class RenderConfig3D:
    world_to_visual: np.ndarray
    world_to_cyl_local: np.ndarray
    default_color_lut_cut: bytes
    default_color_lut_engrave: bytes
    laser_color_luts: Dict[str, dict]
    zero_power_rgba: Tuple[float, ...]
    layer_configs: Optional[Dict[str, LayerRenderConfig]] = None
    layer_color_luts: Optional[Dict[str, dict]] = None
    ops_color_mode: str = "laser"

    def to_dict(self) -> dict:
        d = {
            "world_to_visual": self.world_to_visual.astype(
                np.float32
            ).tobytes(),
            "world_to_cyl_local": self.world_to_cyl_local.astype(
                np.float32
            ).tobytes(),
            "default_color_lut_cut": self.default_color_lut_cut,
            "default_color_lut_engrave": self.default_color_lut_engrave,
            "laser_color_luts": self.laser_color_luts,
            "zero_power_rgba": list(self.zero_power_rgba),
            "ops_color_mode": (
                self.ops_color_mode
                if isinstance(self.ops_color_mode, str)
                else self.ops_color_mode.value
            ),
        }
        if self.layer_configs:
            d["layer_configs"] = {
                k: v.to_dict() for k, v in self.layer_configs.items()
            }
        if self.layer_color_luts:
            d["layer_color_luts"] = self.layer_color_luts
        return d

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
        layer_configs = None
        if "layer_configs" in data:
            layer_configs = {
                k: LayerRenderConfig.from_dict(v)
                for k, v in data["layer_configs"].items()
            }
        layer_color_luts = data.get("layer_color_luts")
        ops_color_mode = data.get("ops_color_mode", "laser")
        return cls(
            world_to_visual=w2v,
            world_to_cyl_local=w2c,
            default_color_lut_cut=data["default_color_lut_cut"],
            default_color_lut_engrave=data["default_color_lut_engrave"],
            laser_color_luts=data["laser_color_luts"],
            zero_power_rgba=tuple(data["zero_power_rgba"]),
            layer_configs=layer_configs,
            layer_color_luts=layer_color_luts,
            ops_color_mode=ops_color_mode,
        )

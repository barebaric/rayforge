from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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

    def to_dict(self) -> Dict:
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
    def from_dict(cls, data: Dict) -> "LayerRenderConfig":
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
    layer_configs: Optional[Dict[str, LayerRenderConfig]] = None

    def to_dict(self) -> Dict:
        d: Dict[str, object] = {
            "world_to_visual": self.world_to_visual.astype(
                np.float32
            ).tobytes(),
            "world_to_cyl_local": self.world_to_cyl_local.astype(
                np.float32
            ).tobytes(),
        }
        if self.layer_configs:
            d["layer_configs"] = {
                k: v.to_dict() for k, v in self.layer_configs.items()
            }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RenderConfig3D":
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
        return cls(
            world_to_visual=w2v,
            world_to_cyl_local=w2c,
            layer_configs=layer_configs,
        )

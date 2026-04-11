import uuid
from enum import Enum
from gettext import gettext as _
from typing import Any, Dict, Optional

import numpy as np
from blinker import Signal

from ...core.matrix import euler_rotation_matrix
from ...core.geo import Rect3D
from ...core.ops.axis import Axis


class RotaryMode(Enum):
    TRUE_4TH_AXIS = "true_4th_axis"
    AXIS_REPLACEMENT = "axis_replacement"


class RotaryModule:
    def __init__(self):
        self.uid: str = str(uuid.uuid4())
        self.name: str = _("Rotary Module")
        self.axis: Axis = Axis.A
        self.mode: RotaryMode = RotaryMode.TRUE_4TH_AXIS
        self.source_axis: Axis = Axis.Y
        self.mm_per_rotation: float = 0.0
        self.default_diameter: float = 25.0
        self.max_workpiece_length: float = 300.0
        self.model_id: Optional[str] = None
        self.transform: np.ndarray = np.eye(4, dtype=np.float64)
        self.changed = Signal()
        self.extra: Dict[str, Any] = {}

    def set_name(self, name: str):
        if self.name == name:
            return
        self.name = name
        self.changed.send(self)

    def set_axis(self, axis: Axis):
        axis.assert_single_axis()
        if self.axis == axis:
            return
        self.axis = axis
        self.changed.send(self)

    def set_mode(self, mode: RotaryMode):
        if self.mode == mode:
            return
        self.mode = mode
        self.changed.send(self)

    def set_source_axis(self, axis: Axis):
        axis.assert_single_axis()
        if self.source_axis == axis:
            return
        self.source_axis = axis
        self.changed.send(self)

    def set_mm_per_rotation(self, value: float):
        if self.mm_per_rotation == value:
            return
        self.mm_per_rotation = value
        self.changed.send(self)

    def set_position(self, x: float, y: float, z: float):
        if (
            self.transform[0, 3] == x
            and self.transform[1, 3] == y
            and self.transform[2, 3] == z
        ):
            return
        self.transform[0, 3] = x
        self.transform[1, 3] = y
        self.transform[2, 3] = z
        self.changed.send(self)

    def get_rotation(self):
        t = self.transform
        sx = float(np.linalg.norm(t[0, :3]))
        sy = float(np.linalg.norm(t[1, :3]))
        sz = float(np.linalg.norm(t[2, :3]))
        rx = np.degrees(np.arctan2(t[2, 1] / sy, t[2, 2] / sz))
        ry = np.degrees(
            np.arctan2(-t[2, 0] / sx, np.sqrt(t[2, 1] ** 2 + t[2, 2] ** 2))
        )
        rz = np.degrees(np.arctan2(t[1, 0] / sx, t[0, 0] / sx))
        return rx, ry, rz

    def set_rotation(self, rx: float, ry: float, rz: float):
        cur = self.get_rotation()
        if cur[0] == rx and cur[1] == ry and cur[2] == rz:
            return
        pos = self.transform[:3, 3].copy()
        scale = self.get_scale()
        self.transform[:3, :3] = euler_rotation_matrix(rx, ry, rz) * scale
        self.transform[:3, 3] = pos
        self.changed.send(self)

    def get_scale(self) -> float:
        return float(np.linalg.norm(self.transform[0, :3]))

    def set_scale(self, scale: float):
        if self.get_scale() == scale:
            return
        pos = self.transform[:3, 3].copy()
        rx, ry, rz = self.get_rotation()
        self.transform[:3, :3] = euler_rotation_matrix(rx, ry, rz) * scale
        self.transform[:3, 3] = pos
        self.changed.send(self)

    def set_default_diameter(self, diameter: float):
        if self.default_diameter == diameter:
            return
        self.default_diameter = diameter
        self.changed.send(self)

    def set_max_workpiece_length(self, length: float):
        if self.max_workpiece_length == length:
            return
        self.max_workpiece_length = length
        self.changed.send(self)

    def set_model_id(self, model_id: Optional[str]):
        if self.model_id == model_id:
            return
        self.model_id = model_id
        self.changed.send(self)

    def get_collision_bbox(self) -> Optional[Rect3D]:
        return None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "uid": self.uid,
            "name": self.name,
            "axis": self.axis.name,
            "mode": self.mode.value,
            "source_axis": self.source_axis.name,
            "default_diameter": self.default_diameter,
            "max_workpiece_length": self.max_workpiece_length,
            "model_id": self.model_id,
            "transform": self.transform.flatten().tolist(),
        }
        if self.mm_per_rotation > 0:
            result["mm_per_rotation"] = self.mm_per_rotation
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RotaryModule":
        known_keys = {
            "uid",
            "name",
            "axis",
            "mode",
            "source_axis",
            "mm_per_rotation",
            "default_diameter",
            "max_workpiece_length",
            "model_id",
            "transform",
            "x",
            "y",
            "z",
            "length",
            "chuck_diameter",
            "tailstock_diameter",
            "max_height",
            "model_path",
            "viz_mode",
            "viz_scale",
            "viz_rotation",
            "chuck_height",
            "tailstock_height",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        rm = cls()
        rm.uid = data.get("uid", str(uuid.uuid4()))
        rm.name = data.get("name", _("Rotary Module"))
        rm.axis = Axis[data.get("axis", "A")]
        rm.mode = RotaryMode(data.get("mode", "true_4th_axis"))
        rm.source_axis = Axis[data.get("source_axis", "Y")]
        rm.mm_per_rotation = data.get("mm_per_rotation", 0.0)
        rm.default_diameter = data.get("default_diameter", 25.0)
        rm.max_workpiece_length = data.get("max_workpiece_length", 300.0)

        rm.model_id = data.get("model_id")
        if rm.model_id is None and "model_path" in data:
            raw_model_path = data["model_path"]
            if raw_model_path:
                rm.model_id = raw_model_path

        raw_transform = data.get("transform")
        if raw_transform is not None:
            rm.transform = np.array(raw_transform, dtype=np.float64).reshape(
                4, 4
            )
        elif any(k in data for k in ("x", "y", "z")):
            rm.transform[0, 3] = data.get("x", 0.0)
            rm.transform[1, 3] = data.get("y", 0.0)
            rm.transform[2, 3] = data.get("z", 0.0)

        rm.extra = extra
        return rm

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("changed", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.changed = Signal()

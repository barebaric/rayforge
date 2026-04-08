import uuid
from enum import Enum
from gettext import gettext as _
from typing import Any, Dict

from blinker import Signal

from ...core.geo.arc import arc_intersects_circle, arc_intersects_rect
from ...core.geo.circle import line_segment_intersects_circle
from ...core.geo.primitives import line_segment_intersects_rect
from ...core.geo.types import Point


class ZoneShape(Enum):
    RECT = "rect"
    BOX = "box"
    CYLINDER = "cylinder"


class Zone:
    def __init__(self):
        self.uid: str = str(uuid.uuid4())
        self.name: str = _("No-Go Zone")
        self.shape: ZoneShape = ZoneShape.RECT
        self.params: Dict[str, float] = {
            "x": 0.0,
            "y": 0.0,
            "w": 10.0,
            "h": 10.0,
        }
        self.enabled: bool = True
        self.changed = Signal()
        self.extra: Dict[str, Any] = {}

    def set_name(self, name: str):
        if self.name == name:
            return
        self.name = name
        self.changed.send(self)

    def set_shape(self, shape: ZoneShape):
        if self.shape == shape:
            return
        self.shape = shape
        if shape == ZoneShape.BOX:
            self.params.setdefault("z", 0.0)
            self.params.setdefault("d", 10.0)
            self.params.setdefault("w", self.params.get("w", 10.0))
            self.params.setdefault("h", self.params.get("h", 10.0))
        elif shape == ZoneShape.CYLINDER:
            self.params.setdefault("z", 0.0)
            self.params.setdefault("radius", 5.0)
            self.params.setdefault("height", 10.0)
        self.changed.send(self)

    def set_param(self, key: str, value: float):
        if self.params.get(key) == value:
            return
        self.params[key] = value
        self.changed.send(self)

    def set_enabled(self, enabled: bool):
        if self.enabled == enabled:
            return
        self.enabled = enabled
        self.changed.send(self)

    def _get_rect(self):
        x = self.params.get("x", 0.0)
        y = self.params.get("y", 0.0)
        w = self.params.get("w", 0.0)
        h = self.params.get("h", 0.0)
        return (x, y, x + w, y + h)

    def _get_circle(self):
        cx = self.params.get("x", 0.0)
        cy = self.params.get("y", 0.0)
        r = self.params.get("radius", 5.0)
        return (cx, cy), r

    def collides_with_line(self, start: Point, end: Point) -> bool:
        colliders = {
            ZoneShape.RECT: self._collide_line_rect,
            ZoneShape.BOX: self._collide_line_rect,
            ZoneShape.CYLINDER: self._collide_line_circle,
        }
        collider = colliders.get(self.shape)
        return collider(start, end) if collider else False

    def collides_with_arc(self, start, end, center, clockwise):
        colliders = {
            ZoneShape.RECT: self._collide_arc_rect,
            ZoneShape.BOX: self._collide_arc_rect,
            ZoneShape.CYLINDER: self._collide_arc_circle,
        }
        collider = colliders.get(self.shape)
        if collider is None:
            return False
        return collider(start, end, center, clockwise)

    def _collide_line_rect(self, start, end):
        return line_segment_intersects_rect(start, end, self._get_rect())

    def _collide_line_circle(self, start, end):
        center, radius = self._get_circle()
        return line_segment_intersects_circle(start, end, center, radius)

    def _collide_arc_rect(self, start, end, center, clockwise):
        return arc_intersects_rect(
            start, end, center, clockwise, self._get_rect()
        )

    def _collide_arc_circle(self, start, end, center, clockwise):
        cc, cr = self._get_circle()
        return arc_intersects_circle(start, end, center, clockwise, cc, cr)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "uid": self.uid,
            "name": self.name,
            "shape": self.shape.value,
            "params": dict(self.params),
            "enabled": self.enabled,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Zone":
        known_keys = {
            "uid",
            "name",
            "shape",
            "params",
            "enabled",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        zone = cls()
        zone.uid = data.get("uid", str(uuid.uuid4()))
        zone.name = data.get("name", _("No-Go Zone"))
        zone.shape = ZoneShape(data.get("shape", "rect"))
        zone.params = dict(data.get("params", zone.params))
        zone.enabled = data.get("enabled", True)
        zone.extra = extra
        return zone

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("changed", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.changed = Signal()


def check_ops_collides_with_zones(ops, zones):
    pos = None
    for cmd in ops.commands:
        if getattr(cmd, "end", None) is not None and pos is not None:
            start_2d = (pos[0], pos[1])
            end_2d = (cmd.end[0], cmd.end[1])
            for zone in zones.values():
                if hasattr(cmd, "center_offset"):
                    arc_center = (
                        start_2d[0] + cmd.center_offset[0],
                        start_2d[1] + cmd.center_offset[1],
                    )
                    if zone.collides_with_arc(
                        start_2d,
                        end_2d,
                        arc_center,
                        cmd.clockwise,
                    ):
                        return True
                else:
                    if zone.collides_with_line(start_2d, end_2d):
                        return True
        if getattr(cmd, "end", None) is not None:
            pos = cmd.end
    return False

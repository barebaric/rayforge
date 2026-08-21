import uuid
from abc import ABC
from collections.abc import Callable
from gettext import gettext as _
from typing import Any, Self, overload

import numpy as np
from blinker import Signal

from ...core.capability import MachineCapability
from ...core.matrix import euler_rotation_matrix

HEAD_TYPE_KEY = "type"

_HEAD_SERIALIZED_KEYS = frozenset(
    {HEAD_TYPE_KEY, "uid", "name", "tool_number", "model_path", "transform"}
)


class head_setting:
    """
    Descriptor declaring a reviewable head attribute.

    Declaring a class attribute with this marker makes the attribute
    participate in device-profile reviews: ``label`` names the setting
    in the review UI, and the optional converters translate between the
    instance value and the serialized (profile YAML) domain. Accessors
    are attached at the attribute's definition site, so there is no
    central registry to keep in sync.
    """

    def __init__(
        self,
        label: str,
        to_yaml: Callable[[Any], Any] | None = None,
        from_yaml: Callable[[Any], Any] | None = None,
    ):
        self.label = label
        self._to_yaml = to_yaml or (lambda v: v)
        self._from_yaml = from_yaml or (lambda v: v)

    def __set_name__(self, owner, name):
        self.key = name

    @overload
    def __get__(self, obj: None, objtype: Any = None) -> "head_setting": ...

    @overload
    def __get__(self, obj: Any, objtype: Any = None) -> Any: ...

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.key)

    def __set__(self, obj, value):
        obj.__dict__[self.key] = value

    def yaml_value(self, obj) -> Any:
        """The attribute's value in the serialized domain."""
        return self._to_yaml(obj.__dict__[self.key])

    def set_yaml_value(self, obj, yaml_value):
        """Sets the attribute from a serialized-domain value."""
        obj.__dict__[self.key] = self._from_yaml(yaml_value)
        obj.changed.send(obj)


def head_settings(head: "Head") -> list[head_setting]:
    """
    All :class:`head_setting` descriptors declared on *head*'s class
    hierarchy, base-class attributes first.
    """
    found: dict[str, head_setting] = {}
    for klass in reversed(type(head).__mro__):
        for value in vars(klass).values():
            if isinstance(value, head_setting):
                found[value.key] = value
    return list(found.values())


class Head(ABC):
    """
    Base class for machine heads.

    Concrete head types (:class:`LaserHead`, :class:`SpindleHead`, ...)
    add their type-specific attributes and declare which machine
    capability they imply via :attr:`machine_capability`.
    """

    # Serialization key identifying the concrete head type.
    HEAD_TYPE: str = "Head"

    name = head_setting(_("Name"))
    tool_number = head_setting(_("Tool Number"))

    def __init__(self):
        self.uid: str = str(uuid.uuid4())
        self.name = _("Head")
        self.tool_number = 0
        self.model_path: str | None = None
        self.transform: np.ndarray = np.eye(4, dtype=np.float64)
        self.changed = Signal()
        self.extra: dict[str, Any] = {}

    @property
    def machine_capability(self) -> MachineCapability | None:
        """
        The machine capability this head type implies, or ``None`` for
        passive heads that contribute no capability.
        """
        return None

    def set_name(self, name: str):
        self.name = name
        self.changed.send(self)

    def set_tool_number(self, tool_number: int):
        self.tool_number = tool_number
        self.changed.send(self)

    def set_model_path(self, model_path: str | None):
        if self.model_path == model_path:
            return
        self.model_path = model_path
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

    def to_dict(self) -> dict[str, Any]:
        result = {
            HEAD_TYPE_KEY: self.HEAD_TYPE,
            "uid": self.uid,
            "name": self.name,
            "tool_number": self.tool_number,
            "model_path": self.model_path,
            "transform": self.transform.flatten().tolist(),
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        head = cls()
        head.uid = data.get("uid", str(uuid.uuid4()))
        head.name = data.get("name", head.name)
        head.tool_number = data.get("tool_number", head.tool_number)
        head.model_path = data.get("model_path")
        raw_transform = data.get("transform")
        if raw_transform is not None:
            head.transform = np.array(raw_transform, dtype=np.float64).reshape(
                4, 4
            )
        return head

    def __getstate__(self):
        """Prepare the object for pickling. Removes unpickleable Signal."""
        state = self.__dict__.copy()
        state.pop("changed", None)
        return state

    def __setstate__(self, state):
        """Restore the object after unpickling. Recreates the Signal."""
        vars(self).update(state)
        self.changed = Signal()


def head_from_dict(
    data: dict[str, Any],
) -> "Head":
    """
    Deserialize a head dict, dispatching on the ``type`` key.

    Old machine files without a ``type`` key map to :class:`LaserHead`
    for backward compatibility.
    """
    from .laser import LaserHead
    from .spindle import SpindleHead

    head_type = data.get(HEAD_TYPE_KEY)
    if head_type == SpindleHead.HEAD_TYPE:
        return SpindleHead.from_dict(data)
    return LaserHead.from_dict(data)

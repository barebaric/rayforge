from gettext import gettext as _
from typing import Any

from ...core.capability import MachineCapability
from .head import _HEAD_SERIALIZED_KEYS, Head, head_setting


class DragKnifeHead(Head):
    """
    A swiveling drag-knife holder.

    The blade tip trails the machine's controlled pivot point by
    ``offset_mm``; implies the :attr:`MachineCapability.DRAG_KNIFE`
    machine capability.
    """

    HEAD_TYPE: str = "DragKnifeHead"

    offset_mm = head_setting(
        _("Blade Offset"),
        to_yaml=float,
        from_yaml=float,
    )

    def __init__(self):
        super().__init__()
        self.name = _("Drag Knife")
        self.offset_mm = 0.5

    @property
    def machine_capability(self) -> MachineCapability:
        return MachineCapability.DRAG_KNIFE

    def set_offset_mm(self, offset_mm: float):
        if self.offset_mm == offset_mm:
            return
        self.offset_mm = max(0.0, float(offset_mm))
        self.changed.send(self)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({"offset_mm": self.offset_mm})
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DragKnifeHead":
        known_keys = _HEAD_SERIALIZED_KEYS | {"offset_mm"}
        extra = {k: v for k, v in data.items() if k not in known_keys}

        head = super().from_dict(data)
        head.offset_mm = float(data.get("offset_mm", head.offset_mm))
        head.extra = extra
        return head


class TangentialKnifeHead(Head):
    """
    A motor-driven knife that rotates to stay tangent to the
    toolpath; implies the
    :attr:`MachineCapability.TANGENTIAL_KNIFE` machine capability.
    """

    HEAD_TYPE: str = "TangentialKnifeHead"

    def __init__(self):
        super().__init__()
        self.name = _("Tangential Knife")

    @property
    def machine_capability(self) -> MachineCapability:
        return MachineCapability.TANGENTIAL_KNIFE

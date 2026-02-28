from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any
import uuid
from gettext import gettext as _


class MacroTrigger(Enum):
    """Defines events in the job lifecycle where G-code can be injected."""

    LAYER_START = "Before processing a layer"
    LAYER_END = "After processing a layer"
    WORKPIECE_START = "Before processing a workpiece"
    WORKPIECE_END = "After processing a workpiece"

    def label(self) -> str:
        """Return a translatable label for this trigger."""
        labels = {
            self.LAYER_START: _("Layer Start"),
            self.LAYER_END: _("Layer End"),
            self.WORKPIECE_START: _("Workpiece Start"),
            self.WORKPIECE_END: _("Workpiece End"),
        }
        return labels[self]

    def description(self) -> str:
        """Return a translatable description for this trigger."""
        labels = {
            self.LAYER_START: _("Before processing a layer"),
            self.LAYER_END: _("After processing a layer"),
            self.WORKPIECE_START: _("Before processing a workpiece"),
            self.WORKPIECE_END: _("After processing a workpiece"),
        }
        return labels[self]


@dataclass
class Macro:
    """A generic, named block of G-code with an enabled state."""

    name: str = ""
    code: List[str] = field(default_factory=list)
    enabled: bool = True
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the macro to a dictionary."""
        result = {
            "uid": self.uid,
            "name": self.name,
            "code": self.code,
            "enabled": self.enabled,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Macro":
        """Creates a macro instance from a dictionary."""
        known_keys = {"uid", "name", "code", "enabled"}
        extra = {k: v for k, v in data.items() if k not in known_keys}
        instance = cls(
            uid=data.get("uid", str(uuid.uuid4())),
            name=data.get("name", _("Unnamed Macro")),
            code=data.get("code", []),
            enabled=data.get("enabled", True),
        )
        instance.extra = extra
        return instance

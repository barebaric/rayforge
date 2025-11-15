from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any
import uuid


class MacroTrigger(Enum):
    """Defines events in the job lifecycle where G-code can be injected."""

    LAYER_START = "Before processing a layer"
    LAYER_END = "After processing a layer"
    WORKPIECE_START = "Before processing a workpiece"
    WORKPIECE_END = "After processing a workpiece"


@dataclass
class Macro:
    """A generic, named block of G-code with an enabled state."""

    name: str = ""
    code: List[str] = field(default_factory=list)
    enabled: bool = True
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the macro to a dictionary."""
        return {
            "uid": self.uid,
            "name": self.name,
            "code": self.code,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Macro":
        """Creates a macro instance from a dictionary."""
        return cls(
            uid=data.get("uid", str(uuid.uuid4())),
            name=data.get("name", _("Unnamed Macro")),
            code=data.get("code", []),
            enabled=data.get("enabled", True),
        )

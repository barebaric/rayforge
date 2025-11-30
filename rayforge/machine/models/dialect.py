from __future__ import annotations
import uuid
from dataclasses import dataclass, field, asdict, replace
from typing import List, Dict, Optional, Any
from ...core.varset import VarSet, Var, TextAreaVar


_DIALECT_REGISTRY: Dict[str, "GcodeDialect"] = {}


def register_dialect(dialect: "GcodeDialect"):
    """
    Adds a dialect to the central registry, keyed by its case-insensitive
    unique `uid`.
    """
    uid_key = dialect.uid.lower()
    if uid_key in _DIALECT_REGISTRY:
        raise ValueError(
            f"Dialect with UID '{dialect.uid}' is already registered."
        )
    _DIALECT_REGISTRY[uid_key] = dialect


def get_dialect(uid: str) -> "GcodeDialect":
    """
    Retrieves a GcodeDialect instance from the registry by its
    case-insensitive UID.
    """
    dialect = _DIALECT_REGISTRY.get(uid.lower())
    if not dialect:
        raise ValueError(f"Unknown or unsupported G-code dialect UID: '{uid}'")
    return dialect


def get_available_dialects() -> List["GcodeDialect"]:
    """Returns a list of all registered GcodeDialect instances."""
    # Sort by display name for consistent UI presentation
    return sorted(_DIALECT_REGISTRY.values(), key=lambda d: d.label)


@dataclass
class GcodeDialect:
    """
    A container for G-code command templates and formatting logic for a
    specific hardware dialect (e.g., GRBL, Marlin, Smoothieware).
    """

    label: str  # User-facing name for UI (e.g., "GRBL")
    description: str

    # Command Templates
    laser_on: str
    laser_off: str
    tool_change: str
    set_speed: str
    travel_move: str
    linear_move: str
    arc_cw: str
    arc_ccw: str

    # Air Assist Control
    air_assist_on: str
    air_assist_off: str

    # Preamble & Postscript
    preamble: List[str] = field(default_factory=list)
    postscript: List[str] = field(default_factory=list)

    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_custom: bool = False
    parent_uid: Optional[str] = None

    def get_editor_varsets(self) -> Dict[str, VarSet]:
        """
        Returns a dictionary of VarSets that define the editable fields for
        this dialect, serving as the single source of truth for the UI.
        """
        info_vs = VarSet(title=_("General Information"))
        info_vs.add(
            Var(
                "label",
                _("Label"),
                str,
                _("User-facing name"),
                value=self.label,
            )
        )
        info_vs.add(
            Var(
                "description",
                _("Description"),
                str,
                _("Short description"),
                value=self.description,
            )
        )

        templates_vs = VarSet(title=_("Command Templates"))
        template_fields = [
            ("laser_on", _("Laser On")),
            ("laser_off", _("Laser Off")),
            ("travel_move", _("Travel Move")),
            ("linear_move", _("Linear Move")),
            ("arc_cw", _("Arc (CW)")),
            ("arc_ccw", _("Arc (CCW)")),
            ("tool_change", _("Tool Change")),
            ("set_speed", _("Set Speed")),
            ("air_assist_on", _("Air On")),
            ("air_assist_off", _("Air Off")),
        ]
        for key, label in template_fields:
            templates_vs.add(Var(key, label, str, value=getattr(self, key)))

        scripts_vs = VarSet(title=_("Scripts"))
        scripts_vs.add(
            TextAreaVar(
                "preamble",
                _("Preamble"),
                description=_("Preamble script"),
                value="\n".join(self.preamble),
            )
        )
        scripts_vs.add(
            TextAreaVar(
                "postscript",
                _("Postscript"),
                description=_("Postscript script"),
                value="\n".join(self.postscript),
            )
        )

        return {
            "info": info_vs,
            "templates": templates_vs,
            "scripts": scripts_vs,
        }

    def copy_as_custom(self, new_label: str) -> "GcodeDialect":
        """
        Creates a new, custom dialect instance from this one, generating a
        new UID.
        """
        return replace(
            self,
            uid=str(uuid.uuid4()),  # Explicitly generate a new UID
            is_custom=True,
            parent_uid=self.uid,
            label=new_label,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the dialect to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GcodeDialect":
        """Creates a dialect instance from a dictionary."""
        return cls(
            label=data.get("label", _("Unnamed Dialect")),
            description=data.get("description", ""),
            laser_on=data.get("laser_on", ""),
            laser_off=data.get("laser_off", ""),
            tool_change=data.get("tool_change", ""),
            set_speed=data.get("set_speed", ""),
            travel_move=data.get("travel_move", ""),
            linear_move=data.get("linear_move", ""),
            arc_cw=data.get("arc_cw", ""),
            arc_ccw=data.get("arc_ccw", ""),
            air_assist_on=data.get("air_assist_on", ""),
            air_assist_off=data.get("air_assist_off", ""),
            preamble=data.get("preamble", []),
            postscript=data.get("postscript", []),
            uid=data.get("uid", str(uuid.uuid4())),
            is_custom=data.get("is_custom", False),
            parent_uid=data.get("parent_uid"),
        )

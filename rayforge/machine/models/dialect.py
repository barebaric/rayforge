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

    # Machine Control Commands
    home_all: str
    home_axis: str
    move_to: str
    jog: str
    clear_alarm: str
    set_wcs_offset: str
    probe_cycle: str

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
            ("home_all", _("Home All")),
            ("home_axis", _("Home Axis")),
            ("move_to", _("Move To")),
            ("jog", _("Jog")),
            ("clear_alarm", _("Clear Alarm")),
            ("set_wcs_offset", _("Set WCS Offset")),
            ("probe_cycle", _("Probe Cycle")),
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
        """
        Creates a dialect instance from a dictionary, correctly handling
        missing fields by inheriting from the parent dialect.
        """
        # 1. Determine the base dialect to inherit defaults from
        parent_uid = data.get("parent_uid")
        base_dialect = None

        if parent_uid:
            # Try to find the specific parent in the loaded registry
            base_dialect = _DIALECT_REGISTRY.get(parent_uid.lower())

        if not base_dialect:
            # If parent not found (or not specified), try to find generic GRBL
            base_dialect = _DIALECT_REGISTRY.get("grbl")

        # 2. Establish defaults
        if base_dialect:
            defaults = asdict(base_dialect)
        else:
            # If registry is empty (bootstrapping/testing), import the
            # built-in GRBL dialect locally to avoid circular imports.
            from .dialect_builtins import GRBL_DIALECT

            defaults = asdict(GRBL_DIALECT)

        # 3. Merge data on top of defaults.
        # This ensures that any key missing in 'data' gets the value from
        # 'defaults' (the parent/base dialect).
        merged_data = defaults.copy()
        merged_data.update(data)

        # 4. Ensure identity fields are correct (don't inherit UID/custom flag)
        merged_data["uid"] = data.get("uid", str(uuid.uuid4()))
        merged_data["is_custom"] = data.get("is_custom", False)
        merged_data["parent_uid"] = data.get("parent_uid")

        return cls(**merged_data)

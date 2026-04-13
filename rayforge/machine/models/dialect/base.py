from __future__ import annotations

import logging
import uuid
from dataclasses import (
    dataclass,
    field,
    asdict,
    fields,
    MISSING,
    replace,
)
from typing import List, Dict, Optional, Any
from gettext import gettext as _

from ....core.varset import VarSet, Var, TextAreaVar, BoolVar

logger = logging.getLogger(__name__)


@dataclass
class GcodeDialect:
    """
    A container for G-code command templates and formatting logic for a
    specific hardware dialect (e.g., GRBL, Marlin, Smoothieware).
    """

    label: str = field(metadata={"template_meta": True})
    description: str = field(metadata={"template_meta": True})

    laser_on: str
    laser_off: str
    focus_laser_on: str
    tool_change: str
    set_speed: str
    travel_move: str
    linear_move: str
    arc_cw: str
    arc_ccw: str
    bezier_cubic: str

    air_assist_on: str
    air_assist_off: str

    home_all: str
    home_axis: str
    move_to: str
    jog: str
    clear_alarm: str
    set_wcs_offset: str
    probe_cycle: str
    dwell: str = "G4 P{seconds:.3f}"

    preamble: List[str] = field(default_factory=list)
    postscript: List[str] = field(default_factory=list)

    inject_wcs_after_preamble: bool = True
    can_g0_with_speed: bool = False
    omit_unchanged_coords: bool = True
    continuous_laser_mode: bool = False
    modal_feedrate: bool = False

    uid: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"template_meta": True},
    )
    is_custom: bool = field(default=False, metadata={"template_meta": True})
    parent_uid: Optional[str] = field(
        default=None, metadata={"template_meta": True}
    )
    extra: Dict[str, Any] = field(
        default_factory=dict, metadata={"template_meta": True}
    )

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

        settings_vs = VarSet(title=_("Settings"))
        settings_vs.add(
            BoolVar(
                "omit_unchanged_coords",
                default=self.omit_unchanged_coords,
                label=_("Omit unchanged coordinates"),
                description=_(
                    "When enabled, axis letters that haven't changed "
                    "are omitted from G0/G1 commands"
                ),
            )
        )
        settings_vs.add(
            BoolVar(
                "continuous_laser_mode",
                default=self.continuous_laser_mode,
                label=_("Continuous laser mode"),
                description=_(
                    "Keeps M4 dynamic power mode continuously active "
                    "during raster engraving instead of toggling "
                    "M4/M5 between each segment"
                ),
            )
        )
        settings_vs.add(
            BoolVar(
                "modal_feedrate",
                default=self.modal_feedrate,
                label=_("Modal feedrate"),
                description=_(
                    "Only include the F feedrate parameter in motion "
                    "commands when it changes from the previous value"
                ),
            )
        )

        templates_vs = VarSet(title=_("Command Templates"))
        template_fields = [
            ("laser_on", _("Laser On")),
            ("laser_off", _("Laser Off")),
            ("focus_laser_on", _("Focus Laser On")),
            ("travel_move", _("Travel Move")),
            ("linear_move", _("Linear Move")),
            ("arc_cw", _("Arc (CW)")),
            ("arc_ccw", _("Arc (CCW)")),
            ("bezier_cubic", _("Bezier Cubic")),
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
            ("dwell", _("Dwell")),
        ]
        for key, label in template_fields:
            templates_vs.add(Var(key, label, str, value=getattr(self, key)))

        scripts_vs = VarSet(title=_("Scripts"))
        scripts_vs.add(
            BoolVar(
                "inject_wcs_after_preamble",
                default=self.inject_wcs_after_preamble,
                label=_("Inject WCS after Preamble"),
                description=_(
                    "Inject the active WCS command (e.g., G54) after "
                    "the preamble script. When disabled, you can use "
                    "{machine.active_wcs} in the preamble instead."
                ),
            )
        )
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
            "settings": settings_vs,
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
            uid=str(uuid.uuid4()),
            is_custom=True,
            parent_uid=self.uid,
            label=new_label,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the dialect to a dictionary."""
        result = asdict(self)
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        registry: Optional[Dict[str, "GcodeDialect"]] = None,
    ) -> "GcodeDialect":
        """
        Creates a dialect instance from a dictionary, correctly handling
        missing fields by inheriting from parent dialect.
        """
        parent_uid = data.get("parent_uid")
        base_dialect = None

        if parent_uid and registry:
            base_dialect = registry.get(parent_uid.lower())

        if not base_dialect and registry:
            base_dialect = registry.get("grbl")

        if not base_dialect:
            from .grbl import GRBL_DIALECT

            base_dialect = GRBL_DIALECT

        defaults = asdict(base_dialect)

        merged_data = defaults.copy()
        merged_data.update(data)

        merged_data["uid"] = data.get("uid", str(uuid.uuid4()))
        merged_data["is_custom"] = data.get("is_custom", False)
        merged_data["parent_uid"] = data.get("parent_uid")

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {
            k: v for k, v in merged_data.items() if k in valid_fields
        }

        extra = {k: v for k, v in merged_data.items() if k not in valid_fields}

        instance = cls(**filtered_data)
        instance.extra = extra
        return instance

    @staticmethod
    def _template_meta_fields() -> frozenset:
        """Meta fields excluded from template serialization."""
        return frozenset(
            f.name
            for f in fields(GcodeDialect)
            if f.metadata.get("template_meta")
        )

    @classmethod
    def _template_field_sets(
        cls,
    ) -> tuple[frozenset, frozenset, frozenset]:
        meta = cls._template_meta_fields()
        required = set()
        optional = set()
        for f in fields(cls):
            if f.name in meta:
                continue
            if f.default is not MISSING:
                optional.add(f.name)
            else:
                required.add(f.name)
        return (
            frozenset(required),
            frozenset(optional),
            frozenset(required | optional),
        )

    def to_template_dict(self) -> Dict[str, Any]:
        """
        Serialize template fields for device profile export.

        Excludes meta fields (marked with ``template_meta`` metadata)
        like ``uid``, ``label``, ``is_custom``, etc.
        """
        meta = self._template_meta_fields()
        result: Dict[str, Any] = {}
        for f in fields(self):
            if f.name not in meta:
                result[f.name] = getattr(self, f.name)
        return result

    @classmethod
    def validate_template_dict(
        cls,
        data: Dict[str, Any],
        source: str = "",
    ):
        """
        Validate that *data* contains all required template fields.

        Raises :class:`ValueError` on missing required fields.
        Logs warnings for unknown keys.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Invalid dialect data: {source}")
        required, _, all_fields = cls._template_field_sets()
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"Missing required dialect fields in {source}: "
                f"{sorted(missing)}"
            )
        for key in data:
            if key not in all_fields:
                logger.warning(f"Unknown dialect field '{key}' in {source}")

    @classmethod
    def from_template_dict(
        cls, data: Dict[str, Any], **overrides
    ) -> "GcodeDialect":
        """
        Create a dialect from a device profile template dict.

        Filters out meta fields from *data*, applies *overrides*,
        then constructs with only valid field names.
        """
        filtered = {
            k: v
            for k, v in data.items()
            if k not in cls._template_meta_fields()
        }
        filtered.update(overrides)
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in filtered.items() if k in valid})

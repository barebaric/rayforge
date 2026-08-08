"""
Tool dataclass: a rayforge-side wrapper around the raygeo
:class:`~raygeo.cnc.tool.Tool` that adds the persistence/identity
fields rayforge needs (``uid``, ``name``, ``max_rpm``).

The geometry itself (diameter, flute count, ...) lives in the raygeo
``ToolModel`` param bag; this wrapper only adds scalar bookkeeping and
YAML round-trip.
"""

import uuid
from dataclasses import dataclass
from gettext import gettext as _
from typing import Any

from raygeo.cnc.tool import (
    ToolCategory,
    ToolMaterial,
    ToolModel,
)

CATEGORY_NAMES = [
    "END_MILL",
    "BALL_NOSE",
    "BULL_NOSE",
    "CHAMFER",
    "DRILL",
    "PROBE",
    "VBIT",
    "SLITTING_SAW",
    "REAMER",
    "TAP",
    "THREAD_MILL",
    "DOVETAIL",
]
TOOL_MATERIAL_NAMES = [
    "CARBIDE",
    "HSS",
    "HSSE",
    "DIAMOND",
    "CBN",
    "CERAMIC",
]

CATEGORY_BY_NAME: dict[str, ToolCategory] = {
    name: getattr(ToolCategory, name) for name in CATEGORY_NAMES
}
TOOL_MATERIAL_BY_NAME: dict[str, ToolMaterial] = {
    name: getattr(ToolMaterial, name) for name in TOOL_MATERIAL_NAMES
}

CATEGORY_LABELS: dict[str, str] = {
    "END_MILL": _("End Mill"),
    "BALL_NOSE": _("Ball Nose"),
    "BULL_NOSE": _("Bull Nose"),
    "CHAMFER": _("Chamfer"),
    "DRILL": _("Drill"),
    "PROBE": _("Probe"),
    "VBIT": _("V-Bit"),
    "SLITTING_SAW": _("Slitting Saw"),
    "REAMER": _("Reamer"),
    "TAP": _("Tap"),
    "THREAD_MILL": _("Thread Mill"),
    "DOVETAIL": _("Dovetail"),
}
TOOL_MATERIAL_LABELS: dict[str, str] = {
    "CARBIDE": _("Carbide"),
    "HSS": _("HSS"),
    "HSSE": _("HSSE"),
    "DIAMOND": _("Diamond"),
    "CBN": _("CBN"),
    "CERAMIC": _("Ceramic"),
}


def category_to_name(category: ToolCategory) -> str:
    """Return the canonical name of a ``ToolCategory`` member."""
    for name, member in CATEGORY_BY_NAME.items():
        if member == category:
            return name
    return CATEGORY_NAMES[0]


def tool_material_to_name(tool_material: ToolMaterial) -> str:
    """Return the canonical name of a ``ToolMaterial`` member."""
    for name, member in TOOL_MATERIAL_BY_NAME.items():
        if member == tool_material:
            return name
    return TOOL_MATERIAL_NAMES[0]


@dataclass(frozen=True)
class ParamSpec:
    """
    One editable geometry parameter for a tool category.

    ``quantity="length"`` fields are edited through the app's
    :class:`~rayforge.ui_gtk.shared.pref_rows.length_choice_spin_row.LengthChoiceSpinRow`
    (stored in base mm, shown in a per-row unit chosen via dropdown);
    ``None`` fields are a plain spin (angle, count).
    """

    key: str
    title: str
    subtitle: str
    quantity: str | None
    upper: float
    digits: int = 1
    is_int: bool = False


def _length(
    key: str,
    title: str,
    subtitle: str,
    upper: float,
    digits: int = 2,
) -> ParamSpec:
    return ParamSpec(key, title, subtitle, "length", upper, digits)


def _int(
    key: str,
    title: str,
    subtitle: str,
    upper: float,
) -> ParamSpec:
    return ParamSpec(key, title, subtitle, None, upper, 0, True)


def _plain(
    key: str,
    title: str,
    subtitle: str,
    upper: float,
    digits: int = 1,
) -> ParamSpec:
    return ParamSpec(key, title, subtitle, None, upper, digits)


_DIAM = _length("diameter", _("Diameter"), _("Cutting diameter"), 100.0)
_FLUTES = _int(
    "flute_count", _("Flute count"), _("Number of cutting flutes"), 20
)
_CEH = _length(
    "cutting_edge_height",
    _("Cutting edge height"),
    _("Length of the cutting flutes"),
    300.0,
)
_SHANK = _length(
    "shank_diameter", _("Shank diameter"), _("Holder-side diameter"), 100.0
)
_OVERALL = _length(
    "overall_length", _("Overall length"), _("Total tool length"), 500.0
)
_CORNER = _length(
    "corner_radius",
    _("Corner radius"),
    _("Radius of the cutting-edge corner"),
    50.0,
)
_ANGLE = _plain("tip_angle", _("Tip angle"), _("Half-angle of the tip"), 90.0)
_PITCH = _length(
    "pitch", _("Thread pitch"), _("Distance per thread turn"), 10.0
)
_BLADE = _length(
    "blade_thickness",
    _("Blade thickness"),
    _("Thickness of the saw blade"),
    50.0,
)
_ARBOR = _length(
    "arbor_diameter", _("Arbor diameter"), _("Mounting hole diameter"), 50.0
)

CATEGORY_PARAMS: dict[str, list[ParamSpec]] = {
    "END_MILL": [_DIAM, _FLUTES, _CEH, _SHANK, _OVERALL],
    "BALL_NOSE": [_DIAM, _FLUTES, _CEH, _SHANK, _OVERALL],
    "BULL_NOSE": [_DIAM, _CORNER, _FLUTES, _CEH, _SHANK, _OVERALL],
    "CHAMFER": [_DIAM, _ANGLE, _FLUTES, _CEH, _SHANK, _OVERALL],
    "DRILL": [_DIAM, _FLUTES, _OVERALL],
    "PROBE": [_DIAM, _OVERALL],
    "VBIT": [_DIAM, _ANGLE, _FLUTES, _OVERALL],
    "SLITTING_SAW": [_DIAM, _BLADE, _ARBOR, _OVERALL],
    "REAMER": [_DIAM, _FLUTES, _OVERALL],
    "TAP": [_DIAM, _PITCH, _OVERALL],
    "THREAD_MILL": [_DIAM, _PITCH, _FLUTES, _OVERALL],
    "DOVETAIL": [_DIAM, _ANGLE, _FLUTES, _OVERALL],
}


@dataclass
class Tool:
    """
    A persisted cutting tool.

    Attributes:
        uid: Stable unique identifier (filename stem).
        name: Human-readable label shown in pickers.
        max_rpm: Spindle speed cap; CNC steps clamp ``spindle_rpm`` to it.
        label: Short label forwarded to the raygeo ``Tool``.
        category: :class:`ToolCategory` classification.
        tool_material: :class:`ToolMaterial` substrate.
        stickout: Tool stickout (mm) set at the holder.
        coating: Optional coating name, or ``None``.
        model: raygeo :class:`ToolModel` param bag (geometry).
    """

    uid: str
    name: str
    max_rpm: float
    label: str
    category: ToolCategory
    tool_material: ToolMaterial
    stickout: float
    coating: str | None
    model: ToolModel

    def diameter(self) -> float:
        """Cutting diameter (mm), delegated to the model."""
        return self.model.diameter()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for YAML."""
        return {
            "uid": self.uid,
            "name": self.name,
            "max_rpm": self.max_rpm,
            "label": self.label,
            "category": category_to_name(self.category),
            "tool_material": tool_material_to_name(self.tool_material),
            "stickout": self.stickout,
            "coating": self.coating,
            "model": dict(self.model.get_parameters()),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tool":
        """Deserialize from a dict (YAML-loaded)."""
        category = CATEGORY_BY_NAME.get(
            data.get("category", CATEGORY_NAMES[0]),
            CATEGORY_BY_NAME[CATEGORY_NAMES[0]],
        )
        tool_material = TOOL_MATERIAL_BY_NAME.get(
            data.get("tool_material", TOOL_MATERIAL_NAMES[0]),
            TOOL_MATERIAL_BY_NAME[TOOL_MATERIAL_NAMES[0]],
        )
        model = ToolModel(**(data.get("model") or {}))
        return cls(
            uid=data.get("uid") or str(uuid.uuid4()),
            name=data.get("name", _("Unnamed Tool")),
            max_rpm=float(data.get("max_rpm", 0.0)),
            label=data.get("label", ""),
            category=category,
            tool_material=tool_material,
            stickout=float(data.get("stickout", 0.0)),
            coating=data.get("coating"),
            model=model,
        )

    @classmethod
    def create_default(cls, name: str = "") -> "Tool":
        """Create a sensible default 6 mm flat end mill."""
        return cls(
            uid=str(uuid.uuid4()),
            name=name or _("6 mm End Mill"),
            max_rpm=24000.0,
            label=_("6mm EM"),
            category=ToolCategory.END_MILL,
            tool_material=ToolMaterial.CARBIDE,
            stickout=18.0,
            coating=None,
            model=ToolModel(
                diameter=6.0,
                shank_diameter=6.0,
                cutting_edge_height=15.0,
                flute_count=3.0,
                overall_length=50.0,
            ),
        )

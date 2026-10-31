"""
Serialize VarSets as JSON-Schema-like form definitions.

The frontend renders each entry as a form field: ``{key: {type, label,
min, max, step, options, unit, default, kind}}``. The shape follows the
Var class hierarchy (see :mod:`rayforge.core.varset`) rather than the
raw ``to_dict`` output, so the mapping of presentation hints lives in
one place.
"""

from typing import TYPE_CHECKING, Any

from rayforge.core.varset import (
    AngleVar,
    BoolVar,
    ChoiceVar,
    LabeledChoiceVar,
    LengthVar,
    SliderFloatVar,
    SliderIntVar,
    SpeedVar,
    TextAreaVar,
    Var,
)

if TYPE_CHECKING:
    from rayforge.core.varset import VarSet


def _var_kind(var: Var) -> str:
    if isinstance(var, (SliderIntVar, SliderFloatVar)):
        return "slider"
    if isinstance(var, LengthVar):
        return "length"
    if isinstance(var, SpeedVar):
        return "speed"
    if isinstance(var, AngleVar):
        return "angle"
    if isinstance(var, BoolVar):
        return "switch"
    if isinstance(var, TextAreaVar):
        return "textarea"
    return "input"


def _var_unit(var: Var) -> str | None:
    if isinstance(var, LengthVar):
        return "mm"
    if isinstance(var, SpeedVar):
        return "mm/min"
    if isinstance(var, AngleVar):
        return "deg"
    return None


def _var_options(var: Var) -> list[dict[str, Any]] | None:
    if isinstance(var, LabeledChoiceVar):
        return [
            {"value": value, "label": var.get_display_for_value(value)}
            for value in var.choices
        ]
    if isinstance(var, ChoiceVar):
        return [{"value": value, "label": value} for value in var.choices]
    return None


def serialize_var(var: Var) -> dict[str, Any]:
    """One form-field descriptor for a single Var."""
    if var.var_type is bool:
        var_type = "boolean"
    elif var.var_type in (int, float):
        var_type = "number"
    else:
        var_type = "string"
    schema: dict[str, Any] = {
        "key": var.key,
        "type": var_type,
        "kind": _var_kind(var),
        "label": var.label,
        "description": var.description,
        "default": var.default,
        "optional": var.optional,
    }
    if var.var_type in (int, float):
        schema["min"] = getattr(var, "min_val", None)
        schema["max"] = getattr(var, "max_val", None)
        step = getattr(var, "step", None)
        if step is None and var.var_type is int:
            step = 1
        schema["step"] = step
    schema["options"] = _var_options(var)
    schema["unit"] = _var_unit(var)
    return schema


def varset_schema(varset: "VarSet") -> dict[str, dict[str, Any]]:
    """``{key: field descriptor}`` for every var in the VarSet."""
    return {var.key: serialize_var(var) for var in varset}


def varset_groups_schema(
    groups: "list[tuple[str, VarSet]]",
) -> list[dict[str, Any]]:
    """Named groups of field descriptors, as step editors need them."""
    return [
        {
            "title": title,
            "fields": [serialize_var(var) for var in varset],
        }
        for title, varset in groups
    ]

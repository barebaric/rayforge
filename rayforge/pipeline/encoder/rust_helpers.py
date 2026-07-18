"""
Helper functions to convert Rayforge domain models (Machine, Doc, Dialect)
into plain dicts that the Rust G-code encoder (`raygeo.ops.encode_gcode`)
accepts via serde.

No domain model objects cross the Python/Rust boundary — only primitive
types in JSON-serialisable dicts.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from raygeo.ops import Ops
from raygeo.ops.convert import GcodeDialectSpec

from ...machine.models.macro import MacroTrigger

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...machine.models.dialect import GcodeDialect
    from ...machine.models.machine import Machine

logger = logging.getLogger(__name__)


def _repr_val(v: float) -> str:
    """Format a float the same way Python's TemplateFormatter does
    (using repr) so path_vars match the old behaviour exactly."""
    return repr(v)


# ── Dialect conversion ───────────────────────────────────────────


def dialect_to_spec(
    dialect: "GcodeDialect", machine: "Machine"
) -> GcodeDialectSpec:
    """Convert a `GcodeDialect` dataclass to a typed
    ``GcodeDialectSpec`` pyclass instance."""
    return GcodeDialectSpec(
        laser_on=dialect.laser_on,
        laser_off=dialect.laser_off,
        tool_change=dialect.tool_change,
        set_speed=dialect.set_speed,
        travel_move=dialect.travel_move,
        linear_move=dialect.linear_move,
        arc_cw=dialect.arc_cw,
        arc_ccw=dialect.arc_ccw,
        bezier_cubic=dialect.bezier_cubic or "",
        air_assist_on=dialect.air_assist_on,
        air_assist_off=dialect.air_assist_off,
        spindle_on_cw=dialect.spindle_on_cw,
        spindle_on_ccw=dialect.spindle_on_ccw,
        spindle_off=dialect.spindle_off,
        coolant_flood=dialect.coolant_flood,
        coolant_mist=dialect.coolant_mist,
        coolant_off=dialect.coolant_off,
        dwell=dialect.dwell,
        preamble=dialect.preamble,
        postscript=dialect.postscript,
        inject_wcs_after_preamble=dialect.inject_wcs_after_preamble,
        can_g0_with_speed=dialect.can_g0_with_speed,
        omit_unchanged_coords=dialect.omit_unchanged_coords,
        continuous_laser_mode=dialect.continuous_laser_mode,
        modal_feedrate=dialect.modal_feedrate,
        gcode_precision=machine.gcode_precision,
    )


# ── Path variable resolution ─────────────────────────────────────


def _build_machine_path_vars(machine: "Machine") -> Dict[str, str]:
    """Static path variables that never change during encoding."""
    wcs_offset = machine.get_active_wcs_offset()
    ax_w, ax_h = machine.axis_extents
    return {
        "machine.name": machine.name,
        "machine.active_wcs": machine.active_wcs,
        "machine.axis_extents[0]": _repr_val(ax_w),
        "machine.axis_extents[1]": _repr_val(ax_h),
        "wcs_offset[0]": _repr_val(wcs_offset[0]),
        "wcs_offset[1]": _repr_val(wcs_offset[1]),
        "wcs_offset[2]": _repr_val(wcs_offset[2]),
    }


def _build_job_path_vars(ops: Ops, doc: "Doc") -> Dict[str, str]:
    """Job-level path variables (depend on ops extents and doc name)."""
    xmin, ymin, xmax, ymax = ops.rect()
    return {
        "doc.name": doc.name if doc else "",
        "job.extents[0]": _repr_val(xmin),
        "job.extents[1]": _repr_val(ymin),
        "job.extents[2]": _repr_val(xmax),
        "job.extents[3]": _repr_val(ymax),
    }


def _build_layer_path_vars_for_doc(
    doc: "Doc", machine: "Machine"
) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    if not doc:
        return result
    for layer in doc.layers:
        wcs = layer.get_effective_wcs(machine)
        wcs_offset = machine.get_wcs_offset(wcs)
        result[layer.uid] = {
            "layer.name": layer.name,
            "wcs_offset[0]": _repr_val(wcs_offset[0]),
            "wcs_offset[1]": _repr_val(wcs_offset[1]),
            "wcs_offset[2]": _repr_val(wcs_offset[2]),
        }
    return result


def _build_workpiece_path_vars_for_doc(
    doc: "Doc",
) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    if not doc:
        return result
    for layer in doc.layers:
        for wp in layer.all_workpieces:
            pos = wp.pos
            size = wp.size
            result[wp.uid] = {
                "workpiece.name": wp.name,
                "workpiece.pos[0]": _repr_val(pos[0]),
                "workpiece.pos[1]": _repr_val(pos[1]),
                "workpiece.size[0]": _repr_val(size[0]),
                "workpiece.size[1]": _repr_val(size[1]),
            }
    return result


# ── Macro table ───────────────────────────────────────────────────


def _build_macro_table(machine: "Machine") -> Dict:
    """Build the MacroTable dict for the Rust encoder."""
    hooks = machine.hookmacros
    macros = machine.macros  # Dict[str, Macro]

    def _macro_dict(m) -> Optional[Dict]:
        if m is None:
            return None
        return {"name": m.name, "code": m.code, "enabled": m.enabled}

    trigger_map = {
        MacroTrigger.LAYER_START: "layer_start",
        MacroTrigger.LAYER_END: "layer_end",
        MacroTrigger.WORKPIECE_START: "workpiece_start",
        MacroTrigger.WORKPIECE_END: "workpiece_end",
    }
    table: Dict = {}
    for trigger, key in trigger_map.items():
        table[key] = _macro_dict(hooks.get(trigger))

    # All macros by name for @include resolution
    all_macros: Dict[str, Dict] = {}
    for name, m in macros.items():
        all_macros[name] = {
            "name": m.name,
            "code": m.code,
            "enabled": m.enabled,
        }
    table["all_macros"] = all_macros

    return table


# ── Laser heads ───────────────────────────────────────────────────


def _build_heads(machine: "Machine") -> List[Dict]:
    return [
        {
            "uid": head.uid,
            "tool_number": head.tool_number,
            "max_power": float(head.max_power),
        }
        for head in machine.heads
    ]


# ── Layer WCS ─────────────────────────────────────────────────────


def _build_layer_wcs(doc: "Doc", machine: "Machine") -> Dict[str, str]:
    """Per-layer WCS command, keyed by layer UID."""
    result: Dict[str, str] = {}
    if not doc:
        return result
    for layer in doc.layers:
        result[layer.uid] = layer.get_effective_wcs(machine)
    return result


# ── Public entry points ──────────────────────────────────────────


def build_encode_context(ops: Ops, machine: "Machine", doc: "Doc") -> Dict:
    """Build the EncodeContext dict for raygeo.ops.encode_gcode."""
    path_vars: Dict[str, str] = {}
    path_vars.update(_build_machine_path_vars(machine))
    path_vars.update(_build_job_path_vars(ops, doc))

    return {
        "gcode_precision": machine.gcode_precision,
        "max_travel_speed": float(machine.max_travel_speed),
        "default_head_uid": machine.get_default_head().uid,
        "heads": _build_heads(machine),
        "active_wcs": machine.active_wcs,
        "layer_wcs": _build_layer_wcs(doc, machine),
        "macros": _build_macro_table(machine),
        "path_vars": path_vars,
        "layer_path_vars": _build_layer_path_vars_for_doc(doc, machine),
        "workpiece_path_vars": _build_workpiece_path_vars_for_doc(doc),
    }

"""
Helpers for the Epilog print-job encoder.

The Epilog encoder lives in raygeo (`raygeo.ops.convert.EpilogSpec`).
This module converts the Rayforge domain models (`Machine`, `Doc`)
into the plain, JSON-serialisable job context the encoder accepts, so
no domain model crosses the Python/Rust boundary.
"""

import json
import logging
from typing import TYPE_CHECKING

from raygeo.ops.convert import EpilogSpec

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...machine.models.machine import Machine

logger = logging.getLogger(__name__)

#: Machine origins the print protocol supports.
SUPPORTED_ORIGIN = "top_left"


def build_epilog_context(machine: "Machine", doc: "Doc") -> dict:
    """Build the job context dict for `raygeo.ops.convert.EpilogSpec`."""
    origin = machine.origin.value
    return {
        "title": doc.name or "Rayforge",
        "resolution": int(machine.driver_args.get("resolution", 500)),
        "frequency": int(machine.driver_args.get("frequency", 1000)),
        "bed_mm": [
            float(machine.axis_extents[0]),
            float(machine.axis_extents[1]),
        ],
        "max_cut_speed": float(machine.max_cut_speed),
        "origin_top_left": origin == SUPPORTED_ORIGIN,
        "rotary": any(layer.rotary_enabled for layer in doc.layers),
    }


def epilog_spec(machine: "Machine", doc: "Doc") -> EpilogSpec:
    """Build the encoder spec for a job on *machine*."""
    context = build_epilog_context(machine, doc)
    return EpilogSpec(json.dumps(context))

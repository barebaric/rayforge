"""
Helpers for the Epilog print-job encoder.

The Epilog encoder lives in raydriver (`raydriver.encode.EpilogSpec`).
This module converts the Rayforge domain models (`Machine`, `Doc`)
into the plain, JSON-serialisable job context the encoder accepts, so
no domain model crosses the Python/Rust boundary.
"""

import json
import logging
from typing import TYPE_CHECKING

from raydriver.encode import EpilogSpec

from ..machine.driver.epilog.epilog_driver import DEFAULT_SPEED_REFERENCE

if TYPE_CHECKING:
    from ..core.doc import Doc
    from ..machine.models.machine import Machine

logger = logging.getLogger(__name__)

#: Machine origins the print protocol supports.
SUPPORTED_ORIGIN = "top_left"


def build_epilog_context(machine: "Machine", doc: "Doc") -> dict:
    """Build the job context dict for `raydriver.encode.EpilogSpec`."""
    origin = machine.origin.value
    return {
        "title": doc.name or "Rayforge",
        "resolution": int(machine.driver_args.get("resolution", 500)),
        "frequency": int(machine.driver_args.get("frequency", 1000)),
        "bed_mm": [
            float(machine.axis_extents[0]),
            float(machine.axis_extents[1]),
        ],
        # The machine's percentage scale is its own calibrated
        # reference, not the machine's speed limit (that would scale
        # every job by the wrong factor).
        "speed_reference_mm_min": float(
            machine.driver_args.get("speed_reference")
            or DEFAULT_SPEED_REFERENCE
        ),
        "origin_top_left": origin == SUPPORTED_ORIGIN,
        "rotary": any(layer.rotary_enabled for layer in doc.layers),
    }


def epilog_spec(machine: "Machine", doc: "Doc") -> EpilogSpec:
    """Build the encoder spec for a job on *machine*."""
    context = build_epilog_context(machine, doc)
    return EpilogSpec(json.dumps(context))

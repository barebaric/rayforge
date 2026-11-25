"""Inputs for the Epilog print-job encoder.

The encoder itself lives in raydriver (``raydriver.encode.EpilogSpec``).
This module converts the Rayforge domain models into the plain,
JSON-serialisable job context the encoder accepts, so no domain model
crosses the Python/Rust boundary. It belongs to the driver package:
the pipeline only asks the driver for an encoder spec.
"""

import json
from typing import TYPE_CHECKING

from raydriver.encode import EpilogSpec

if TYPE_CHECKING:
    from ....core.doc import Doc
    from ...models.machine import Machine

#: Supported raster and vector resolutions in dots per inch.
RESOLUTIONS = ("100", "200", "250", "400", "500", "1000")
DEFAULT_RESOLUTION = "500"

#: Cutting speed the machine reaches at 100 %, in mm/min.
#:
#: The machine's speed control is a percentage of its own reference,
#: which is a property of the model: the reference driver calibrates a
#: Helix at 21.3 mm/s (1278 mm/min). It is not the machine's maximum
#: travel speed. Zing-specific calibration is still owed; this default
#: matches the 600 mm/min = 10 % figure the profile notes used.
DEFAULT_SPEED_REFERENCE = 6000

DEFAULT_FREQUENCY = 1000
MIN_FREQUENCY = 10
MAX_FREQUENCY = 5000

#: The only origin the print protocol can address.
SUPPORTED_ORIGIN = "top_left"


def build_epilog_context(machine: "Machine", doc: "Doc") -> dict:
    """Build the job context dict for ``raydriver.encode.EpilogSpec``."""
    driver_args = machine.driver_args
    return {
        "title": (doc.name if doc else "") or "Rayforge",
        "resolution": int(driver_args.get("resolution") or DEFAULT_RESOLUTION),
        "frequency": int(driver_args.get("frequency") or DEFAULT_FREQUENCY),
        "bed_mm": [
            float(machine.axis_extents[0]),
            float(machine.axis_extents[1]),
        ],
        # The machine's percentage scale is its own calibrated
        # reference, not the machine's speed limit (that would scale
        # every job by the wrong factor).
        "speed_reference_mm_min": float(
            driver_args.get("speed_reference") or DEFAULT_SPEED_REFERENCE
        ),
        "origin_top_left": machine.origin.value == SUPPORTED_ORIGIN,
        "rotary": bool(
            doc and any(layer.rotary_enabled for layer in doc.layers)
        ),
    }


def build_epilog_spec(machine: "Machine", doc: "Doc") -> EpilogSpec:
    """Build the encoder spec for a job on *machine*."""
    return EpilogSpec(json.dumps(build_epilog_context(machine, doc)))

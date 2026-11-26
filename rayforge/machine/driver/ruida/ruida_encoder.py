"""Ruida encoder bridge.

Builds the plain-data context the native raydriver encoder needs and
runs it against the machine-space Ops. The encoder policy itself lives
in ``raydriver.encode.RuidaSpec``.
"""

import json
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

from raydriver.encode import RuidaSpec
from raygeo.ops.convert import Encoder
from raygeo.ops.types import RasterMode, SectionType

from rayforge.pipeline.artifact.encoded import (
    EncodedOutput,
    MachineCodeOpMap,
)

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from rayforge.core.doc import Doc
    from rayforge.core.layer import Layer
    from rayforge.machine.models.machine import Machine

DEFAULT_POWER_FLOOR = 8.0
DEFAULT_IMAGE_POWER_BIAS = 8.0

_WCS_TO_REF_POINT = {
    "MACHINE": "MACHINE",
    "ANCHOR": "ABSOLUTE",
    "CURRENT": "CURRENT",
    "SET_POINT": "SET_POINT",
}

_DEFAULT_LAYER_SPEED_MMS = 100.0
_DEFAULT_LAYER_POWER = 0.2
_DEFAULT_LAYER_FREQUENCY_HZ = 20000


class _LayerSettings(TypedDict):
    """The flattened encoder context for one document layer."""

    uid: str
    name: str
    color: str
    speed_mm_s: float
    frequency_khz: float
    power_pct: float
    min_power_pct: float | None


@runtime_checkable
class _LaserStep(Protocol):
    """A workflow step that carries laser process settings."""

    power: float
    frequency: int


@runtime_checkable
class _HasMinPower(Protocol):
    """A workflow step that declares a minimum power."""

    min_power: float | None


def _layer_settings(layer: "Layer") -> _LayerSettings:
    """Flatten one document layer's workflow settings for the encoder."""
    speed_mms = _DEFAULT_LAYER_SPEED_MMS
    power_fraction = _DEFAULT_LAYER_POWER
    frequency_hz = _DEFAULT_LAYER_FREQUENCY_HZ
    min_power_pct = None
    if layer.workflow is not None and layer.workflow.steps:
        step = layer.workflow.steps[0]
        speed_mms = float(step.cut_speed) / 60.0
        if isinstance(step, _LaserStep):
            power_fraction = float(step.power)
            frequency_hz = int(step.frequency)
        else:
            power_fraction = float(
                step.extra.get("power", _DEFAULT_LAYER_POWER)
            )
            frequency_hz = int(
                step.extra.get("frequency", _DEFAULT_LAYER_FREQUENCY_HZ)
            )
        raw_min = step.min_power if isinstance(step, _HasMinPower) else None
        if raw_min is None:
            raw_min = step.extra.get("min_power", None)
        if raw_min is None:
            raw_min = layer.extra.get("min_power", None)
        if raw_min is not None:
            min_power_pct = float(raw_min) * 100.0
    frequency_khz = frequency_hz / 1000.0 if frequency_hz > 0 else 20.0
    return {
        "uid": layer.uid,
        "name": layer.name,
        "color": layer.color,
        "speed_mm_s": speed_mms,
        "frequency_khz": frequency_khz,
        "power_pct": power_fraction * 100.0,
        "min_power_pct": min_power_pct,
    }


def build_ruida_spec(machine: "Machine | None", doc: "Doc") -> RuidaSpec:
    """Build the native Ruida encoder context from the machine and doc."""
    wcs = machine.active_wcs if machine is not None else None
    ref_point = _WCS_TO_REF_POINT.get(wcs or "")
    if ref_point is None:
        # G54 is the framework default; it has no Ruida reference-point
        # name, so it maps to the machine origin.
        if wcs == "G54" or machine is None:
            ref_point = "MACHINE"
        else:
            raise ValueError(
                f"Unsupported WCS for Ruida reference point: {wcs!r} — "
                f"valid names: {', '.join(_WCS_TO_REF_POINT)}"
            )
    driver_args = machine.driver_args if machine is not None else {}
    context = {
        "job_label": doc.name or "Rayforge Job",
        "ref_point": ref_point,
        "power_cap": (
            machine.get_job_power_cap() if machine is not None else None
        ),
        "power_floor_pct": min(
            max(
                float(driver_args.get("power_floor", DEFAULT_POWER_FLOOR)), 0.0
            ),
            100.0,
        ),
        "power_floor_enabled": bool(
            driver_args.get("power_floor_enabled", False)
        ),
        "image_power_bias_pct": min(
            max(
                float(
                    driver_args.get(
                        "image_power_bias", DEFAULT_IMAGE_POWER_BIAS
                    )
                ),
                0.0,
            ),
            100.0,
        ),
        "power_scaling_enabled": bool(
            driver_args.get("power_scaling_enabled", True)
        ),
        "head_tool_numbers": (
            {head.uid: head.tool_number for head in machine.heads}
            if machine is not None
            else {}
        ),
        "layers": [_layer_settings(layer) for layer in doc.layers],
    }
    return RuidaSpec(context_json=json.dumps(context))


class RuidaEncoder:
    """Encodes Ops into a Ruida job through the native encoder."""

    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self.op_map: MachineCodeOpMap | None = None
        self.active_laser = 1

    def encode(
        self, ops: "Ops", machine: "Machine", doc: "Doc"
    ) -> EncodedOutput:
        spec = build_ruida_spec(machine, doc)
        result = ops.encode(Encoder(spec))
        payload = bytes(result.payload) if result.payload else b""
        self.op_map = MachineCodeOpMap(
            op_to_machine_code=result.op_to_machine_code,
            machine_code_to_op=result.machine_code_to_op,
        )
        driver_data = {"payload": payload} if payload else {}
        return EncodedOutput(
            text=result.text or "",
            op_map=self.op_map,
            driver_data=driver_data,
        )

    @staticmethod
    def _section_layer_mode(
        section_type: SectionType, raster_mode: RasterMode | None
    ) -> str:
        """Map a section's type and raster mode to a layer mode."""
        if section_type != SectionType.RASTER_FILL:
            return "VECTOR"
        if raster_mode == RasterMode.DITHER:
            return "DITHER"
        if raster_mode == RasterMode.DEPTH_MAP:
            return "DEPTHMAP"
        if raster_mode == RasterMode.CONSTANT_POWER:
            return "RASTER"
        if raster_mode == RasterMode.VARIABLE_POWER:
            return "IMAGE"
        return "VECTOR"

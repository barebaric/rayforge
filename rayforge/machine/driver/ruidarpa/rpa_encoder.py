"""
RPA Encoder - Produces GlueScript transcripts for the Ruida Protocol
Analyzer driver.

The GlueScript transcript is the source: the encoder drives the ruida-pa
GlueScript API (``rd_gluescript.GlueScript``), which owns job framing,
layer attribute blocks, per-layer action routing, and the bounding-box
math. The returned ``text`` is the transcript; the backend compiles it
to rpascript (the compiled output) via ``stage_gluescript`` when the job
runs. Coordinates use mm natively (no unit conversion needed).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from raygeo.geo.types import Point3D
from raygeo.ops import Ops
from raygeo.ops.state import AirAssistMode, CoolantMode
from raygeo.ops.types import CommandType, RasterMode, SectionType
from ruidadriver.rd_gluescript import GlueScript

from rayforge.pipeline.encoder.base import (
    EncodedOutput,
    MachineCodeOpMap,
    OpsEncoder,
)

if TYPE_CHECKING:
    from rayforge.core.doc import Doc
    from rayforge.core.layer import Layer
    from rayforge.machine.models.machine import Machine

logger = logging.getLogger(__name__)

# Overscan derivation tolerances. Raster scan lines are classified as
# horizontal (X_BI) or vertical (Y_BI) when their angle from the axis is
# within this tolerance; diagonal lines are unsupported by the Ruida
# controller and fall back to no overscan. The epsilon guards against
# degenerate zero-length scan lines.
_OVERSCAN_ANGLE_TOLERANCE_DEG = 0.01
_OVERSCAN_ANGLE_EPSILON = 1e-6
_DEFAULT_LAYER_SPEED_MMS = 100.0
_DEFAULT_LAYER_FREQUENCY_KHZ = 20.0
_DEFAULT_LAYER_FREQUENCY_HZ = _DEFAULT_LAYER_FREQUENCY_KHZ * 1000
_DEFAULT_LAYER_POWER = 0.2  # fraction, i.e. 20%
_DEFAULT_JOB_LABEL = "Rayforge Job"
_DEFAULT_LAYER_COLOR = "#00ccff"
DEFAULT_POWER_FLOOR = 100.0  # percent, i.e. 100%
DEFAULT_IMAGE_POWER_BIAS = 8.0  # percent, i.e. 8%


@runtime_checkable
class _LaserProcessStep(Protocol):
    """A workflow step carrying laser process attributes.

    The concrete laser step classes live in a dynamically loaded
    add-on and cannot be imported here, so the encoder recognizes
    them structurally.
    """

    power: float
    frequency: int


@runtime_checkable
class _HasMinPower(Protocol):
    """A workflow step that optionally declares a minimum power."""

    min_power: float | None


# Maps the framework WCS slot names to the Ruida reference point strings
# accepted by GlueScript.declare_job. The framework default WCS ("G54")
# is deliberately absent — it maps to "MACHINE" to keep golden output
# byte-identical.
_WCS_TO_REF_POINT = {
    "MACHINE": "MACHINE",
    "ANCHOR": "ABSOLUTE",
    "CURRENT": "CURRENT",
    "SET_POINT": "SET_POINT",
}

# Last WCS that triggered the G54 fallback warning. Encoders are fresh
# per encode, so dedup state lives at module level; the warning fires
# only when the value changes (first G54, and after any real WCS).
# Worst case under concurrent encodes: a duplicated or suppressed
# warning — assignments are atomic under the GIL and output is
# unaffected.
_last_fallback_wcs: str | None = None


class RuidaRPAEncoder(OpsEncoder):
    """Encodes Ops commands into a GlueScript transcript.

    The transcript is the source: each Ops command is translated into a
    GlueScript call line, and the encoder's ``text`` output IS that
    transcript. The backend compiles it to rpascript (the compiled
    output) via ``stage_gluescript`` when the job runs, so the staged
    rpascript stays controller-valid and the bounding boxes are computed
    by GlueScript from the actual cut extents.
    """

    def __init__(self, gluescript: Any = None) -> None:
        """Create an encoder, optionally bound to an injected GlueScript.

        When ``gluescript`` is provided (a live backend GlueScript such as
        the wrapped RdDriver or an RpcRdDriver), the encoder authors ops
        directly into it instead of creating its own ``GlueScript()``.
        The injected instance survives ``_reset_state()`` across encodes.
        """
        self._injected_gluescript = gluescript
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all encoder state for a new encoding session."""
        self.current_pos: Point3D = (0.0, 0.0, 0.0)
        self.active_laser: int = 1
        self.doc: Doc | None = None
        self.machine: Machine | None = None
        self.op_map: MachineCodeOpMap | None = None
        self._gluescript: Any = self._injected_gluescript
        self._layer_key: int = 0
        self._layer_uid: str | None = None
        self._layer_declared: bool = False
        self._section_type: SectionType | None = None
        self._section_raster_mode: RasterMode | None = None
        self._layer_mode: str = "VECTOR"
        self._overscan: str = "NONE"
        self._power_fraction: float = 0.0
        self._power_min_fraction: float = 0.0
        self._emitted_min_fraction: float = 0.0
        self._power_floor: float = DEFAULT_POWER_FLOOR / 100.0
        self._image_power_bias: float = DEFAULT_IMAGE_POWER_BIAS / 100.0
        self._snapshot_len: int = 0
        self._op_count: int = 0
        self._op_contributions: dict[int, list[tuple[int, int]]] = {}
        self._job_started: bool = False
        self._job_ended: bool = False

    # -- Public API ---------------------------------------------------------

    def encode(self, ops: Ops, machine: Machine, doc: Doc) -> EncodedOutput:
        """Encode Ops commands into a GlueScript transcript.

        The transcript IS the source: each Ops command is translated into
        a GlueScript call line, and the returned ``text`` is that
        transcript. The backend compiles it to rpascript (the compiled
        output) via ``stage_gluescript`` when the job runs.

        Args:
            ops: Ops object from raygeo containing commands to encode.
            machine: The machine configuration (used for laser head lookup).
            doc: The document being processed.

        Returns:
            EncodedOutput whose text is the GlueScript transcript, with
            an op_map spanning the transcript lines.

        Raises:
            RuntimeError: If the ruida-pa GlueScript API is incompatible or
                the job was not completed (missing JOB_END).
        """
        self._reset_state()
        # Version gate: stage_gluescript is the re-staging entry point the
        # adapter relies on to compile the transcript into rpascript.
        if not hasattr(GlueScript, "stage_gluescript"):
            raise RuntimeError(
                "GlueScript.stage_gluescript() is missing — ruida-pa "
                ">= 0.15.2 is required to use the ruidarpa driver"
            )
        if ops.len() == 0:
            self.op_map = MachineCodeOpMap()
            return EncodedOutput(text="", op_map=self.op_map)

        self.doc = doc
        self.machine = machine
        driver_args = machine.driver_args if machine is not None else {}
        raw_floor = driver_args.get("power_floor", DEFAULT_POWER_FLOOR)
        self._power_floor = min(max(float(raw_floor), 0.0), 100.0) / 100.0
        raw_bias = driver_args.get(
            "image_power_bias", DEFAULT_IMAGE_POWER_BIAS
        )
        self._image_power_bias = min(max(float(raw_bias), 0.0), 100.0) / 100.0
        self.op_map = MachineCodeOpMap()
        self._op_count = ops.len()
        if self._gluescript is None:
            gluescript_type: Any = GlueScript
            self._gluescript = gluescript_type()
        else:
            # Injected backend: reset it for a fresh job so a prior
            # encode's transcript does not leak into this one.
            self._gluescript.new_gluescript()

        for i in range(ops.len()):
            self._snapshot_sections()
            self._handle_command(ops, i, machine)
            self._record_contribution(i)

        if not self._job_started or not self._job_ended:
            raise RuntimeError(
                "Ops sequence must start with JOB_START and end with JOB_END"
            )

        lines = list(self._gluescript.gluescript)
        text = "\n".join(lines)
        self._build_op_map(len(lines))
        return EncodedOutput(text=text, op_map=self.op_map)

    # -- Command dispatch ---------------------------------------------------

    def _handle_command(self, ops: Ops, idx: int, machine: Machine) -> None:
        """Dispatch a single Ops command to the appropriate handler."""
        ct = ops.command_type(idx)
        if ct == CommandType.SET_POWER:
            self._handle_set_power(ops, idx)
        elif ct == CommandType.SET_FEED_RATE:
            self._handle_set_cut_speed(ops, idx)
        elif ct == CommandType.SET_RAPID_RATE:
            self._handle_set_travel_speed(ops, idx)
        elif ct == CommandType.SET_FREQUENCY:
            self._handle_set_frequency(ops, idx)
        elif ct == CommandType.SET_PULSE_WIDTH:
            self._handle_set_pulse_width(ops, idx)
        elif ct == CommandType.SET_COOLANT:
            self._handle_coolant_as_air_assist(ops, idx)
        elif ct == CommandType.SET_HEAD:
            self._handle_set_laser(ops, idx, machine)
        elif ct == CommandType.MOVE_TO:
            self._handle_move_to(ops, idx)
        elif ct == CommandType.LINE_TO:
            self._handle_line_to(ops, idx)
        elif ct == CommandType.ARC_TO:
            self._handle_arc_to(ops, idx)
        elif ct == CommandType.SCAN_LINE:
            self._handle_scan_line(ops, idx)
        elif ct == CommandType.DWELL:
            self._handle_dwell(ops, idx)
        elif ct == CommandType.BEZIER_TO:
            self._handle_bezier_to(ops, idx)
        elif ct == CommandType.QUADRATIC_BEZIER_TO:
            self._handle_quadratic_bezier_to(ops, idx)
        elif ct == CommandType.JOB_START:
            self._handle_job_start()
        elif ct == CommandType.JOB_END:
            self._handle_job_end(idx)
        elif ct == CommandType.LAYER_START:
            self._handle_layer_start(ops, idx)
        elif ct == CommandType.LAYER_END:
            if not self._layer_declared:
                raise ValueError(
                    "LAYER_END encountered before any WORKPIECE_START — "
                    "every Rayforge layer must contain at least one workpiece"
                )
            # GlueScript closes layers implicitly; an unclosed ops
            # section must not leak into the next layer. Reset the
            # layer mode so a stray section after LAYER_END fails
            # loudly instead of reusing the previous layer's mode.
            self._section_type = None
            self._section_raster_mode = None
            self._layer_mode = "VECTOR"
            self._power_fraction = 0.0
            self._power_min_fraction = 0.0
            self._emitted_min_fraction = 0.0
            self._layer_uid = None
        elif ct == CommandType.WORKPIECE_START:
            self._handle_workpiece_start(ops, idx)
        elif ct == CommandType.WORKPIECE_END:
            self._handle_workpiece_end()
        elif ct == CommandType.OPS_SECTION_START:
            self._handle_ops_section_start(ops, idx)
        elif ct == CommandType.OPS_SECTION_END:
            self._handle_ops_section_end()
        elif ct == CommandType.SET_AIR_ASSIST:
            self._handle_set_air_assist(ops, idx)
        elif ct == CommandType.SET_SPINDLE_RPM:
            pass  # Ruida is laser-only; spindle not applicable
        elif ct == CommandType.SET_HEAD_COOLANT:
            pass  # Per-head coolant not yet supported
        elif (
            ct == CommandType.STATE_BLOCK_START
            or ct == CommandType.STATE_BLOCK_END
        ):
            pass  # Structural marker; no rpascript output
        else:
            raise ValueError(f"Unknown command type: {ct}")
        self._gluescript.comment([f"# Op {idx}: {ct.name}"])

    # -- Helpers ------------------------------------------------------------

    def _require_active_layer(self) -> None:
        """Fail fast when a layer-scoped op arrives before any LAYER_START."""
        if self._layer_key == 0:
            raise ValueError(
                "Layer-scoped op encountered before LAYER_START — "
                "GlueScript routing requires an active layer"
            )

    def _emit_power(self, power_fraction: float) -> None:
        """Emit laser power for the current layer action block.

        Vector layers without overscan declare a power range whose lower
        bound is the layer's resolved min-power fraction, letting the
        controller ramp power during acceleration and deceleration. All
        other modes keep min == max.
        """
        self._require_active_layer()
        section_type = self._section_type
        if power_fraction == 0.0:
            return
        if section_type is not None and self._layer_mode in (
            "IMAGE",
            "DEPTHMAP",
        ):
            if power_fraction == 0.0:
                return
            self._gluescript.power(power_fraction * 100.0)
            return

        min_fraction = power_fraction
        if (
            self._layer_mode == "VECTOR"
            and self._overscan == "NONE"
            and 0.0 < self._power_min_fraction < power_fraction
        ):
            min_fraction = self._power_min_fraction
        if power_fraction != self._power_fraction or (
            min_fraction != self._emitted_min_fraction
        ):
            self._power_fraction = power_fraction
            self._emitted_min_fraction = min_fraction
            self._gluescript.power_range(
                min_fraction * 100.0, power_fraction * 100.0
            )

    def _find_layer(self, layer_uid: str) -> Layer | None:
        """Look up a document layer by uid, or None when unknown."""
        if self.doc is None:
            return None
        return next(
            (layer for layer in self.doc.layers if layer.uid == layer_uid),
            None,
        )

    def _layer_settings(
        self, layer: Layer | None
    ) -> tuple[float, float, float]:
        """Extract (speed_mms, frequency_khz, power_pct) for a layer.

        Reads the first workflow step, falling back to safe defaults. The
        raw power percent is passed through unchanged; GlueScript clamps
        power below its 8% minimum by emitting a ``# warning:`` comment
        into the layer attributes rather than raising.
        """
        speed_mms = _DEFAULT_LAYER_SPEED_MMS
        power_fraction = _DEFAULT_LAYER_POWER
        frequency_hz = _DEFAULT_LAYER_FREQUENCY_HZ
        if (
            layer is not None
            and layer.workflow is not None
            and layer.workflow.steps
        ):
            first_step = layer.workflow.steps[0]
            # cut_speed is stored in mm/min; GlueScript expects mm/s.
            speed_mms = float(first_step.cut_speed) / 60.0
            if isinstance(first_step, _LaserProcessStep):
                power_fraction = float(first_step.power)
                frequency_hz = int(first_step.frequency)
            else:
                power_fraction = float(
                    first_step.extra.get("power", _DEFAULT_LAYER_POWER)
                )
                frequency_hz = int(
                    first_step.extra.get(
                        "frequency", _DEFAULT_LAYER_FREQUENCY_HZ
                    )
                )

        power_pct = power_fraction * 100.0

        frequency_khz = (
            frequency_hz / 1000.0
            if frequency_hz > 0
            else _DEFAULT_LAYER_FREQUENCY_KHZ
        )
        return speed_mms, frequency_khz, power_pct

    def _layer_min_power_fraction(self, layer: Layer | None) -> float:
        """Resolve the layer's min-power fraction for power compensation.

        Reads the first workflow step's min_power attribute when the
        step declares one, falling back from step.extra to layer.extra
        and defaulting to the configured power floor (default 100%).
        The raw value is clamped once at this boundary so a min below
        the floor or above 100% never reaches GlueScript as a lower
        power bound.
        """
        min_fraction = self._power_floor
        if (
            layer is not None
            and layer.workflow is not None
            and layer.workflow.steps
        ):
            first_step = layer.workflow.steps[0]
            raw_min: float | None = None
            if isinstance(first_step, _HasMinPower):
                raw_min = first_step.min_power
            if raw_min is None:
                raw_min = first_step.extra.get("min_power", None)
            if raw_min is None:
                raw_min = layer.extra.get("min_power", None)
            if raw_min is not None:
                min_fraction = float(raw_min)
        return min(max(min_fraction, self._power_floor), 1.0)

    # -- Movement handlers --------------------------------------------------

    def _movement_form(self, dx: float, dy: float) -> str:
        """Classify a movement delta into an overscan-aware axis form.

        X_BI/Y_BI overscan layers emit single-axis moves/cuts for
        horizontal/vertical motion; diagonal motion and NONE overscan
        layers always use the XY form.
        """
        if self._overscan == "NONE":
            return "XY"
        if (
            abs(dx) < _OVERSCAN_ANGLE_EPSILON
            and abs(dy) < _OVERSCAN_ANGLE_EPSILON
        ):
            return "XY"
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        if angle <= _OVERSCAN_ANGLE_TOLERANCE_DEG:
            return "X"
        if angle >= 90.0 - _OVERSCAN_ANGLE_TOLERANCE_DEG:
            return "Y"
        return "XY"

    def _emit_movement(self, x: float, y: float, cut: bool) -> None:
        """Emit a move or cut to (x, y) using the overscan-aware form."""
        dx = x - self.current_pos[0]
        dy = y - self.current_pos[1]
        form = self._movement_form(dx, dy)
        if cut:
            if form == "X":
                self._gluescript.cut_x_to(x)
            elif form == "Y":
                self._gluescript.cut_y_to(y)
            else:
                self._gluescript.cut_xy_to(x, y)
        else:
            if form == "X":
                self._gluescript.move_x_to(x)
            elif form == "Y":
                self._gluescript.move_y_to(y)
            else:
                self._gluescript.move_xy_to(x, y)

    def _handle_move_to(self, ops: Ops, idx: int) -> None:
        """Rapid move (laser off) to an absolute position."""
        x, y, z = ops.endpoint(idx)
        self._require_active_layer()
        if z != 0:
            logger.warning(
                "Ignoring Z=%.3fmm on MOVE_TO — laser jobs are 2D "
                "and rpascript has no Z move for this driver",
                z,
            )
        self._emit_movement(x, y, cut=False)
        self.current_pos = (x, y, z)

    def _handle_line_to(self, ops: Ops, idx: int) -> None:
        """Cutting move (laser on) to an absolute position."""
        x, y, z = ops.endpoint(idx)
        self._require_active_layer()
        if z != 0:
            logger.warning(
                "Ignoring Z=%.3fmm on LINE_TO — laser jobs are 2D "
                "and rpascript has no Z move for this driver",
                z,
            )
        self._emit_movement(x, y, cut=True)
        self.current_pos = (x, y, z)

    def _linearize_curve(self, ops: Ops, idx: int) -> None:
        """Linearize a curve op into cut and power actions.

        Rpascript has no native arc/bezier command, so curves are
        decomposed via ops.linearize() into cut segments and per-segment
        power adjustments. A zero-power segment turns the laser off, so
        the head moves (rapid) instead of cutting at 0% power.
        """
        self._require_active_layer()
        start_pos = self.current_pos
        end = ops.endpoint(idx)

        sub_ops = ops.linearize(idx, start_pos)
        power = self._power_fraction
        for j in range(sub_ops.len()):
            sub_ct = sub_ops.command_type(j)
            if sub_ct == CommandType.LINE_TO:
                sx, sy, sz = sub_ops.endpoint(j)
                if power > 0.0:
                    self._emit_movement(sx, sy, cut=True)
                    self.current_pos = (sx, sy, sz)
                else:
                    self._emit_movement(sx, sy, cut=False)
                    self.current_pos = (sx, sy, sz)
            elif sub_ct == CommandType.SET_POWER:
                power = sub_ops.power(j)
                if power > 0.0:
                    bias = (
                        self._power_floor
                        if self._layer_mode == "VECTOR"
                        else self._image_power_bias
                    )
                    self._emit_power(min(power + bias, 1.0))
                else:
                    self._emit_power(0.0)

        self.current_pos = end

    def _handle_arc_to(self, ops: Ops, idx: int) -> None:
        self._linearize_curve(ops, idx)

    def _handle_scan_line(self, ops: Ops, idx: int) -> None:
        self._linearize_curve(ops, idx)

    def _handle_bezier_to(self, ops: Ops, idx: int) -> None:
        self._linearize_curve(ops, idx)

    def _handle_quadratic_bezier_to(self, ops: Ops, idx: int) -> None:
        self._linearize_curve(ops, idx)

    def _handle_dwell(self, ops: Ops, idx: int) -> None:
        """Dwell is unsupported — the Ruida controller has no direct
        equivalent, so nothing is emitted."""
        logger.warning(
            "DWELL is not supported — the Ruida controller has no "
            "direct equivalent; no delay was emitted."
        )

    # -- Configuration handlers ---------------------------------------------

    def _handle_set_power(self, ops: Ops, idx: int) -> None:
        """Set laser power for the remaining cuts on this layer."""
        self._emit_power(ops.power(idx))

    def _handle_set_cut_speed(self, ops: Ops, idx: int) -> None:
        """Set cutting speed in mm/s."""
        self._require_active_layer()
        # ops.rate is in mm/min; GlueScript expects mm/s.
        self._gluescript.cut_speed(float(ops.rate(idx)) / 60.0)

    def _handle_set_travel_speed(self, ops: Ops, idx: int) -> None:
        """Set travel (rapid move) speed in mm/s."""
        self._require_active_layer()
        # ops.rate is in mm/min; GlueScript expects mm/s.
        self._gluescript.move_speed(float(ops.rate(idx)) / 60.0)

    def _handle_set_frequency(self, ops: Ops, idx: int) -> None:
        """Set laser frequency (Hz → KHz)."""
        freq_khz = ops.frequency(idx) / 1000.0
        self._require_active_layer()
        self._gluescript.frequency(freq_khz)

    def _handle_set_pulse_width(
        self,
        ops: Ops,
        idx: int,
    ) -> None:
        """Set laser pulse width in microseconds."""
        self._require_active_layer()
        self._gluescript.pwm(ops.pulse_width(idx))

    def _handle_coolant_as_air_assist(self, ops: Ops, idx: int) -> None:
        """Handle legacy SET_COOLANT used for air assist.

        ``ops.coolant()`` returns a ``CoolantMode`` enum (OFF/FLOOD/MIST),
        never the legacy ``"Air"`` string, so the comparison below never
        enables air assist. A non-OFF mode is logged as unsupported
        rather than silently dropped; use SET_AIR_ASSIST instead.
        """
        mode = ops.coolant(idx)
        if mode != CoolantMode.OFF:
            logger.warning(
                "SET_COOLANT %s is not acted upon — air assist "
                "requires SET_AIR_ASSIST",
                mode.name,
            )
        self._set_air_assist(mode == "Air")

    def _handle_set_air_assist(self, ops: Ops, idx: int) -> None:
        """Handle SET_AIR_ASSIST by reading AirAssistMode directly."""
        self._set_air_assist(ops.air_assist(idx) == AirAssistMode.ON)

    def _set_air_assist(self, enabled: bool) -> None:
        """Toggle air assist for the current layer."""
        self._require_active_layer()
        if enabled:
            self._gluescript.air_assist_on()
        else:
            self._gluescript.air_assist_off()

    def _handle_set_laser(
        self,
        ops: Ops,
        idx: int,
        machine: Machine,
    ) -> None:
        """Select laser device by resolving laser_uid to a tool number.

        Tries machine.heads first, then falls back to a deterministic
        device number derived from the laser_uid suffix.
        """
        laser_uid = ops.head_uid(idx)
        try:
            device = ((int(laser_uid.split("_")[-1]) - 1) % 2) + 1
        except (ValueError, IndexError):
            device = (sum(ord(c) for c in laser_uid) % 2) + 1

        try:
            laser_head = next(
                (head for head in machine.heads if head.uid == laser_uid),
                None,
            )
            if laser_head is not None:
                device = laser_head.tool_number
        except (AttributeError, TypeError):
            logger.debug(
                "machine.heads not available, falling back to "
                "parsed laser_uid modulo for "
                "1-based laser device selection"
            )

        if device == self.active_laser:
            return
        self.active_laser = device
        self._gluescript.select_laser(device)

    # -- Structural handlers ------------------------------------------------

    def _handle_job_start(self) -> None:
        """Declare the job in GlueScript, which emits the job header."""
        global _last_fallback_wcs
        self._job_started = True
        label = (
            self.doc.name
            if self.doc is not None and self.doc.name
            else _DEFAULT_JOB_LABEL
        )
        machine = self.machine
        if machine is None:
            ref_point = "MACHINE"
        else:
            wcs = machine.active_wcs
            if wcs in _WCS_TO_REF_POINT:
                _last_fallback_wcs = None
                ref_point = _WCS_TO_REF_POINT[wcs]
            elif wcs == "G54":
                if wcs != _last_fallback_wcs:
                    logger.warning(
                        "Active WCS %s is the framework default — "
                        "using the MACHINE reference point",
                        wcs,
                    )
                _last_fallback_wcs = wcs
                ref_point = "MACHINE"
            else:
                raise ValueError(
                    f"Unsupported WCS for Ruida reference point: {wcs!r} "
                    f"— valid names: {', '.join(_WCS_TO_REF_POINT)}"
                )
        self._gluescript.declare_job(label, ref_point, None, 1, 1, 0.0, 0.0)

    def _handle_layer_start(self, ops: Ops, idx: int) -> None:
        """Record the active layer uid for the next WORKPIECE_START.

        LAYER_START emits nothing; the layer is declared when the first
        workpiece of the layer begins, so each Rayforge workpiece maps
        to its own Ruida layer.
        """
        self._layer_uid = ops.layer_uid(idx)
        self._layer_key += 1
        self._layer_declared = False

    @staticmethod
    def _section_layer_mode(
        section_type: SectionType, raster_mode: RasterMode | None
    ) -> str:
        """Map a section's type and raster mode to a GlueScript layer mode.

        Non-raster sections are always "VECTOR". Raster fill sections
        map their raster mode: DITHER -> "DITHER", DEPTH_MAP ->
        "DEPTHMAP", CONSTANT_POWER -> "RASTER", VARIABLE_POWER ->
        "IMAGE", and any other raster mode defaults to "VECTOR".
        """
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

    def _compute_layer_mode(self, ops: Ops, idx: int) -> str:
        """Derive the layer mode from its first raster ops section.

        Scans forward from the WORKPIECE_START command to the next
        workpiece, layer, or job boundary. The first RASTER_FILL section
        determines the mode via _section_layer_mode; a layer without any
        raster section defaults to "VECTOR".
        """
        for i in range(idx + 1, ops.len()):
            command = ops.command_type(i)
            if command in (
                CommandType.LAYER_END,
                CommandType.LAYER_START,
                CommandType.WORKPIECE_END,
                CommandType.WORKPIECE_START,
                CommandType.JOB_END,
            ):
                break
            if command != CommandType.OPS_SECTION_START:
                continue
            section_type, _, raster_mode = ops.section_params(i)
            if section_type == SectionType.RASTER_FILL:
                return self._section_layer_mode(section_type, raster_mode)
        return "VECTOR"

    def _compute_overscan(
        self,
        ops: Ops,
        idx: int,
        layer_mode: str,
        section_bounded: bool = False,
    ) -> str:
        """Derive the layer overscan from its raster scan lines.

        Scans forward from the WORKPIECE_START (or OPS_SECTION_START for
        section-bounded scans) command to the next workpiece, layer, or
        job boundary, tracking the current position. The first
        non-degenerate raster line determines the overscan: horizontal
        lines yield "X_BI", vertical lines yield "Y_BI", and diagonal
        lines (unsupported by the Ruida controller) yield "NONE". Raster
        lines may be emitted as SCAN_LINE (per-pixel power) or as
        LINE_TO (constant-power fills such as the material test grid).
        Vector layers are forced to "NONE" by GlueScript's layer-mode
        override. The remaining stop commands are defensive stops for
        malformed sequences.
        """
        if layer_mode == "VECTOR":
            return "NONE"
        pos = (0.0, 0.0, 0.0)
        in_raster_fill = section_bounded
        for i in range(idx + 1, ops.len()):
            command = ops.command_type(i)
            if command in (
                CommandType.LAYER_END,
                CommandType.LAYER_START,
                CommandType.WORKPIECE_END,
                CommandType.WORKPIECE_START,
                CommandType.JOB_END,
            ):
                break
            if command in (
                CommandType.OPS_SECTION_END,
                CommandType.OPS_SECTION_START,
            ):
                if section_bounded:
                    break
                if command == CommandType.OPS_SECTION_START:
                    section_type, _, _ = ops.section_params(i)
                    in_raster_fill = section_type == SectionType.RASTER_FILL
                else:
                    in_raster_fill = False
                continue
            if command in (CommandType.MOVE_TO, CommandType.LINE_TO):
                if command == CommandType.LINE_TO and in_raster_fill:
                    overscan = self._classify_overscan(pos, ops.endpoint(i))
                    if overscan is not None:
                        return overscan
                pos = ops.endpoint(i)
                continue
            if command == CommandType.SCAN_LINE and in_raster_fill:
                overscan = self._classify_overscan(pos, ops.endpoint(i))
                if overscan is not None:
                    return overscan
        return "NONE"

    @staticmethod
    def _classify_overscan(start, end):
        """Classify a raster line direction as an overscan mode.

        Returns "X_BI" for horizontal lines, "Y_BI" for vertical lines,
        "NONE" for diagonal lines, and None for degenerate (zero-length)
        lines.
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if (
            abs(dx) < _OVERSCAN_ANGLE_EPSILON
            and abs(dy) < _OVERSCAN_ANGLE_EPSILON
        ):
            return None
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        if angle <= _OVERSCAN_ANGLE_TOLERANCE_DEG:
            return "X_BI"
        if angle >= 90.0 - _OVERSCAN_ANGLE_TOLERANCE_DEG:
            return "Y_BI"
        return "NONE"

    def _handle_job_end(self, idx: int) -> None:
        """Finalize the job in GlueScript, which emits END_JOB and EOF."""
        self._job_ended = True
        self._gluescript.end_job()

    def _handle_workpiece_start(self, ops: Ops, idx: int) -> None:
        """Declare the Ruida layer for this workpiece.

        Each Rayforge workpiece becomes its own Ruida layer, so the
        layer attribute block is emitted here (not at LAYER_START) using
        the active layer's settings and the mode/overscan derived from
        this workpiece's sections.
        """
        if self._layer_uid is None:
            raise ValueError(
                "WORKPIECE_START encountered before LAYER_START — "
                "GlueScript routing requires an active layer"
            )
        layer_uid = self._layer_uid
        wp_uid = ops.workpiece_uid(idx)
        self._gluescript.comment([f"# Workpiece Start uid={wp_uid}"])

        layer = self._find_layer(layer_uid)
        label = (
            layer.name if layer is not None else f"Layer {self._layer_key - 1}"
        )
        color = layer.color if layer is not None else _DEFAULT_LAYER_COLOR
        speed_mms, frequency_khz, power_pct = self._layer_settings(layer)
        layer_mode = self._compute_layer_mode(ops, idx)
        self._layer_mode = layer_mode
        overscan = self._compute_overscan(ops, idx, layer_mode)
        self._overscan = overscan
        self._power_min_fraction = self._layer_min_power_fraction(layer)
        if (
            layer_mode == "VECTOR"
            and overscan == "NONE"
            and self._power_min_fraction * 100.0 < power_pct
        ):
            min_power_1 = self._power_min_fraction * 100.0
            max_power_1 = power_pct
        else:
            min_power_1 = power_pct
            max_power_1 = power_pct
        self._gluescript.declare_layer(
            label=label,
            color=color,
            mode=layer_mode,
            overscan=overscan,
            speed=speed_mms,
            frequency=frequency_khz,
            min_power_1=min_power_1,
            max_power_1=max_power_1,
        )
        self._layer_declared = True

    def _handle_workpiece_end(self) -> None:
        """Emit a workpiece end marker and reset per-workpiece state."""
        self._overscan = "NONE"
        self._power_fraction = 0.0
        self._power_min_fraction = 0.0
        self._emitted_min_fraction = 0.0
        self._gluescript.comment(["# Workpiece End"])

    def _handle_ops_section_start(self, ops: Ops, idx: int) -> None:
        """Record the active section and apply per-section overscan changes.

        The Ruida layer was already declared at WORKPIECE_START; this
        records the section type and raster mode and, when the section's
        own overscan differs from the declared layer's, switches the
        controller overscan at the section boundary via set_overscan.
        GlueScript instances that lack set_overscan keep the declared
        layer overscan so movement forms stay consistent with the
        controller.
        """
        self._section_type, _workpiece_uid, self._section_raster_mode = (
            ops.section_params(idx)
        )
        self._gluescript.comment(
            [
                "# Ops Actions",
                "# Ops Section Start",
            ]
        )
        section_mode = self._section_layer_mode(
            self._section_type, self._section_raster_mode
        )
        section_overscan = self._compute_overscan(
            ops, idx, section_mode, section_bounded=True
        )
        if section_overscan != self._overscan:
            if hasattr(self._gluescript, "set_overscan"):
                self._gluescript.set_overscan(section_overscan)
                self._overscan = section_overscan
            else:
                logger.info(
                    "GlueScript lacks set_overscan — keeping the "
                    "declared layer overscan %s for section mode %s",
                    self._overscan,
                    section_mode,
                )

    def _handle_ops_section_end(self) -> None:
        """Clear the active section and emit a comment for its end."""
        self._section_type = None
        self._section_raster_mode = None
        self._gluescript.comment(["# Ops Section End"])

    # -- Op map bookkeeping --------------------------------------------------

    def _snapshot_sections(self) -> None:
        """Record the transcript length before dispatching the current op."""
        self._snapshot_len = len(self._gluescript.gluescript)

    def _record_contribution(self, op_index: int) -> None:
        """Record the transcript span the last op appended.

        The span is only recorded when the op actually appended lines.
        The guard also prevents an assert-crash when a pre-JOB_START op
        appends lines that declare_job()'s internal new_gluescript()
        wipes: the declare_job line then gets claimed by that op, which
        is acceptable misattribution for an invalid sequence.
        """
        gs = self._gluescript
        end = len(gs.gluescript)
        start = self._snapshot_len
        if end > start:
            self._op_contributions.setdefault(op_index, []).append(
                (start, end)
            )

    def _build_op_map(self, line_count: int) -> None:
        """Populate the op_map from the GlueScript transcript spans.

        Each op's contribution is a contiguous span of transcript lines
        recorded by _record_contribution. Spans are clamped to the final
        transcript length: declare_job() internally calls
        new_gluescript(), which wipes lines appended by any pre-JOB_START
        op, so a recorded end can exceed the final length.
        """
        line_spans: list[tuple[int, int]] = []
        machine_code_to_op = [-1] * line_count
        for op_index in range(self._op_count):
            contributions = self._op_contributions.get(op_index, [])
            start = 0
            end = 0
            for contribution in contributions:
                contrib_start, contrib_end = contribution
                if contrib_end > contrib_start:
                    start = contrib_start
                    end = contrib_end
                    break
            if end <= start:
                line_spans.append((0, 0))
                continue
            end = min(end, line_count)
            if end <= start:
                line_spans.append((0, 0))
                continue
            line_spans.append((start, end - start))
            for line_num in range(start, end):
                assert 0 <= line_num < line_count
                machine_code_to_op[line_num] = op_index

        self.op_map = MachineCodeOpMap.from_lists(
            line_spans, machine_code_to_op
        )

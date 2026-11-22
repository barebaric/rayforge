"""Tests for power handling across workpiece boundaries.

Workpieces whose attribute block matches the open Ruida layer reuse it
instead of declaring a new one — controllers limit the number of
layers. Within an open layer the power context persists, so duplicated
workpieces whose SET_POWER was hoisted to the first workpiece by the
path optimizer keep cutting at the last requested power. A workpiece
that opens a new Ruida layer gets the last requested power re-applied
after its declaration (issue #564).
"""

from typing import Any
from unittest.mock import MagicMock

from raygeo.ops import Ops
from raygeo.ops.types import RasterMode, SectionType

from rayforge.core.doc import Doc
from rayforge.core.step import Step
from rayforge.machine.driver.ruidarpa.rpa_encoder import RuidaRPAEncoder
from rayforge.machine.models.laser import Laser
from rayforge.machine.models.machine import Machine


class CutStep(Step):
    """Minimal concrete Step for encoder layer-settings tests."""

    power: float
    frequency: float

    def __init__(self):
        super().__init__(typelabel="cut")


def _build_machine() -> Machine:
    context: Any = MagicMock()
    machine = Machine(context)
    laser1 = Laser()
    laser1.uid = "laser-1"
    laser1.tool_number = 1
    machine.heads.clear()
    machine.add_head(laser1)
    return machine


def _build_doc() -> Doc:
    doc = Doc()
    cut_step = CutStep()
    cut_step.power = 0.5
    cut_step.cut_speed = 300
    cut_step.frequency = 20000
    workflow = doc.layers[0].workflow
    assert workflow is not None
    workflow.add_step(cut_step)
    return doc


def _build_duplicated_workpieces_job(doc: Doc) -> Ops:
    """One layer, three workpieces, SET_POWER only inside the first."""
    layer_uid = doc.layers[0].uid
    ops = Ops()
    ops.job_start()
    ops.layer_start(layer_uid=layer_uid)
    for index in range(3):
        wp_uid = f"wp-{index}"
        ops.workpiece_start(wp_uid)
        ops.ops_section_start(SectionType.VECTOR_OUTLINE, wp_uid)
        if index == 0:
            ops.set_power(0.6)
            ops.set_feed_rate(250)
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(5.0, 5.0, 0.0)
        ops.ops_section_end(SectionType.VECTOR_OUTLINE)
        ops.workpiece_end(wp_uid)
    ops.layer_end(layer_uid=layer_uid)
    ops.job_end()
    return ops


def _build_mode_changing_job(doc: Doc) -> Ops:
    """One layer; the second workpiece switches to a raster fill."""
    layer_uid = doc.layers[0].uid
    ops = Ops()
    ops.job_start()
    ops.layer_start(layer_uid=layer_uid)
    ops.workpiece_start("wp-0")
    ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-0")
    ops.set_power(0.6)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(5.0, 5.0, 0.0)
    ops.ops_section_end(SectionType.VECTOR_OUTLINE)
    ops.workpiece_end("wp-0")
    ops.workpiece_start("wp-1")
    ops.ops_section_start(
        SectionType.RASTER_FILL,
        "wp-1",
        raster_mode=RasterMode.VARIABLE_POWER,
    )
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(5.0, 0.0, 0.0)
    ops.ops_section_end(
        SectionType.RASTER_FILL,
        raster_mode=RasterMode.VARIABLE_POWER,
    )
    ops.workpiece_end("wp-1")
    ops.layer_end(layer_uid=layer_uid)
    ops.job_end()
    return ops


class TestDuplicateWorkpiecePower:
    """Duplicated workpieces must keep the layer's requested power."""

    def test_duplicates_share_one_layer(self):
        doc = _build_doc()
        ops = _build_duplicated_workpieces_job(doc)
        text = RuidaRPAEncoder().encode(ops, _build_machine(), doc).text

        declared = [
            line
            for line in text.split("\n")
            if line.startswith("declare_layer(")
        ]
        assert len(declared) == 1

    def test_power_survives_later_workpiece_starts(self):
        doc = _build_doc()
        ops = _build_duplicated_workpieces_job(doc)
        text = RuidaRPAEncoder().encode(ops, _build_machine(), doc).text

        lines = text.split("\n")
        power_lines = [
            index
            for index, line in enumerate(lines)
            if line.startswith("power_range(")
        ]
        second_wp_start = min(
            index
            for index, line in enumerate(lines)
            if "Workpiece Start uid=wp-1" in line
        )
        assert len(power_lines) == 1
        assert power_lines[0] < second_wp_start

    def test_no_power_emitted_without_set_power_op(self):
        doc = _build_doc()
        layer_uid = doc.layers[0].uid
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=layer_uid)
        ops.workpiece_start("wp-0")
        ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-0")
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(5.0, 5.0, 0.0)
        ops.ops_section_end(SectionType.VECTOR_OUTLINE)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=layer_uid)
        ops.job_end()

        text = RuidaRPAEncoder().encode(ops, _build_machine(), doc).text
        assert "power_range(" not in text


class TestModeChangePower:
    """A new Ruida layer must carry the last requested power."""

    def test_new_layer_reapplies_last_requested_power(self):
        doc = _build_doc()
        ops = _build_mode_changing_job(doc)
        text = RuidaRPAEncoder().encode(ops, _build_machine(), doc).text

        lines = text.split("\n")
        declare_lines = [
            index
            for index, line in enumerate(lines)
            if line.startswith("declare_layer(")
        ]
        assert len(declare_lines) == 2
        following = [
            line
            for line in lines[declare_lines[1] + 1 :]
            if not line.startswith("comment(")
        ]
        assert following[0].startswith("power_range(")

"""Behavioral tests for the Ruida encoder bridge.

The encoder policy lives in raydriver; these tests pin the Rayforge
side: the context built from the machine/document, and the resulting
transcript for the policies the application controls (power floor,
image bias, overscan, layer reuse and WCS mapping).
"""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from raygeo.ops import Ops
from raygeo.ops.types import RasterMode, SectionType

from rayforge.core.doc import Doc
from rayforge.core.step import Step
from rayforge.machine.models.laser import Laser
from rayforge.machine.models.machine import Machine
from rayforge.pipeline.encode_inputs import build_ruida_spec, encode_ops


class CutStep(Step):
    power: float
    frequency: float

    def __init__(self):
        super().__init__(typelabel="cut")


def _machine(**driver_args: Any) -> Machine:
    context: Any = MagicMock()
    machine = Machine(context)
    laser = Laser()
    laser.uid = "laser-1"
    laser.tool_number = 1
    machine.heads.clear()
    machine.add_head(laser)
    machine.driver_args = dict(driver_args)
    return machine


def _layer(doc: Doc, index: int = 0):
    doc.layers[index].name = "Cut"
    doc.layers[index].color = "#ff6600"
    step = CutStep()
    step.power = 0.5
    step.cut_speed = 300
    step.frequency = 20000
    workflow = doc.layers[index].workflow
    assert workflow is not None
    workflow.add_step(step)
    return doc.layers[index]


def _vector_job(layer_uid: str) -> Ops:
    ops = Ops()
    ops.job_start()
    ops.layer_start(layer_uid=layer_uid)
    ops.workpiece_start("wp")
    ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp")
    ops.set_power(0.6)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(5.0, 5.0, 0.0)
    ops.ops_section_end(SectionType.VECTOR_OUTLINE)
    ops.workpiece_end("wp")
    ops.layer_end(layer_uid=layer_uid)
    ops.job_end()
    return ops


def _encode(machine: Machine, doc: Doc, ops: Ops) -> str:
    return encode_ops(ops, build_ruida_spec(machine, doc)).text


class TestContext:
    def test_wcs_ref_point_mapping(self):
        machine = _machine()
        doc = Doc()
        machine.active_wcs = "ANCHOR"
        context = json.loads(build_ruida_spec(machine, doc).context_json)
        assert context["ref_point"] == "ABSOLUTE"

    def test_g54_falls_back_to_machine(self):
        machine = _machine()
        doc = Doc()
        machine.active_wcs = "G54"
        context = json.loads(build_ruida_spec(machine, doc).context_json)
        assert context["ref_point"] == "MACHINE"

    def test_unsupported_wcs_raises(self):
        machine = _machine()
        doc = Doc()
        machine.active_wcs = "G99"
        with pytest.raises(ValueError, match="Unsupported WCS"):
            build_ruida_spec(machine, doc)

    def test_power_policy_from_driver_args(self):
        machine = _machine(
            power_floor=12.5,
            power_floor_enabled=True,
            image_power_bias=9.0,
            power_scaling_enabled=False,
        )
        doc = Doc()
        context = json.loads(build_ruida_spec(machine, doc).context_json)
        assert context["power_floor_pct"] == 12.5
        assert context["power_floor_enabled"] is True
        assert context["image_power_bias_pct"] == 9.0
        assert context["power_scaling_enabled"] is False


class TestTranscript:
    def test_vector_layer_declaration(self):
        machine = _machine()
        doc = Doc()
        layer = _layer(doc)
        text = _encode(machine, doc, _vector_job(layer.uid))
        assert (
            "declare_layer('Cut', '#ff6600', 'VECTOR', 'NONE', "
            "5.0, 20.0, 50.0, 50.0)" in text
        )

    def test_power_floor_enables_min_power_ramp(self):
        machine = _machine(power_floor=20.0, power_floor_enabled=True)
        doc = Doc()
        layer = _layer(doc)
        ops = _vector_job(layer.uid)
        text = _encode(machine, doc, ops)
        assert "power_range(20.0, 60.0)" in text

    def test_power_floor_disabled_emits_constant_power(self):
        machine = _machine(power_floor=20.0, power_floor_enabled=False)
        doc = Doc()
        layer = _layer(doc)
        text = _encode(machine, doc, _vector_job(layer.uid))
        assert "power_range(60.0, 60.0)" in text
        assert "power_range(20.0, 60.0)" not in text

    def test_horizontal_raster_uses_x_overscan(self):
        machine = _machine()
        doc = Doc()
        layer = _layer(doc)
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=layer.uid)
        ops.workpiece_start("wp")
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp",
            raster_mode=RasterMode.CONSTANT_POWER,
        )
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(10.0, 0.0, 0.0)
        ops.ops_section_end(
            SectionType.RASTER_FILL,
            raster_mode=RasterMode.CONSTANT_POWER,
        )
        ops.workpiece_end("wp")
        ops.layer_end(layer_uid=layer.uid)
        ops.job_end()
        text = _encode(machine, doc, ops)
        assert "'RASTER', 'X_BI'" in text

    def test_two_workpieces_reuse_the_layer(self):
        machine = _machine()
        doc = Doc()
        layer = _layer(doc)
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=layer.uid)
        for name in ("wp-1", "wp-2"):
            ops.workpiece_start(name)
            ops.ops_section_start(SectionType.VECTOR_OUTLINE, name)
            ops.set_power(0.5)
            ops.move_to(0.0, 0.0, 0.0)
            ops.line_to(1.0, 1.0, 0.0)
            ops.ops_section_end(SectionType.VECTOR_OUTLINE)
            ops.workpiece_end(name)
        ops.layer_end(layer_uid=layer.uid)
        ops.job_end()
        text = _encode(machine, doc, ops)
        assert text.count("declare_layer(") == 1

    def test_op_map_covers_every_op(self):
        machine = _machine()
        doc = Doc()
        layer = _layer(doc)
        ops = _vector_job(layer.uid)
        encoded = encode_ops(ops, build_ruida_spec(machine, doc))
        assert encoded.op_map.op_count == ops.len()
        for op_index in range(ops.len()):
            assert encoded.op_map.span_for_op(op_index) is not None

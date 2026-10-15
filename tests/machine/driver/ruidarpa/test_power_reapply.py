"""Tests for re-applying power at workpiece boundaries.

The path optimizer hoists duplicate SET_POWER commands to the first
workpiece of a layer. Each workpiece maps to its own Ruida layer whose
power context resets at the boundary, so without a re-applied power the
controller cuts later workpieces with the laser off (issue #564).
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from raygeo.ops import Ops
from raygeo.ops.types import SectionType

from rayforge.core.doc import Doc
from rayforge.core.step import Step
from rayforge.machine.driver.ruidarpa.rpa_encoder import RuidaRPAEncoder
from rayforge.machine.models.laser import Laser
from rayforge.machine.models.machine import Machine

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))


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


def _build_duplicated_workpieces_job():
    """One layer, three workpieces, SET_POWER only before the first."""
    doc = Doc()
    cut_step = CutStep()
    cut_step.power = 0.5
    cut_step.cut_speed = 300
    cut_step.frequency = 20000
    workflow = doc.layers[0].workflow
    assert workflow is not None
    workflow.add_step(cut_step)

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
    return doc, ops, _build_machine()


class TestPowerReapply:
    """Each workpiece must carry a power command after its declaration."""

    def test_every_workpiece_gets_power(self):
        doc, ops, machine = _build_duplicated_workpieces_job()
        text = RuidaRPAEncoder().encode(ops, machine, doc).text

        lines = text.split("\n")
        blocks: list[list[str]] = []
        for line in lines:
            if "'# Workpiece Start" in line:
                blocks.append([])
            elif blocks:
                blocks[-1].append(line)
        assert len(blocks) == 3
        for index, block in enumerate(blocks):
            assert any(line.startswith("power_range(") for line in block), (
                f"workpiece {index} has no power command"
            )

    def test_no_power_emitted_without_set_power_op(self):
        doc, _ops, machine = _build_duplicated_workpieces_job()
        ops2 = Ops()
        ops2.job_start()
        layer_uid = doc.layers[0].uid
        ops2.layer_start(layer_uid=layer_uid)
        ops2.workpiece_start("wp-0")
        ops2.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-0")
        ops2.move_to(0.0, 0.0, 0.0)
        ops2.line_to(5.0, 5.0, 0.0)
        ops2.ops_section_end(SectionType.VECTOR_OUTLINE)
        ops2.workpiece_end("wp-0")
        ops2.layer_end(layer_uid=layer_uid)
        ops2.job_end()

        text = RuidaRPAEncoder().encode(ops2, machine, doc).text
        assert "power_range(" not in text

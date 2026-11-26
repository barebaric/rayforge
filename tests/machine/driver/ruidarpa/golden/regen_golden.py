"""Regenerate the ruidarpa golden fixture.

Run from the repo root inside the default pixi environment:

    pixi run python \\
        tests/machine/driver/ruidarpa/golden/regen_golden.py

The fixture locks the encoder's GlueScript transcript byte-for-byte.
Regenerate it only when the encoder or upstream GlueScript legitimately
changes the transcript; the golden tests then confirm the new bytes.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from raygeo.ops import Ops
from raygeo.ops.state import AirAssistMode
from raygeo.ops.types import RasterMode, SectionType

from rayforge.core.doc import Doc
from rayforge.core.step import Step
from rayforge.machine.driver.ruidarpa.rpa_encoder import RuidaRPAEncoder
from rayforge.machine.models.laser import Laser
from rayforge.machine.models.machine import Machine

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

GOLDEN_FILE = Path(__file__).resolve().parent / "transcript.cglu"


class CutStep(Step):
    """Minimal concrete Step for encoder layer-settings tests."""

    power: float
    frequency: float

    def __init__(self):
        super().__init__(typelabel="cut")


def _build_machine():
    """Build the two-laser machine used by the representative job."""
    context: Any = MagicMock()
    machine = Machine(context)
    laser1 = Laser()
    laser1.uid = "laser-1"
    laser1.tool_number = 1
    laser2 = Laser()
    laser2.uid = "laser-2"
    laser2.tool_number = 2
    machine.heads.clear()
    machine.add_head(laser1)
    machine.add_head(laser2)
    return machine


def build_representative_job():
    """Return (doc, ops, machine) for the golden-fixture representative job.

    The job exercises two workflow-driven layers plus one defaults
    layer, covering raw power pass-through, near/far move and cut forms,
    a linearized arc, a scan line, and per-op settings.
    """
    doc = Doc()
    for index, name in enumerate(("Cut", "Engrave", "Default")):
        doc.layers[index].name = name
    doc.layers[0].color = "#ff6600"
    doc.layers[1].color = "#33cc33"
    doc.layers[2].color = "#00ccff"

    cut_step = CutStep()
    cut_step.power = 0.5
    cut_step.cut_speed = 300
    cut_step.frequency = 20000
    cut_workflow = doc.layers[0].workflow
    assert cut_workflow is not None
    cut_workflow.add_step(cut_step)

    engrave_step = CutStep()
    engrave_step.power = 0.5  # 50%
    engrave_step.cut_speed = 150
    engrave_step.frequency = 30000
    engrave_workflow = doc.layers[1].workflow
    assert engrave_workflow is not None
    engrave_workflow.add_step(engrave_step)

    ops = Ops()
    ops.job_start()

    # Layer 0: Cut — vector outline, per-op settings, near move/cut,
    # arc, air assist.
    layer0 = doc.layers[0].uid
    ops.layer_start(layer_uid=layer0)
    ops.workpiece_start("wp-0")
    ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-0")
    ops.set_power(0.6)
    ops.set_feed_rate(250)
    ops.set_frequency(25000)
    ops.set_pulse_width(50)
    ops.set_air_assist(AirAssistMode.ON)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(5.0, 5.0, 0.0)
    ops.arc_to(10.0, 0.0, 5.0, 0.0, clockwise=True)
    ops.set_air_assist(AirAssistMode.OFF)
    ops.dwell(250)
    ops.ops_section_end(SectionType.VECTOR_OUTLINE)
    ops.workpiece_end("wp-0")
    ops.layer_end(layer_uid=layer0)

    # Layer 1: Engrave — image section, low-power pass-through, far
    # move/cut, scan line.
    layer1 = doc.layers[1].uid
    ops.layer_start(layer_uid=layer1)
    ops.workpiece_start("wp-1")
    ops.ops_section_start(
        SectionType.RASTER_FILL, "wp-1", raster_mode=RasterMode.VARIABLE_POWER
    )
    ops.set_power(0.05)  # low power passes through unclamped on power()
    ops.set_feed_rate(150)
    ops.set_frequency(30000)
    ops.move_to(20.0, 20.0, 0.0)
    ops.line_to(50.0, 20.0, 0.0)
    ops.move_to(0.0, 0.0, 0.0)
    power_values = bytearray([0, 128, 255, 128, 0])
    ops.scan_to(5.0, 0.0, 0.0, power_values)
    ops.ops_section_end(
        SectionType.RASTER_FILL, raster_mode=RasterMode.VARIABLE_POWER
    )
    ops.workpiece_end("wp-1")
    ops.layer_end(layer_uid=layer1)

    # Layer 2: Default — vector outline, no workflow steps, encoder
    # defaults apply.
    layer2 = doc.layers[2].uid
    ops.layer_start(layer_uid=layer2)
    ops.workpiece_start("wp-2")
    ops.ops_section_start(SectionType.VECTOR_OUTLINE, "wp-2")
    ops.move_to(55.0, 22.0, 0.0)
    ops.line_to(57.0, 22.0, 0.0)
    ops.ops_section_end(SectionType.VECTOR_OUTLINE)
    ops.workpiece_end("wp-2")
    ops.layer_end(layer_uid=layer2)

    ops.job_end()
    return doc, ops, _build_machine()


def main():
    """Encode the representative job and write the golden fixture."""
    doc, ops, machine = build_representative_job()
    result = RuidaRPAEncoder().encode(ops, machine, doc)
    GOLDEN_FILE.write_text(result.text + "\n", encoding="utf-8")
    line_count = len(result.text.split("\n"))
    print(f"Wrote {GOLDEN_FILE} ({line_count} lines)")


if __name__ == "__main__":
    main()

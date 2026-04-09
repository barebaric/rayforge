import pytest
import math

from rayforge.core.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    SetPowerCommand,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
)
from rayforge.pipeline.transformer.base import ExecutionPhase
from post_processors.transformers import LeadInOutTransformer


@pytest.fixture
def transformer() -> LeadInOutTransformer:
    return LeadInOutTransformer(enabled=True, lead_in_mm=5.0, lead_out_mm=5.0)


def test_initialization_and_properties():
    t = LeadInOutTransformer(enabled=True, lead_in_mm=2.5, lead_out_mm=3.5)
    assert t.enabled is True
    assert t.lead_in_mm == 2.5
    assert t.lead_out_mm == 3.5
    t.lead_in_mm = -10.0
    assert t.lead_in_mm == 0.0
    t.lead_out_mm = -5.0
    assert t.lead_out_mm == 0.0
    t.lead_in_mm = 7.0
    t.lead_out_mm = 8.0
    assert t.lead_in_mm == 7.0
    assert t.lead_out_mm == 8.0


def test_serialization_and_deserialization():
    original = LeadInOutTransformer(
        enabled=False, lead_in_mm=3.14, lead_out_mm=2.71, auto=False
    )
    data = original.to_dict()
    recreated = LeadInOutTransformer.from_dict(data)
    assert data["name"] == "LeadInOutTransformer"
    assert data["enabled"] is False
    assert data["lead_in_mm"] == 3.14
    assert data["lead_out_mm"] == 2.71
    assert isinstance(recreated, LeadInOutTransformer)
    assert recreated.enabled is False
    assert recreated.lead_in_mm == 3.14
    assert recreated.lead_out_mm == 2.71


def test_no_op_when_disabled(transformer: LeadInOutTransformer):
    ops = Ops()
    ops.add(OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.line_to(30, 10, 0)
    ops.line_to(30, 30, 0)
    ops.line_to(10, 30, 0)
    ops.line_to(10, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
    original_commands_list = ops.commands

    transformer.enabled = False
    transformer.run(ops)

    assert ops.commands is original_commands_list


def test_no_op_with_zero_distances(transformer: LeadInOutTransformer):
    ops = Ops()
    ops.add(OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.line_to(30, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
    original_commands_list = ops.commands

    transformer.lead_in_mm = 0.0
    transformer.lead_out_mm = 0.0
    transformer.run(ops)

    assert ops.commands is original_commands_list


def test_execution_phase(transformer: LeadInOutTransformer):
    assert transformer.execution_phase == ExecutionPhase.POST_PROCESSING


def test_square_contour_with_both_lead_in_out(
    transformer: LeadInOutTransformer,
):
    ops = Ops()
    ops.add(SetPowerCommand(0.8))
    ops.add(OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.line_to(30, 10, 0)
    ops.line_to(30, 30, 0)
    ops.line_to(10, 30, 0)
    ops.line_to(10, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
    transformer.run(ops)

    cmds = ops.commands
    # Expected: SP(0.8), Start, Move, SP(0), Line, SP(0.8), Line*4,
    #           SP(0), Line, End
    assert isinstance(cmds[1], OpsSectionStartCommand)
    assert isinstance(cmds[2], MoveToCommand)
    # Lead-in start: 10-5=5, 10 (opposite of first segment direction)
    assert cmds[2].end == pytest.approx((5.0, 10.0, 0.0))
    # Lead-in line to original start
    assert isinstance(cmds[3], SetPowerCommand) and cmds[3].power == 0
    assert isinstance(cmds[4], LineToCommand)
    assert cmds[4].end == pytest.approx((10.0, 10.0, 0.0))
    # Content
    assert isinstance(cmds[5], SetPowerCommand) and cmds[5].power == 0.8
    # Lead-out at end
    lead_out = cmds[-2]
    assert isinstance(lead_out, LineToCommand)
    # Last segment direction is (10-10, 10-30) = (0, -20) normalized = (0, -1)
    # Lead-out end: (10, 10) + 5*(0, -1) = (10, 5)
    assert lead_out.end == pytest.approx((10.0, 5.0, 0.0))
    assert isinstance(cmds[-1], OpsSectionEndCommand)


def test_lead_in_only(transformer: LeadInOutTransformer):
    transformer.lead_out_mm = 0.0
    ops = Ops()
    ops.add(SetPowerCommand(0.8))
    ops.add(OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.line_to(30, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
    transformer.run(ops)

    cmds = ops.commands
    move_cmds = [c for c in cmds if isinstance(c, MoveToCommand)]
    assert len(move_cmds) == 1
    assert move_cmds[0].end == pytest.approx((5.0, 10.0, 0.0))

    # No lead-out: last command before section end should be a content
    # LineTo, not a zero-power one
    assert isinstance(cmds[-2], LineToCommand)
    assert cmds[-2].end == pytest.approx((30.0, 10.0, 0.0))


def test_lead_out_only(transformer: LeadInOutTransformer):
    transformer.lead_in_mm = 0.0
    ops = Ops()
    ops.add(SetPowerCommand(0.8))
    ops.add(OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.line_to(30, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
    transformer.run(ops)

    cmds = ops.commands
    # MoveTo should be unchanged
    assert isinstance(cmds[2], MoveToCommand)
    assert cmds[2].end == pytest.approx((10.0, 10.0, 0.0))

    # Lead-out at end
    lead_out = cmds[-2]
    assert isinstance(lead_out, LineToCommand)
    assert lead_out.end == pytest.approx((35.0, 10.0, 0.0))


def test_diagonal_contour(transformer: LeadInOutTransformer):
    ops = Ops()
    ops.add(SetPowerCommand(0.5))
    ops.add(OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "wp_123"))
    ops.move_to(0, 0, 0)
    ops.line_to(10, 10, 0)
    ops.line_to(0, 0, 0)
    ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
    transformer.run(ops)

    cmds = ops.commands
    move_cmd = cmds[2]
    assert isinstance(move_cmd, MoveToCommand)
    # First segment: (10,10) - (0,0) = (10,10), normalized = (1/√2, 1/√2)
    # Lead-in start: (0,0) - 5*(1/√2, 1/√2) = (-5/√2, -5/√2)
    norm = 1.0 / math.sqrt(2)
    assert move_cmd.end == pytest.approx((-5.0 * norm, -5.0 * norm, 0.0))

    lead_out = cmds[-2]
    assert isinstance(lead_out, LineToCommand)
    # Last segment: (0,0) - (10,10) = (-10,-10), normalized = (-1/√2, -1/√2)
    # Lead-out end: (0,0) + 5*(-1/√2, -1/√2) = (-5/√2, -5/√2)
    assert lead_out.end == pytest.approx((-5.0 * norm, -5.0 * norm, 0.0))


def test_does_not_modify_commands_outside_vector_section(
    transformer: LeadInOutTransformer,
):
    ops = Ops()
    ops.move_to(0, 0, 0)
    ops.line_to(5, 5, 0)
    ops.add(OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.line_to(20, 10, 0)
    ops.line_to(10, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
    original_vector_cmds = ops.commands[:2]

    transformer.run(ops)

    assert ops.commands[0] is original_vector_cmds[0]
    assert ops.commands[1] is original_vector_cmds[1]
    assert len(ops.commands) > 8


def test_does_not_modify_raster_sections(transformer: LeadInOutTransformer):
    ops = Ops()
    ops.add(OpsSectionStartCommand(SectionType.RASTER_FILL, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.line_to(30, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.RASTER_FILL))
    original_cmds = ops.commands[:]

    transformer.run(ops)

    assert ops.commands == original_cmds


def test_handles_zero_length_first_segment(transformer: LeadInOutTransformer):
    ops = Ops()
    ops.add(SetPowerCommand(0.8))
    ops.add(OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.line_to(10, 10, 0)
    ops.line_to(30, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
    transformer.run(ops)

    cmds = ops.commands
    # Lead-in is skipped because first segment has zero length,
    # but lead-out should still be applied using the last segment's
    # tangent.
    move_cmds = [c for c in cmds if isinstance(c, MoveToCommand)]
    assert len(move_cmds) == 1
    assert move_cmds[0].end == pytest.approx((10.0, 10.0, 0.0))

    # Lead-out: last segment is (10,10)->(30,10), tangent=(1,0)
    # Lead-out end: (30,10) + 5*(1,0) = (35, 10)
    lead_out = cmds[-2]
    assert isinstance(lead_out, LineToCommand)
    assert lead_out.end == pytest.approx((35.0, 10.0, 0.0))


def test_handles_multiple_contours_in_section(
    transformer: LeadInOutTransformer,
):
    ops = Ops()
    ops.add(SetPowerCommand(0.8))
    ops.add(OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "wp_123"))
    # First contour
    ops.move_to(10, 10, 0)
    ops.line_to(30, 10, 0)
    ops.line_to(10, 10, 0)
    # Second contour (separate MoveTo)
    ops.move_to(50, 50, 0)
    ops.line_to(70, 50, 0)
    ops.line_to(50, 50, 0)
    ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
    transformer.run(ops)

    cmds = ops.commands
    move_cmds = [c for c in cmds if isinstance(c, MoveToCommand)]
    assert len(move_cmds) == 2
    # First contour lead-in: (10,10) - 5*(1,0) = (5, 10)
    assert move_cmds[0].end == pytest.approx((5.0, 10.0, 0.0))
    # Second contour lead-in: (50,50) - 5*(1,0) = (45, 50)
    assert move_cmds[1].end == pytest.approx((45.0, 50.0, 0.0))


def test_auto_distance_calculation():
    distance = LeadInOutTransformer.calculate_auto_distance(
        step_speed=3000, max_acceleration=1000
    )
    # speed = 50 mm/s, d = 50^2 / (2 * 1000 * 2) = 2500 / 4000 = 0.625
    assert distance == pytest.approx(0.625)

    distance = LeadInOutTransformer.calculate_auto_distance(
        step_speed=6000, max_acceleration=500
    )
    # speed = 100 mm/s, d = 10000 / 2000 = 5.0
    assert distance == pytest.approx(5.0)


def test_auto_distance_minimum():
    distance = LeadInOutTransformer.calculate_auto_distance(
        step_speed=100, max_acceleration=5000
    )
    # speed = 1.667 mm/s, very small, should return minimum 0.5
    assert distance == 0.5


def test_with_z_height(transformer: LeadInOutTransformer):
    ops = Ops()
    ops.add(SetPowerCommand(0.8))
    ops.add(OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "wp_123"))
    ops.move_to(10, 10, 3.0)
    ops.line_to(30, 10, 3.0)
    ops.line_to(10, 10, 3.0)
    ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
    transformer.run(ops)

    cmds = ops.commands
    move_cmd = cmds[2]
    assert isinstance(move_cmd, MoveToCommand)
    assert move_cmd.end == pytest.approx((5.0, 10.0, 3.0))

    lead_out = cmds[-2]
    assert isinstance(lead_out, LineToCommand)
    assert lead_out.end[2] == pytest.approx(3.0)


def test_separate_lead_in_out_distances():
    t = LeadInOutTransformer(enabled=True, lead_in_mm=3.0, lead_out_mm=7.0)
    ops = Ops()
    ops.add(SetPowerCommand(0.8))
    ops.add(OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, "wp_123"))
    ops.move_to(10, 10, 0)
    ops.line_to(30, 10, 0)
    ops.line_to(10, 10, 0)
    ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
    t.run(ops)

    cmds = ops.commands
    move_cmd = next(c for c in cmds if isinstance(c, MoveToCommand))
    # Lead-in: 10 - 3 = 7
    assert move_cmd.end == pytest.approx((7.0, 10.0, 0.0))

    lead_out = cmds[-2]
    assert isinstance(lead_out, LineToCommand)
    # Lead-out: 10 - 7 = 3 (last segment goes right to left)
    assert lead_out.end == pytest.approx((3.0, 10.0, 0.0))

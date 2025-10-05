"""Tests for the preview widget and timeline."""

from rayforge.core.ops import Ops, MoveToCommand, LineToCommand
from rayforge.workbench.simulation_widget import OpsTimeline, PreviewRenderer


def test_ops_timeline_empty():
    """Test timeline with no operations."""
    timeline = OpsTimeline()
    assert timeline.get_step_count() == 0
    assert timeline.get_steps_up_to(0) == []


def test_ops_timeline_basic():
    """Test timeline with basic operations."""
    ops = Ops()
    ops.set_power(0.5)
    ops.set_cut_speed(1000)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    ops.line_to(10.0, 10.0, 0.0)

    timeline = OpsTimeline(ops)

    # Should have 3 moving commands
    assert timeline.get_step_count() == 3

    # Check first step
    steps = timeline.get_steps_up_to(0)
    assert len(steps) == 1
    cmd, state, start_pos = steps[0]
    assert isinstance(cmd, MoveToCommand)
    assert state.power == 0.5
    assert state.cut_speed == 1000
    assert start_pos == (0.0, 0.0, 0.0)


def test_ops_timeline_power_changes():
    """Test that power changes are tracked correctly."""
    ops = Ops()
    ops.set_power(0.2)
    ops.line_to(5.0, 0.0, 0.0)
    ops.set_power(0.8)
    ops.line_to(10.0, 0.0, 0.0)

    timeline = OpsTimeline(ops)
    assert timeline.get_step_count() == 2

    # First line should have power=0.2
    steps = timeline.get_steps_up_to(0)
    _, state1, _ = steps[0]
    assert state1.power == 0.2

    # Second line should have power=0.8
    steps = timeline.get_steps_up_to(1)
    _, state2, _ = steps[1]
    assert state2.power == 0.8


def test_ops_timeline_position_tracking():
    """Test that start positions are tracked correctly."""
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    ops.line_to(10.0, 10.0, 0.0)

    timeline = OpsTimeline(ops)

    steps = timeline.get_steps_up_to(2)

    # First command starts at origin
    _, _, start1 = steps[0]
    assert start1 == (0.0, 0.0, 0.0)

    # Second command starts where first ended
    _, _, start2 = steps[1]
    assert start2 == (0.0, 0.0, 0.0)

    # Third command starts where second ended
    _, _, start3 = steps[2]
    assert start3 == (10.0, 0.0, 0.0)


def test_preview_renderer_initialization():
    """Test renderer initialization."""
    renderer = PreviewRenderer(400, 300)
    assert renderer.width == 400
    assert renderer.height == 300
    assert renderer.bounds == (0.0, 0.0, 100.0, 100.0)


def test_preview_renderer_resize():
    """Test renderer resize."""
    renderer = PreviewRenderer(400, 300)
    renderer.resize(800, 600)
    assert renderer.width == 800
    assert renderer.height == 600


def test_preview_renderer_set_bounds():
    """Test setting bounds."""
    renderer = PreviewRenderer(400, 300)
    renderer.set_bounds(0.0, 0.0, 50.0, 50.0)
    assert renderer.bounds == (0.0, 0.0, 50.0, 50.0)


def test_preview_renderer_render_empty():
    """Test rendering with no steps."""
    renderer = PreviewRenderer(400, 300)
    surface = renderer.render([])
    assert surface is not None
    assert surface.get_width() == 400
    assert surface.get_height() == 300


def test_preview_renderer_render_with_steps():
    """Test rendering with actual steps."""
    from rayforge.core.ops.commands import State

    renderer = PreviewRenderer(400, 300)
    renderer.set_bounds(0.0, 0.0, 20.0, 20.0)

    # Create some test steps
    state = State(power=0.5, cut_speed=1000)
    steps = [
        (MoveToCommand((0.0, 0.0, 0.0)), state, (0.0, 0.0, 0.0)),
        (LineToCommand((10.0, 0.0, 0.0)), state, (0.0, 0.0, 0.0)),
        (LineToCommand((10.0, 10.0, 0.0)), state, (10.0, 0.0, 0.0)),
    ]

    surface = renderer.render(steps)
    assert surface is not None
    assert surface.get_width() == 400
    assert surface.get_height() == 300


def test_get_steps_up_to_negative():
    """Test get_steps_up_to with negative index."""
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)

    timeline = OpsTimeline(ops)
    steps = timeline.get_steps_up_to(-1)
    assert steps == []


def test_get_steps_up_to_beyond_end():
    """Test get_steps_up_to with index beyond timeline."""
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)

    timeline = OpsTimeline(ops)
    steps = timeline.get_steps_up_to(100)

    # Should return all steps (slicing is forgiving)
    assert len(steps) == 2


def test_ops_timeline_with_section_commands():
    """Test timeline ignores section commands."""
    from rayforge.core.ops import (
        SectionType,
    )

    ops = Ops()
    ops.ops_section_start(
        section_type=SectionType.VECTOR_OUTLINE, workpiece_uid="test-uid"
    )
    ops.set_power(0.5)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    ops.ops_section_end(section_type=SectionType.VECTOR_OUTLINE)

    timeline = OpsTimeline(ops)

    # Should only have 2 moving commands, section commands ignored
    assert timeline.get_step_count() == 2

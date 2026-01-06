from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import RoundedRectCommand
from rayforge.core.sketcher.constraints import TangentConstraint
from rayforge.core.sketcher.entities import Line, Arc


def test_rounded_rect_calculate_geometry():
    """Test the static geometry calculation for a rounded rectangle."""
    result = RoundedRectCommand.calculate_geometry(0, 0, 100, 50, 10.0)
    assert result is not None
    points = result["points"]
    assert len(points) == 12  # 8 tangent + 4 center
    assert len(result["entities"]) == 8  # 4 lines + 4 arcs
    assert len(result["constraints"]) == 17

    assert sum(isinstance(e, Line) for e in result["entities"]) == 4
    assert sum(isinstance(e, Arc) for e in result["entities"]) == 4
    assert (
        sum(isinstance(c, TangentConstraint) for c in result["constraints"])
        == 8
    )


def test_rounded_rect_calculate_geometry_degenerate():
    """Test static calculation returns None for a zero-size rectangle."""
    assert RoundedRectCommand.calculate_geometry(0, 0, 0, 50, 10.0) is None


def test_rounded_rect_command_execute():
    """Test command execution creates the correct number of items."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    cmd = RoundedRectCommand(sketch, start_pid, (100, 50), 10.0)
    cmd.execute()

    # Points: 1 origin + 1 existing start + (8 tangent + 4 center) added = 14
    # The virtual corner points are not added to the sketch.
    assert len(sketch.registry.points) == 14
    assert len(sketch.registry.entities) == 8
    assert len(sketch.constraints) == 17

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import RectangleCommand
from rayforge.core.sketcher.constraints import (
    HorizontalConstraint,
    VerticalConstraint,
)
from rayforge.core.sketcher.entities import Point


def test_rectangle_calculate_geometry_no_snap():
    """Test static calculation when no points are snapped."""
    result = RectangleCommand.calculate_geometry(0, 0, 100, 50, 0, None)
    assert result is not None
    points = result["points"]
    assert len(points) == 4
    assert len(result["entities"]) == 4
    assert len(result["constraints"]) == 4

    # p1 is the start point, represented by its ID
    assert points["p1_id"] == 0

    # p2, p3, p4 are new Point objects
    assert points["p2"].x == 100 and points["p2"].y == 0
    assert points["p3"].x == 100 and points["p3"].y == 50
    assert points["p4"].x == 0 and points["p4"].y == 50


def test_rectangle_calculate_geometry_with_snap():
    """Test static calculation when the end point is snapped."""
    result = RectangleCommand.calculate_geometry(0, 0, 100, 50, 0, 7)
    assert result is not None
    points = result["points"]
    # Check that the snapped point ID is preserved
    assert points["p3"].id == 7


def test_rectangle_calculate_geometry_degenerate():
    """Test static calculation returns None for a zero-size rectangle."""
    assert RectangleCommand.calculate_geometry(0, 0, 0, 50, 0, None) is None
    assert RectangleCommand.calculate_geometry(0, 0, 100, 0, 0, None) is None


def test_rectangle_command_execute_no_snap():
    """Test command execution with no point snapping."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    cmd = RectangleCommand(sketch, start_pid, (100, 50))
    cmd.execute()

    # 1 origin + 1 start_point + 2 new corners + 1 new end corner
    #   = 5 points total
    assert len(sketch.registry.points) == 5
    assert len(sketch.registry.entities) == 4
    assert len(sketch.constraints) == 4
    assert (
        sum(isinstance(c, HorizontalConstraint) for c in sketch.constraints)
        == 2
    )
    assert (
        sum(isinstance(c, VerticalConstraint) for c in sketch.constraints) == 2
    )


def test_rectangle_command_execute_with_snap():
    """Test command execution when snapping to an existing end point."""
    sketch = Sketch()
    start_pid = sketch.add_point(0, 0)
    end_pid = sketch.add_point(100, 50)
    cmd = RectangleCommand(sketch, start_pid, (100, 50), end_pid=end_pid)
    cmd.execute()

    # 1 origin + 1 start + 1 end + 2 new corners = 5 points total
    assert len(sketch.registry.points) == 5


def test_rectangle_command_execute_temp_start():
    """Test command execution when the start point was temporary."""
    sketch = Sketch()
    # Manually add a "temp" point to the registry
    start_pid = 100
    sketch.registry.points.append(Point(start_pid, 0, 0))
    assert len(sketch.registry.points) == 2  # Origin + temp start

    cmd = RectangleCommand(sketch, start_pid, (100, 50), is_start_temp=True)
    cmd.execute()

    # 1 origin + 3 new points (p2, p3, p4) + 1 re-added temp start point = 5
    assert len(sketch.registry.points) == 5
    # Verify the start point ID was reassigned by AddItemsCommand
    assert cmd.add_cmd is not None
    # Find the re-added start point in the command's point list
    re_added_start_point = next(
        p for p in cmd.add_cmd.points if p.x == 0 and p.y == 0
    )
    assert re_added_start_point.id != 100

    cmd.undo()
    # The temp point should be restored (id=100), plus the origin (id=0)
    assert len(sketch.registry.points) == 2
    assert sketch.registry.get_point(start_pid) is not None

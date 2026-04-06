import math
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand
from post_processors.transformers import MergeLinesTransformer


def test_no_duplicate_lines():
    """Test that non-overlapping lines are preserved."""
    ops = Ops()
    ops.set_power(1.0)

    ops.move_to(0, 0)
    ops.line_to(10, 0)
    ops.move_to(20, 0)
    ops.line_to(30, 0)

    original_move_count = sum(
        1 for c in ops.commands if isinstance(c, MoveToCommand)
    )
    original_line_count = sum(
        1 for c in ops.commands if isinstance(c, LineToCommand)
    )

    transformer = MergeLinesTransformer(enabled=True, tolerance=0.1)
    transformer.run(ops)

    move_count = sum(1 for c in ops.commands if isinstance(c, MoveToCommand))
    line_count = sum(1 for c in ops.commands if isinstance(c, LineToCommand))

    assert move_count == original_move_count
    assert line_count == original_line_count


def test_identical_duplicate_lines_removed():
    """Test that identical overlapping lines have one removed."""
    ops = Ops()
    ops.set_power(1.0)

    ops.move_to(0, 0)
    ops.line_to(10, 0)

    ops.move_to(0, 0)
    ops.line_to(10, 0)

    transformer = MergeLinesTransformer(enabled=True, tolerance=0.1)
    transformer.run(ops)

    line_count = sum(1 for c in ops.commands if isinstance(c, LineToCommand))

    assert line_count == 1


def test_opposite_direction_duplicate_lines_removed():
    """Test that overlapping lines in opposite directions are merged."""
    ops = Ops()
    ops.set_power(1.0)

    ops.move_to(0, 0)
    ops.line_to(10, 0)

    ops.move_to(10, 0)
    ops.line_to(0, 0)

    transformer = MergeLinesTransformer(enabled=True, tolerance=0.1)
    transformer.run(ops)

    line_count = sum(1 for c in ops.commands if isinstance(c, LineToCommand))

    assert line_count == 1


def test_tolerance_affects_merging():
    """Test tolerance parameter affects which lines are merged."""
    ops = Ops()
    ops.set_power(1.0)

    ops.move_to(0, 0)
    ops.line_to(10, 0)

    ops.move_to(0, 0.05)
    ops.line_to(10, 0.05)

    transformer_tight = MergeLinesTransformer(enabled=True, tolerance=0.01)
    ops_copy_tight = ops.copy()
    transformer_tight.run(ops_copy_tight)
    line_count_tight = sum(
        1 for c in ops_copy_tight.commands if isinstance(c, LineToCommand)
    )
    assert line_count_tight == 2

    transformer_loose = MergeLinesTransformer(enabled=True, tolerance=0.2)
    ops_copy_loose = ops.copy()
    transformer_loose.run(ops_copy_loose)
    line_count_loose = sum(
        1 for c in ops_copy_loose.commands if isinstance(c, LineToCommand)
    )
    assert line_count_loose == 1


def test_adjacent_rectangles_shared_edge():
    """Test merging shared edge between two adjacent rectangles."""
    ops = Ops()
    ops.set_power(1.0)

    ops.move_to(0, 0)
    ops.line_to(10, 0)
    ops.line_to(10, 10)
    ops.line_to(0, 10)
    ops.line_to(0, 0)

    ops.move_to(10, 0)
    ops.line_to(20, 0)
    ops.line_to(20, 10)
    ops.line_to(10, 10)
    ops.line_to(10, 0)

    original_line_count = sum(
        1 for c in ops.commands if isinstance(c, LineToCommand)
    )

    transformer = MergeLinesTransformer(enabled=True, tolerance=0.1)
    transformer.run(ops)

    line_count = sum(1 for c in ops.commands if isinstance(c, LineToCommand))

    assert line_count < original_line_count


def test_disabled_transformer():
    """Test that disabled transformer doesn't modify ops."""
    ops = Ops()
    ops.set_power(1.0)

    ops.move_to(0, 0)
    ops.line_to(10, 0)
    ops.move_to(0, 0)
    ops.line_to(10, 0)

    original_line_count = sum(
        1 for c in ops.commands if isinstance(c, LineToCommand)
    )

    transformer = MergeLinesTransformer(enabled=False)
    transformer.run(ops)

    line_count = sum(1 for c in ops.commands if isinstance(c, LineToCommand))

    assert line_count == original_line_count


def test_empty_ops():
    """Test that empty ops is handled gracefully."""
    ops = Ops()

    transformer = MergeLinesTransformer(enabled=True)
    transformer.run(ops)

    assert ops.is_empty()


def test_serialization():
    """Test to_dict and from_dict methods."""
    transformer1 = MergeLinesTransformer(enabled=True, tolerance=0.5)
    data = transformer1.to_dict()

    transformer2 = MergeLinesTransformer.from_dict(data)

    assert transformer2.enabled == transformer1.enabled
    assert transformer2.tolerance == transformer1.tolerance


def test_serialization_default_values():
    """Test deserialization with missing values uses defaults."""
    data = {"name": "MergeLinesTransformer"}
    transformer = MergeLinesTransformer.from_dict(data)

    assert transformer.enabled is True
    assert transformer.tolerance == MergeLinesTransformer.DEFAULT_TOLERANCE


def test_overlapping_collinear_segments():
    """
    Test that partially overlapping collinear segments are sliced correctly.
    """
    ops = Ops()
    ops.set_power(1.0)

    ops.move_to(0, 0)
    ops.line_to(10, 0)

    ops.move_to(5, 0)
    ops.line_to(15, 0)

    transformer = MergeLinesTransformer(enabled=True, tolerance=0.1)
    transformer.run(ops)

    line_count = sum(1 for c in ops.commands if isinstance(c, LineToCommand))

    # Under the 1D boolean union logic with horizontal tolerance expansion,
    # the second segment is trimmed so we have exactly two LineToCommands.
    # The total cut length is perfectly merged, minus the tolerance padding.
    assert line_count == 2
    assert math.isclose(ops.cut_distance(), 15.0 - transformer.tolerance)


def test_perpendicular_lines_not_merged():
    """Test that perpendicular lines are not merged."""
    ops = Ops()
    ops.set_power(1.0)

    ops.move_to(0, 0)
    ops.line_to(10, 0)

    ops.move_to(5, -5)
    ops.line_to(5, 5)

    original_line_count = sum(
        1 for c in ops.commands if isinstance(c, LineToCommand)
    )

    transformer = MergeLinesTransformer(enabled=True, tolerance=0.1)
    transformer.run(ops)

    line_count = sum(1 for c in ops.commands if isinstance(c, LineToCommand))

    assert line_count == original_line_count


def test_triangle_shared_edge():
    """Test merging shared edges between two triangles."""
    ops = Ops()
    ops.set_power(1.0)

    ops.move_to(0, 0)
    ops.line_to(10, 0)
    ops.line_to(5, 10)
    ops.line_to(0, 0)

    ops.move_to(10, 0)
    ops.line_to(0, 0)
    ops.line_to(5, -10)
    ops.line_to(10, 0)

    original_line_count = sum(
        1 for c in ops.commands if isinstance(c, LineToCommand)
    )

    transformer = MergeLinesTransformer(enabled=True, tolerance=0.1)
    transformer.run(ops)

    line_count = sum(1 for c in ops.commands if isinstance(c, LineToCommand))

    assert line_count < original_line_count

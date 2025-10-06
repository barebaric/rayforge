"""
Unit tests for MaterialTestGridProducer.

Tests the material test grid generation including:
- Initialization and defaults
- Serialization/deserialization
- Ops generation
- Risk-sorted execution order
- Grid dimensions and sizing
"""

import pytest
from rayforge.core.ops import (
    SetPowerCommand,
    SetCutSpeedCommand,
    MoveToCommand,
    LineToCommand,
)
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.producer.base import OpsProducer
from rayforge.pipeline.producer.material_test_grid import (
    MaterialTestGridProducer,
    MaterialTestGridType,
)
from rayforge.machine.models.machine import Laser


@pytest.fixture
def producer() -> MaterialTestGridProducer:
    """Returns a default-initialized MaterialTestGridProducer."""
    return MaterialTestGridProducer(include_labels=False)


@pytest.fixture
def custom_producer() -> MaterialTestGridProducer:
    """Returns a producer with custom parameters."""
    return MaterialTestGridProducer(
        test_type=MaterialTestGridType.ENGRAVE,
        speed_range=(1000.0, 5000.0),
        power_range=(20.0, 80.0),
        grid_dimensions=(3, 4),
        shape_size=15.0,
        spacing=3.0,
        include_labels=True,
    )


@pytest.fixture
def laser() -> Laser:
    """Returns a default laser model."""
    return Laser()


@pytest.fixture
def mock_workpiece() -> WorkPiece:
    """Returns a mock workpiece with a default size."""
    wp = WorkPiece(name="test_wp")
    wp.uid = "wp_test_123"
    wp.set_size(50.0, 50.0)
    return wp


def test_initialization_defaults(producer: MaterialTestGridProducer):
    """Verify the producer initializes with expected default values."""
    assert producer.test_type == MaterialTestGridType.CUT
    assert producer.speed_range == (100.0, 500.0)
    assert producer.power_range == (10.0, 100.0)
    assert producer.grid_dimensions == (5, 5)
    assert producer.shape_size == 10.0
    assert producer.spacing == 2.0


def test_initialization_custom(custom_producer: MaterialTestGridProducer):
    """Verify custom initialization values are stored correctly."""
    assert custom_producer.test_type == MaterialTestGridType.ENGRAVE
    assert custom_producer.speed_range == (1000.0, 5000.0)
    assert custom_producer.power_range == (20.0, 80.0)
    assert custom_producer.grid_dimensions == (3, 4)
    assert custom_producer.shape_size == 15.0
    assert custom_producer.spacing == 3.0


def test_requires_full_render_is_false(producer: MaterialTestGridProducer):
    """Material test doesn't need rendering - it generates ops directly."""
    assert producer.requires_full_render is False


def test_serialization_and_deserialization():
    """
    Tests that the producer can be serialized to a dict and recreated
    with all parameters intact.
    """
    # Arrange
    original = MaterialTestGridProducer(
        test_type=MaterialTestGridType.ENGRAVE,
        speed_range=(1000.0, 5000.0),
        power_range=(20.0, 80.0),
        grid_dimensions=(4, 6),
        shape_size=12.5,
        spacing=2.5,
        include_labels=False,
    )

    # Act
    data = original.to_dict()
    recreated = OpsProducer.from_dict(data)

    # Assert type
    assert data["type"] == "MaterialTestGridProducer"
    assert isinstance(recreated, MaterialTestGridProducer)

    # Assert parameters
    params = data["params"]
    assert params["test_type"] == "Engrave"
    assert params["speed_range"] == [1000.0, 5000.0]
    assert params["power_range"] == [20.0, 80.0]
    assert params["grid_dimensions"] == [4, 6]
    assert params["shape_size"] == 12.5
    assert params["spacing"] == 2.5

    # Assert recreated values (note: lists become tuples on deserialization)
    assert recreated.test_type == MaterialTestGridType.ENGRAVE
    assert tuple(recreated.speed_range) == (1000.0, 5000.0)
    assert tuple(recreated.power_range) == (20.0, 80.0)
    assert tuple(recreated.grid_dimensions) == (4, 6)
    assert recreated.shape_size == 12.5
    assert recreated.spacing == 2.5


def test_basic_ops_generation(
    producer: MaterialTestGridProducer, laser: Laser, mock_workpiece: WorkPiece
):
    """Test that ops generation produces a valid artifact."""
    # Act
    artifact = producer.run(
        laser=laser,
        surface=None,
        pixels_per_mm=None,
        workpiece=mock_workpiece,
    )

    # Assert
    assert artifact is not None
    assert artifact.ops is not None
    assert artifact.is_scalable is True
    assert (
        artifact.source_coordinate_system == CoordinateSystem.MILLIMETER_SPACE
    )


def test_grid_dimensions_calculation(laser: Laser):
    """Test that grid dimensions are calculated correctly."""
    producer = MaterialTestGridProducer(
        grid_dimensions=(3, 3),
        shape_size=10.0,
        spacing=2.0,
        include_labels=False,
    )

    artifact = producer.run(
        laser=laser,
        surface=None,
        pixels_per_mm=None,
        workpiece=None,
    )

    # Grid: 3 cols × 3 rows
    # Size per cell: 10mm + 2mm spacing
    # Total width: 3 * 12 - 2 = 34mm (subtract last spacing)
    expected_width = 3 * (10.0 + 2.0) - 2.0
    expected_height = 3 * (10.0 + 2.0) - 2.0

    assert artifact.source_dimensions == (expected_width, expected_height)
    assert artifact.generation_size == (expected_width, expected_height)


def test_risk_sorted_execution_order(laser: Laser):
    """
    Verify that test elements are executed in risk-sorted order:
    highest speed first, then lowest power.
    """
    producer = MaterialTestGridProducer(
        speed_range=(100.0, 300.0),
        power_range=(10.0, 30.0),
        grid_dimensions=(3, 3),
        include_labels=False,
    )

    artifact = producer.run(
        laser=laser, surface=None, pixels_per_mm=None, workpiece=None
    )

    # Extract speed and power commands
    speeds = []
    powers = []

    for cmd in artifact.ops.commands:
        if isinstance(cmd, SetCutSpeedCommand):
            speeds.append(cmd.speed)
        elif isinstance(cmd, SetPowerCommand):
            powers.append(cmd.power)

    # Should have 9 test cells (3x3 grid)
    assert len(speeds) == 9
    assert len(powers) == 9

    # First operation should be highest speed (safest)
    assert speeds[0] == 300.0, "First speed should be maximum (safest)"

    # Verify risk-sorted: speed decreases or stays same (never increases)
    # until it resets for a new speed column
    # With 3x3 grid:
    # - Speed varies across columns: 100, 200, 300
    # - Power varies across rows: 10, 20, 30
    # Risk-sorted: (-speed, power) gives us high speeds first
    # So we should see: 300, 300, 300, 200, 200, 200, 100, 100, 100

    # Check that we have three distinct speed levels
    unique_speeds = sorted(set(speeds), reverse=True)
    assert unique_speeds == [300.0, 200.0, 100.0]

    # First three should all be max speed
    assert all(s == 300.0 for s in speeds[0:3])


def test_ops_contains_rectangles(laser: Laser):
    """Test that generated ops contain rectangle drawing commands."""
    producer = MaterialTestGridProducer(
        grid_dimensions=(2, 2),
        shape_size=10.0,
        spacing=2.0,
        include_labels=False,
    )

    artifact = producer.run(
        laser=laser, surface=None, pixels_per_mm=None, workpiece=None
    )

    # Count move and line commands (rectangles should have these)
    move_count = sum(
        1 for cmd in artifact.ops.commands if isinstance(cmd, MoveToCommand)
    )
    line_count = sum(
        1 for cmd in artifact.ops.commands if isinstance(cmd, LineToCommand)
    )

    # 2x2 grid = 4 rectangles
    # Each rectangle: 1 move + 4 lines (including close)
    assert move_count >= 4, "Should have at least one move per rectangle"
    assert line_count >= 16, "Should have at least 4 lines per rectangle"


def test_power_and_speed_ranges(laser: Laser):
    """Test that power and speed values are within specified ranges."""
    min_speed, max_speed = 200.0, 800.0
    min_power_percent, max_power_percent = 20.0, 80.0

    producer = MaterialTestGridProducer(
        speed_range=(min_speed, max_speed),
        power_range=(min_power_percent, max_power_percent),
        grid_dimensions=(5, 5),
        include_labels=False,
    )

    artifact = producer.run(
        laser=laser, surface=None, pixels_per_mm=None, workpiece=None
    )

    speeds = [
        cmd.speed
        for cmd in artifact.ops.commands
        if isinstance(cmd, SetCutSpeedCommand)
    ]
    powers = [
        cmd.power
        for cmd in artifact.ops.commands
        if isinstance(cmd, SetPowerCommand)
    ]

    # Power values are now normalized to 0.0-1.0 range
    min_power = min_power_percent / 100.0
    max_power = max_power_percent / 100.0

    # All speeds should be within range
    assert all(min_speed <= s <= max_speed for s in speeds)
    # All powers should be within range (with tolerance for floating point)
    assert all(min_power - 0.001 <= p <= max_power + 0.001 for p in powers)

    # Should include both min and max values
    assert min_speed in speeds
    assert max_speed in speeds
    assert any(abs(p - min_power) < 0.001 for p in powers)
    assert any(abs(p - max_power) < 0.001 for p in powers)


def test_single_column_grid(laser: Laser):
    """Test edge case: single column (only speed varies on Y-axis)."""
    producer = MaterialTestGridProducer(
        speed_range=(100.0, 500.0),
        power_range=(50.0, 50.0),  # Same min and max
        grid_dimensions=(1, 5),
        include_labels=False,
    )

    artifact = producer.run(
        laser=laser, surface=None, pixels_per_mm=None, workpiece=None
    )

    speeds = [
        cmd.speed
        for cmd in artifact.ops.commands
        if isinstance(cmd, SetCutSpeedCommand)
    ]
    powers = [
        cmd.power
        for cmd in artifact.ops.commands
        if isinstance(cmd, SetPowerCommand)
    ]

    # Speeds should vary (across 5 rows)
    assert len(set(speeds)) == 5
    # All powers should be the same (normalized to 0.0-1.0 range: 50% = 0.5)
    expected_power = 50.0 / 100.0
    assert all(abs(p - expected_power) < 0.001 for p in powers)


def test_single_row_grid(laser: Laser):
    """Test edge case: single row (only power varies on X-axis)."""
    producer = MaterialTestGridProducer(
        speed_range=(100.0, 100.0),  # Same min and max
        power_range=(10.0, 50.0),
        grid_dimensions=(5, 1),
        include_labels=False,
    )

    artifact = producer.run(
        laser=laser, surface=None, pixels_per_mm=None, workpiece=None
    )

    speeds = [
        cmd.speed
        for cmd in artifact.ops.commands
        if isinstance(cmd, SetCutSpeedCommand)
    ]
    powers = [
        cmd.power
        for cmd in artifact.ops.commands
        if isinstance(cmd, SetPowerCommand)
    ]

    # All speeds should be the same
    assert all(s == 100.0 for s in speeds)
    # Powers should vary (across 5 columns)
    assert len(set(powers)) == 5


def test_workpiece_uid_in_section_commands(
    mock_workpiece: WorkPiece, laser: Laser
):
    """
    Test that workpiece UID is included in section commands when provided.
    """
    producer = MaterialTestGridProducer(
        grid_dimensions=(2, 2),
        include_labels=False,
    )

    artifact = producer.run(
        laser=laser,
        surface=None,
        pixels_per_mm=None,
        workpiece=mock_workpiece,
    )

    # Check for section start command with workpiece UID
    from rayforge.core.ops import OpsSectionStartCommand

    section_starts = [
        cmd
        for cmd in artifact.ops.commands
        if isinstance(cmd, OpsSectionStartCommand)
    ]

    assert len(section_starts) > 0
    assert section_starts[0].workpiece_uid == mock_workpiece.uid


def test_minimum_grid_size(laser: Laser):
    """Test the minimum allowed grid size (2x2)."""
    producer = MaterialTestGridProducer(
        grid_dimensions=(2, 2),
        shape_size=5.0,
        spacing=1.0,
        include_labels=False,
    )

    artifact = producer.run(
        laser=laser, surface=None, pixels_per_mm=None, workpiece=None
    )

    # Should have 4 test cells
    speeds = [
        cmd.speed
        for cmd in artifact.ops.commands
        if isinstance(cmd, SetCutSpeedCommand)
    ]
    assert len(speeds) == 4


def test_large_grid(laser: Laser):
    """Test that large grids (e.g., 10x10) work correctly."""
    producer = MaterialTestGridProducer(
        grid_dimensions=(10, 10),
        shape_size=5.0,
        spacing=1.0,
        include_labels=False,
    )

    artifact = producer.run(
        laser=laser, surface=None, pixels_per_mm=None, workpiece=None
    )

    # Should have 100 test cells
    speeds = [
        cmd.speed
        for cmd in artifact.ops.commands
        if isinstance(cmd, SetCutSpeedCommand)
    ]
    assert len(speeds) == 100

    # Verify dimensions
    expected_width = 10 * (5.0 + 1.0) - 1.0
    expected_height = 10 * (5.0 + 1.0) - 1.0
    assert artifact.source_dimensions == (expected_width, expected_height)

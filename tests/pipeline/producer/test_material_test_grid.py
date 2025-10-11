"""
Unit tests for MaterialTestGridProducer.

Tests the material test grid generation including:
- Initialization and defaults
- Serialization/deserialization
- Ops generation
- Risk-sorted execution order
- Grid dimensions and sizing
"""

from unittest.mock import MagicMock, patch

import pytest
from rayforge.core.ops import (
    SetPowerCommand,
    SetCutSpeedCommand,
    MoveToCommand,
    LineToCommand,
    OpsSectionStartCommand,
)
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.producer.base import OpsProducer
from rayforge.pipeline.producer.material_test_grid import (
    MaterialTestGridProducer,
    MaterialTestGridType,
    get_material_test_proportional_size,
    draw_material_test_preview,
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
    wp.set_size(100.0, 100.0)  # Default size for basic tests
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


def test_serialization_and_deserialization():
    """
    Tests that the producer can be serialized to a dict and recreated
    with all parameters intact.
    """
    original = MaterialTestGridProducer(
        test_type=MaterialTestGridType.ENGRAVE,
        speed_range=(1000.0, 5000.0),
        power_range=(20.0, 80.0),
        grid_dimensions=(4, 6),
        shape_size=12.5,
        spacing=2.5,
        include_labels=False,
    )
    data = original.to_dict()
    recreated = OpsProducer.from_dict(data)
    assert data["type"] == "MaterialTestGridProducer"
    assert isinstance(recreated, MaterialTestGridProducer)
    assert recreated.test_type == MaterialTestGridType.ENGRAVE
    assert tuple(recreated.grid_dimensions) == (4, 6)
    assert recreated.shape_size == 12.5


def test_get_material_test_proportional_size():
    """Test the standalone size calculation function."""
    params_no_labels = {
        "grid_dimensions": (3, 3),
        "shape_size": 10.0,
        "spacing": 2.0,
        "include_labels": False,
    }
    width, height = get_material_test_proportional_size(params_no_labels)
    assert width == 3 * 10.0 + 2 * 2.0
    assert height == 3 * 10.0 + 2 * 2.0

    params_with_labels = {
        "grid_dimensions": (5, 5),
        "shape_size": 10.0,
        "spacing": 2.0,
        "include_labels": True,
    }
    width, height = get_material_test_proportional_size(params_with_labels)
    base_width = 5 * 10.0 + 4 * 2.0
    base_height = 5 * 10.0 + 4 * 2.0
    assert width == base_width + 15.0  # margin_left
    assert height == base_height + 15.0  # margin_top


@patch(
    "rayforge.pipeline.producer.material_test_grid."
    "MaterialTestGridProducer.draw_preview"
)
def test_draw_material_test_preview_delegates_call(
    mock_draw_preview: MagicMock,
):
    """Verify the standalone draw function correctly calls the class method."""
    mock_ctx = MagicMock()
    params = {"key": "value"}
    draw_material_test_preview(mock_ctx, 100, 200, params)
    mock_draw_preview.assert_called_once_with(mock_ctx, 100, 200, params)


def test_basic_ops_generation(
    producer: MaterialTestGridProducer, laser: Laser, mock_workpiece: WorkPiece
):
    """Test that ops generation produces a valid artifact."""
    artifact = producer.run(
        laser=laser,
        surface=None,
        pixels_per_mm=None,
        workpiece=mock_workpiece,
    )
    assert artifact is not None
    assert artifact.ops is not None
    assert not artifact.is_scalable
    assert (
        artifact.source_coordinate_system == CoordinateSystem.MILLIMETER_SPACE
    )
    assert artifact.source_dimensions == mock_workpiece.size


def test_risk_sorted_execution_order(laser: Laser, mock_workpiece: WorkPiece):
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
        laser=laser, surface=None, pixels_per_mm=None, workpiece=mock_workpiece
    )
    speeds = [
        c.speed
        for c in artifact.ops.commands
        if isinstance(c, SetCutSpeedCommand)
    ]
    assert len(speeds) == 9
    assert speeds[0] == 300.0, "First speed should be maximum (safest)"
    unique_speeds = sorted(set(speeds), reverse=True)
    assert unique_speeds == [300.0, 200.0, 100.0]
    assert all(s == 300.0 for s in speeds[0:3])


def test_ops_contains_rectangles(laser: Laser, mock_workpiece: WorkPiece):
    """Test that generated ops contain rectangle drawing commands."""
    producer = MaterialTestGridProducer(
        grid_dimensions=(2, 2),
        shape_size=10.0,
        spacing=2.0,
        include_labels=False,
    )
    artifact = producer.run(
        laser=laser, surface=None, pixels_per_mm=None, workpiece=mock_workpiece
    )
    move_count = sum(
        1 for cmd in artifact.ops.commands if isinstance(cmd, MoveToCommand)
    )
    line_count = sum(
        1 for cmd in artifact.ops.commands if isinstance(cmd, LineToCommand)
    )
    assert move_count >= 4
    assert line_count >= 16


def test_power_and_speed_ranges(laser: Laser, mock_workpiece: WorkPiece):
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
        laser=laser, surface=None, pixels_per_mm=None, workpiece=mock_workpiece
    )
    speeds = [
        c.speed
        for c in artifact.ops.commands
        if isinstance(c, SetCutSpeedCommand)
    ]
    powers = [
        c.power
        for c in artifact.ops.commands
        if isinstance(c, SetPowerCommand)
    ]
    min_power, max_power = min_power_percent / 100.0, max_power_percent / 100.0
    assert all(min_speed <= s <= max_speed for s in speeds)
    assert all(min_power - 1e-3 <= p <= max_power + 1e-3 for p in powers)
    assert min_speed in speeds
    assert max_speed in speeds
    assert any(abs(p - min_power) < 1e-3 for p in powers)
    assert any(abs(p - max_power) < 1e-3 for p in powers)


def test_single_column_grid(laser: Laser, mock_workpiece: WorkPiece):
    """Test edge case: single column (only speed varies on Y-axis)."""
    producer = MaterialTestGridProducer(
        speed_range=(100.0, 500.0),
        power_range=(50.0, 50.0),
        grid_dimensions=(1, 5),
        include_labels=False,
    )
    artifact = producer.run(
        laser=laser, surface=None, pixels_per_mm=None, workpiece=mock_workpiece
    )
    speeds = [
        c.speed
        for c in artifact.ops.commands
        if isinstance(c, SetCutSpeedCommand)
    ]
    powers = [
        c.power
        for c in artifact.ops.commands
        if isinstance(c, SetPowerCommand)
    ]
    assert len(set(speeds)) == 5
    assert all(abs(p - 0.5) < 1e-3 for p in powers)


def test_single_row_grid(laser: Laser, mock_workpiece: WorkPiece):
    """Test edge case: single row (only power varies on X-axis)."""
    producer = MaterialTestGridProducer(
        speed_range=(100.0, 100.0),
        power_range=(10.0, 50.0),
        grid_dimensions=(5, 1),
        include_labels=False,
    )
    artifact = producer.run(
        laser=laser, surface=None, pixels_per_mm=None, workpiece=mock_workpiece
    )
    speeds = [
        c.speed
        for c in artifact.ops.commands
        if isinstance(c, SetCutSpeedCommand)
    ]
    powers = [
        c.power
        for c in artifact.ops.commands
        if isinstance(c, SetPowerCommand)
    ]
    assert all(s == 100.0 for s in speeds)
    assert len(set(powers)) == 5


def test_workpiece_uid_in_section_commands(
    mock_workpiece: WorkPiece, laser: Laser
):
    """Test that workpiece UID is included in section commands."""
    producer = MaterialTestGridProducer()
    artifact = producer.run(
        laser=laser,
        surface=None,
        pixels_per_mm=None,
        workpiece=mock_workpiece,
    )
    section_starts = [
        c
        for c in artifact.ops.commands
        if isinstance(c, OpsSectionStartCommand)
    ]
    assert len(section_starts) > 0
    assert section_starts[0].workpiece_uid == mock_workpiece.uid


def test_grid_cell_count(laser: Laser):
    """Test grids of various sizes produce the correct number of cells."""
    for dims in [(2, 2), (10, 10), (3, 7)]:
        cols, rows = dims
        params = {
            "grid_dimensions": dims,
            "shape_size": 5.0,
            "spacing": 1.0,
            "include_labels": False,
        }
        producer = MaterialTestGridProducer(**params)
        wp_size = get_material_test_proportional_size(params)
        workpiece = WorkPiece(name="sized_wp")
        workpiece.set_size(wp_size[0], wp_size[1])

        artifact = producer.run(
            laser=laser, surface=None, pixels_per_mm=None, workpiece=workpiece
        )
        speeds = [
            c.speed
            for c in artifact.ops.commands
            if isinstance(c, SetCutSpeedCommand)
        ]
        assert len(speeds) == cols * rows
        assert artifact.source_dimensions == wp_size

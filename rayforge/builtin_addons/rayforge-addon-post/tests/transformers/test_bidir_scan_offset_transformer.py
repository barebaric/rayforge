import pytest
from post_processors.transformers import BidirScanOffsetTransformer
from raygeo.ops import Ops
from raygeo.ops.types import CommandType

from rayforge.pipeline.transformer.base import ExecutionPhase


@pytest.fixture
def transformer() -> BidirScanOffsetTransformer:
    """Provides a default, enabled BidirScanOffsetTransformer instance."""
    return BidirScanOffsetTransformer(enabled=True)


def _build_zigzag() -> Ops:
    """Three-row zigzag raster: row0 LTR, row1 RTL, row2 LTR."""
    ops = Ops()
    ops.move_to(0.0, 0.6, 0.0)
    ops.scan_to(2.0, 0.6, 0.0, power_values=[200, 200, 200, 200])
    ops.move_to(2.0, 0.5, 0.0)
    ops.scan_to(0.0, 0.5, 0.0, power_values=[210, 210, 210, 210])
    ops.move_to(0.0, 0.4, 0.0)
    ops.scan_to(2.0, 0.4, 0.0, power_values=[220, 220, 220, 220])
    return ops


def test_execution_phase_is_correct(transformer: BidirScanOffsetTransformer):
    assert transformer.execution_phase == ExecutionPhase.POST_PROCESSING


def test_serialization_and_deserialization():
    original = BidirScanOffsetTransformer(enabled=False)
    data = original.to_dict()
    recreated = BidirScanOffsetTransformer.from_dict(data)
    assert data["name"] == "BidirScanOffsetTransformer"
    assert data["enabled"] is False
    assert isinstance(recreated, BidirScanOffsetTransformer)
    assert recreated.enabled is False


def test_no_op_when_disabled():
    ops = _build_zigzag()
    original = [ops.endpoint(i) for i in range(ops.len())]

    transformer = BidirScanOffsetTransformer(enabled=False)
    transformer.run(ops, settings={"bidir_x_offset_mm": 0.3})

    assert [ops.endpoint(i) for i in range(ops.len())] == original


def test_no_op_with_zero_offset(transformer: BidirScanOffsetTransformer):
    ops = _build_zigzag()
    original = [ops.endpoint(i) for i in range(ops.len())]

    transformer.run(ops, settings={"bidir_x_offset_mm": 0.0})

    assert [ops.endpoint(i) for i in range(ops.len())] == original


def test_no_op_without_settings(transformer: BidirScanOffsetTransformer):
    ops = _build_zigzag()
    original = [ops.endpoint(i) for i in range(ops.len())]

    transformer.run(ops, settings=None)

    assert [ops.endpoint(i) for i in range(ops.len())] == original


def test_shifts_only_right_to_left_passes(
    transformer: BidirScanOffsetTransformer,
):
    ops = _build_zigzag()

    transformer.run(ops, settings={"bidir_x_offset_mm": 0.3})

    assert ops.len() == 6
    # Row 0 (LTR): untouched.
    assert ops.command_type(0) == CommandType.MOVE_TO
    assert ops.endpoint(0) == pytest.approx((0.0, 0.6, 0.0))
    assert ops.command_type(1) == CommandType.SCAN_LINE
    assert ops.endpoint(1) == pytest.approx((2.0, 0.6, 0.0))
    assert list(ops.scanline_data(1)) == [200, 200, 200, 200]
    # Row 1 (RTL): entry MoveTo and ScanLine endpoint both shifted by +0.3.
    assert ops.command_type(2) == CommandType.MOVE_TO
    assert ops.endpoint(2) == pytest.approx((2.3, 0.5, 0.0))
    assert ops.command_type(3) == CommandType.SCAN_LINE
    assert ops.endpoint(3) == pytest.approx((0.3, 0.5, 0.0))
    assert list(ops.scanline_data(3)) == [210, 210, 210, 210]
    # Row 2 (LTR): untouched, including the absolute MoveTo into it.
    assert ops.command_type(4) == CommandType.MOVE_TO
    assert ops.endpoint(4) == pytest.approx((0.0, 0.4, 0.0))
    assert ops.command_type(5) == CommandType.SCAN_LINE
    assert ops.endpoint(5) == pytest.approx((2.0, 0.4, 0.0))
    assert list(ops.scanline_data(5)) == [220, 220, 220, 220]


def test_negative_offset_shifts_left(transformer: BidirScanOffsetTransformer):
    ops = _build_zigzag()

    transformer.run(ops, settings={"bidir_x_offset_mm": -0.5})

    assert ops.endpoint(2) == pytest.approx((1.5, 0.5, 0.0))
    assert ops.endpoint(3) == pytest.approx((-0.5, 0.5, 0.0))


def test_preserves_intermediate_state_commands(
    transformer: BidirScanOffsetTransformer,
):
    """A SetPower between the entry MoveTo and the ScanLine must survive."""
    ops = Ops()
    ops.move_to(2.0, 0.5, 0.0)
    ops.set_power(0.5)
    ops.scan_to(0.0, 0.5, 0.0, power_values=[210, 210, 210, 210])

    transformer.run(ops, settings={"bidir_x_offset_mm": 0.3})

    assert ops.len() == 3
    assert ops.command_type(0) == CommandType.MOVE_TO
    assert ops.endpoint(0) == pytest.approx((2.3, 0.5, 0.0))
    assert ops.command_type(1) == CommandType.SET_POWER
    assert ops.power(1) == pytest.approx(0.5)
    assert ops.command_type(2) == CommandType.SCAN_LINE
    assert ops.endpoint(2) == pytest.approx((0.3, 0.5, 0.0))

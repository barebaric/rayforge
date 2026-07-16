"""Unit tests for the pure-geometry array strategies."""

import math

import pytest

from rayforge.doceditor.array import (
    ArrayMode,
    ArrayParams,
    CircularArrayParams,
    CircularArrayStrategy,
    GridArrayParams,
    GridArrayStrategy,
    PointRotationParams,
    PointRotationStrategy,
    SpacingMode,
    make_array_strategy,
)

# A 20 (w) x 10 (h) unit bounding box placed with its lower-left at
# the origin. Center = (10, 5).
UNIT_BBOX = (0.0, 0.0, 20.0, 10.0)


def _trans(delta):
    """Returns the (tx, ty) component of a delta matrix."""
    return delta.get_translation()


# --------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------
class TestGridStrategy:
    def test_displacement_layout_and_origin_identity(self):
        params = GridArrayParams(
            rows=2,
            cols=3,
            spacing_mode=SpacingMode.DISPLACEMENT,
            col_spacing_mm=15.0,
            row_spacing_mm=12.0,
        )
        placements = GridArrayStrategy(
            UNIT_BBOX, params
        ).calculate_placements()

        # 2 rows x 3 cols = 6 instances.
        assert len(placements) == 6
        # First instance is the identity (original stays in place).
        assert placements[0].is_identity()

        # Cell (row=0, col=1) -> translation (15, 0).
        assert _trans(placements[1]) == pytest.approx((15.0, 0.0))
        # Cell (row=0, col=2) -> translation (30, 0).
        assert _trans(placements[2]) == pytest.approx((30.0, 0.0))
        # Cell (row=1, col=0) -> translation (0, -12): rows grow toward
        # the screen bottom, i.e. negative world Y.
        assert _trans(placements[3]) == pytest.approx((0.0, -12.0))
        # Cell (row=1, col=2) -> translation (30, -12).
        assert _trans(placements[5]) == pytest.approx((30.0, -12.0))

    def test_gap_mode_adds_unit_size(self):
        params = GridArrayParams(
            rows=1,
            cols=2,
            spacing_mode=SpacingMode.GAP,
            col_spacing_mm=5.0,
            row_spacing_mm=5.0,
        )
        placements = GridArrayStrategy(
            UNIT_BBOX, params
        ).calculate_placements()

        assert len(placements) == 2
        # pitch_x = unit_w(20) + gap(5) = 25; pitch_y unused (1 row).
        assert _trans(placements[1]) == pytest.approx((25.0, 0.0))

    def test_single_cell_is_no_op(self):
        params = GridArrayParams(rows=1, cols=1)
        placements = GridArrayStrategy(
            UNIT_BBOX, params
        ).calculate_placements()
        assert len(placements) == 1
        assert placements[0].is_identity()


# --------------------------------------------------------------------
# Point rotation
# --------------------------------------------------------------------
class TestPointRotationStrategy:
    def test_rotates_in_place_around_unit_center(self):
        # Unit center is (10, 5). Copies are rotated around that point,
        # so it must remain a fixed point of every delta.
        params = PointRotationParams(count=4, total_angle_deg=360.0)
        placements = PointRotationStrategy(
            UNIT_BBOX, params
        ).calculate_placements()

        assert len(placements) == 4
        assert placements[0].is_identity()

        cx, cy = 10.0, 5.0  # unit center
        # Full circle -> step = 90 degrees (angles are normalized to
        # [-180, 180], so 270 deg shows as -90).
        assert placements[1].get_x_axis_angle() == pytest.approx(
            90.0, abs=1e-6
        )
        assert placements[3].get_x_axis_angle() == pytest.approx(
            -90.0, abs=1e-6
        )
        # Every delta keeps the unit center fixed (pure in-place rotation).
        for delta in placements[1:]:
            assert delta.transform_point((cx, cy)) == pytest.approx((cx, cy))

    def test_single_copy_is_identity(self):
        params = PointRotationParams(count=1)
        placements = PointRotationStrategy(
            UNIT_BBOX, params
        ).calculate_placements()
        assert len(placements) == 1
        assert placements[0].is_identity()


# --------------------------------------------------------------------
# Circular
# --------------------------------------------------------------------
class TestCircularStrategy:
    def test_full_circle_even_distribution(self):
        # Center at (0,0); unit center at (10,5) -> base radius = sqrt(125).
        params = CircularArrayParams(
            count=4,
            total_angle_deg=360.0,
            center_mm=(0.0, 0.0),
            radius_mm=math.hypot(10.0, 5.0),
            rotate_copies=False,
        )
        placements = CircularArrayStrategy(
            UNIT_BBOX, params
        ).calculate_placements()

        assert len(placements) == 4
        assert placements[0].is_identity()

        radius = math.hypot(10.0, 5.0)
        base_angle = math.atan2(5.0, 10.0)
        # Full circle -> step = 360 / count = 90 degrees.
        expected_center_1 = (
            radius * math.cos(base_angle + math.radians(90)),
            radius * math.sin(base_angle + math.radians(90)),
        )
        # Delta moves the unit center (10, 5) to expected_center_1.
        tx, ty = _trans(placements[1])
        assert (tx + 10.0, ty + 5.0) == pytest.approx(expected_center_1)

    def test_partial_arc_uses_count_minus_one_step(self):
        params = CircularArrayParams(
            count=3,
            total_angle_deg=180.0,
            center_mm=(0.0, 0.0),
            rotate_copies=False,
        )
        # Force a well-defined orbit: place the unit center at (10, 0).
        bbox = (5.0, -5.0, 15.0, 5.0)  # center (10, 0), size 10x10
        placements = CircularArrayStrategy(bbox, params).calculate_placements()

        assert len(placements) == 3
        assert placements[0].is_identity()
        # Step = 180 / 3 = 60 degrees. Instance 2 (120 deg) lands at
        # (10*cos(120), 10*sin(120)) = (-5, 8.66).
        expected = (
            10.0 * math.cos(math.radians(120)),
            10.0 * math.sin(math.radians(120)),
        )
        tx, ty = _trans(placements[2])
        assert (tx + 10.0, ty + 0.0) == pytest.approx(expected)

    def test_rotate_copies_applies_spin(self):
        params = CircularArrayParams(
            count=2,
            total_angle_deg=360.0,
            center_mm=(0.0, 0.0),
            rotate_copies=True,
        )
        bbox = (5.0, -5.0, 15.0, 5.0)  # center (10, 0)
        placements = CircularArrayStrategy(bbox, params).calculate_placements()

        # With rotate_copies, instance 1 must include a 180-degree
        # rotation component (in addition to its translation).
        angle = placements[1].get_x_axis_angle()
        assert angle == pytest.approx(180.0, abs=1e-6)

    def test_single_copy_is_identity(self):
        params = CircularArrayParams(count=1)
        placements = CircularArrayStrategy(
            UNIT_BBOX, params
        ).calculate_placements()
        assert len(placements) == 1
        assert placements[0].is_identity()


# --------------------------------------------------------------------
# Factory / integration of params
# --------------------------------------------------------------------
class TestMakeArrayStrategy:
    def test_factory_dispatches_grid(self):
        params = ArrayParams(mode=ArrayMode.GRID)
        strategy = make_array_strategy(UNIT_BBOX, params)
        assert isinstance(strategy, GridArrayStrategy)

    def test_factory_dispatches_point_rotation(self):
        params = ArrayParams(
            mode=ArrayMode.POINT_ROTATION,
            point_rotation=PointRotationParams(count=3, total_angle_deg=180.0),
        )
        strategy = make_array_strategy(UNIT_BBOX, params)
        assert isinstance(strategy, PointRotationStrategy)

    def test_factory_dispatches_circular(self):
        params = ArrayParams(
            mode=ArrayMode.CIRCULAR,
            circular=CircularArrayParams(count=3, total_angle_deg=180.0),
        )
        strategy = make_array_strategy(UNIT_BBOX, params)
        assert isinstance(strategy, CircularArrayStrategy)


# --------------------------------------------------------------------
# Serialization (config persistence)
# --------------------------------------------------------------------
class TestParamsSerialization:
    def test_grid_round_trip(self):
        params = ArrayParams(
            mode=ArrayMode.GRID,
            grid=GridArrayParams(
                rows=4,
                cols=5,
                spacing_mode=SpacingMode.GAP,
                col_spacing_mm=7.5,
                row_spacing_mm=3.25,
            ),
        )
        restored = ArrayParams.from_dict(params.to_dict())
        assert restored.mode == ArrayMode.GRID
        g = restored.grid
        assert (g.rows, g.cols) == (4, 5)
        assert g.spacing_mode == SpacingMode.GAP
        assert g.col_spacing_mm == pytest.approx(7.5)
        assert g.row_spacing_mm == pytest.approx(3.25)

    def test_point_rotation_round_trip(self):
        params = ArrayParams(
            mode=ArrayMode.POINT_ROTATION,
            point_rotation=PointRotationParams(count=9, total_angle_deg=120.0),
        )
        restored = ArrayParams.from_dict(params.to_dict())
        assert restored.mode == ArrayMode.POINT_ROTATION
        pr = restored.point_rotation
        assert pr.count == 9
        assert pr.total_angle_deg == pytest.approx(120.0)

    def test_circular_round_trip(self):
        params = ArrayParams(
            mode=ArrayMode.CIRCULAR,
            circular=CircularArrayParams(
                count=12,
                total_angle_deg=180.0,
                center_mm=(3.5, -2.0),
                rotate_copies=False,
            ),
        )
        restored = ArrayParams.from_dict(params.to_dict())
        assert restored.mode == ArrayMode.CIRCULAR
        c = restored.circular
        assert c.count == 12
        assert c.total_angle_deg == pytest.approx(180.0)
        assert c.center_mm == (3.5, -2.0)
        assert c.radius_mm == pytest.approx(10.0)
        assert c.rotate_copies is False

    def test_from_dict_handles_invalid_and_empty(self):
        # Unknown enum values fall back to defaults instead of raising.
        params = ArrayParams.from_dict(
            {"mode": "nonsense", "grid": {"spacing_mode": "nope"}}
        )
        assert params.mode == ArrayMode.GRID
        assert params.grid.spacing_mode == SpacingMode.GAP
        # None / empty data yields default params.
        assert ArrayParams.from_dict(None).mode == ArrayMode.GRID
        assert ArrayParams.from_dict({}).circular.count == 6

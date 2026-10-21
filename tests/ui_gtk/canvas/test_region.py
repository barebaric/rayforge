"""Tests for the move gizmo region geometry and hit-testing."""

import pytest

from rayforge.ui_gtk.canvas.region import (
    MOVE_HANDLES,
    RESIZE_HANDLES,
    ElementRegion,
    check_region_hit,
    flat_axis_padding,
    get_region_rect,
)


class TestMoveGizmoGeometry:
    """Tests for get_region_rect with ElementRegion.MOVE."""

    @pytest.mark.parametrize(
        "width, height, base_size, scale",
        [
            (100, 80, 20, 1.0),
            (200, 150, 20, 2.0),
            (50, 50, 20, 0.5),
        ],
    )
    def test_move_gizmo_position_non_flipped(
        self, width, height, base_size, scale
    ):
        """In a standard Y-down system, the gizmo is below the
        bottom edge (y > height)."""
        x, y, w, h = get_region_rect(
            ElementRegion.MOVE,
            width,
            height,
            base_size,
            scale_compensation=scale,
        )
        effective_hw = min(base_size / scale, width / 3.0) * 1.95
        effective_hh = min(base_size / scale, height / 3.0) * 1.95
        handle_dist = 5.0 / scale
        move_margin = 5.0 / scale

        assert w == pytest.approx(effective_hw)
        assert h == pytest.approx(effective_hh)
        assert x == pytest.approx(width / 2 - effective_hw / 2)
        assert y == pytest.approx(height + handle_dist + move_margin)

    def test_move_gizmo_position_flipped(self):
        """In a flipped Y system, the gizmo is above the top edge
        (y < 0). It is 50% larger than resize handles with a 5px margin
        between the frame and the gizmo."""
        width, height, base_size = 100, 80, 20
        scale = (1.0, -1.0)
        x, y, w, h = get_region_rect(
            ElementRegion.MOVE,
            width,
            height,
            base_size,
            scale_compensation=scale,
        )
        effective_hw = min(base_size / 1.0, width / 3.0) * 1.95
        effective_hh = min(base_size / 1.0, height / 3.0) * 1.95
        handle_dist = 5.0 / 1.0
        move_margin = 5.0 / 1.0

        assert w == pytest.approx(effective_hw)
        assert h == pytest.approx(effective_hh)
        assert x == pytest.approx(width / 2 - effective_hw / 2)
        assert y == pytest.approx(-effective_hh - handle_dist - move_margin)

    def test_move_gizmo_centered_horizontally(self):
        """The gizmo should be horizontally centered relative to the
        bounding box."""
        width, height = 120, 90
        x, _y, w, _h = get_region_rect(
            ElementRegion.MOVE, width, height, 20, 1.0
        )
        center_x = x + w / 2
        assert center_x == pytest.approx(width / 2)

    def test_move_gizmo_in_move_handles_set(self):
        """ElementRegion.MOVE should be in MOVE_HANDLES."""
        assert ElementRegion.MOVE in MOVE_HANDLES
        assert len(MOVE_HANDLES) == 1


class TestMoveGizmoHitTest:
    """Tests for check_region_hit with ElementRegion.MOVE."""

    def test_hit_on_move_gizmo_non_flipped(self):
        """Clicking inside the gizmo area should return MOVE region."""
        width, height, base_size = 100, 80, 20
        gx, gy, gw, gh = get_region_rect(
            ElementRegion.MOVE, width, height, base_size, 1.0
        )
        center_x = gx + gw / 2
        center_y = gy + gh / 2
        result = check_region_hit(
            center_x,
            center_y,
            width,
            height,
            base_size,
            1.0,
            candidates=MOVE_HANDLES,
        )
        assert result == ElementRegion.MOVE

    def test_hit_outside_move_gizmo(self):
        """Clicking outside the gizmo area should return NONE."""
        width, height, base_size = 100, 80, 20
        result = check_region_hit(
            0,
            0,
            width,
            height,
            base_size,
            1.0,
            candidates=MOVE_HANDLES,
        )
        assert result == ElementRegion.NONE

    def test_hit_on_move_gizmo_flipped(self):
        """Clicking inside the gizmo area in a flipped system should
        return MOVE region."""
        width, height, base_size = 100, 80, 20
        scale = (1.0, -1.0)
        gx, gy, gw, gh = get_region_rect(
            ElementRegion.MOVE, width, height, base_size, scale
        )
        center_x = gx + gw / 2
        center_y = gy + gh / 2
        result = check_region_hit(
            center_x,
            center_y,
            width,
            height,
            base_size,
            scale,
            candidates=MOVE_HANDLES,
        )
        assert result == ElementRegion.MOVE

    def test_move_gizmo_alongside_resize_handles(self):
        """The MOVE gizmo should be hit-testable together with resize
        handles."""
        width, height, base_size = 100, 80, 20
        gx, gy, gw, gh = get_region_rect(
            ElementRegion.MOVE, width, height, base_size, 1.0
        )
        center_x = gx + gw / 2
        center_y = gy + gh / 2

        candidates = RESIZE_HANDLES | MOVE_HANDLES
        result = check_region_hit(
            center_x,
            center_y,
            width,
            height,
            base_size,
            1.0,
            candidates=candidates,
        )
        assert result == ElementRegion.MOVE


class TestFlatAxisHandles:
    """Flat elements get a minimum-thickness selection box."""

    def test_padding_is_zero_for_normal_axis(self):
        assert flat_axis_padding(100.0, 1.0) == 0.0

    def test_padding_is_positive_for_flat_axis(self):
        assert flat_axis_padding(0.0, 1.0) == 8.0

    def test_body_rect_is_padded_for_zero_height_box(self):
        bx, by, bw, bh = get_region_rect(
            ElementRegion.BODY, 200.0, 0.0, 20.0, 1.0
        )
        assert (bx, by, bw, bh) == (0.0, -8.0, 200.0, 16.0)

    def test_corner_handle_has_nonzero_size_for_zero_height_box(self):
        x, y, w, h = get_region_rect(
            ElementRegion.TOP_LEFT, 200.0, 0.0, 20.0, 1.0
        )
        assert w > 0
        assert h > 0
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(-8.0)

    def test_flat_body_hit_within_tolerance(self):
        assert (
            check_region_hit(50.0, 0.0, 200.0, 0.0, 20.0, 1.0)
            == ElementRegion.BODY
        )
        assert (
            check_region_hit(50.0, 9.0, 200.0, 0.0, 20.0, 1.0)
            == ElementRegion.NONE
        )

    def test_flat_handle_wins_over_body(self):
        # The bottom-middle handle overlaps the padded body; the handle
        # must stay hittable so the flat element can be resized.
        assert (
            check_region_hit(50.0, 7.0, 200.0, 0.0, 20.0, 1.0)
            == ElementRegion.BOTTOM_MIDDLE
        )

    def test_corner_handle_wins_over_body(self):
        assert (
            check_region_hit(5.0, 5.0, 100.0, 100.0, 20.0, 1.0)
            == ElementRegion.TOP_LEFT
        )

    def test_flat_corner_handle_is_hittable(self):
        result = check_region_hit(
            10.0,
            -6.0,
            200.0,
            0.0,
            20.0,
            1.0,
            candidates=RESIZE_HANDLES,
        )
        assert result == ElementRegion.TOP_LEFT

    def test_workpiece_like_flat_scale_gets_visible_handles(self):
        """A near-zero axis scale must still yield on-screen handles."""
        scale = (253.98, 1.0325e-09)
        _, _, _, rh = get_region_rect(
            ElementRegion.TOP_LEFT, 1.0, 1.0, 20.0, scale
        )
        assert rh * scale[1] > 1.0

        _, _, _, bh = get_region_rect(
            ElementRegion.BODY, 1.0, 1.0, 20.0, scale
        )
        assert bh * scale[1] > 1.0

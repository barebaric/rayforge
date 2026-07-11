"""Tests for the move gizmo region geometry and hit-testing."""

import pytest

from rayforge.ui_gtk.canvas.region import (
    MOVE_HANDLES,
    ElementRegion,
    check_region_hit,
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
        x, y, w, h = get_region_rect(
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
        from rayforge.ui_gtk.canvas.region import RESIZE_HANDLES

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

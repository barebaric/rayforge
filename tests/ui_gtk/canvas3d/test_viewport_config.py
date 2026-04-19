import numpy as np
import pytest

from rayforge.context import RayforgeContext
from rayforge.machine.models.machine import Machine, Origin
from rayforge.ui_gtk.sim3d.canvas3d.viewport import ViewportConfig


def _make_machine() -> Machine:
    ctx = RayforgeContext()
    return Machine(ctx)


class TestViewportConfigDefaults:
    def test_default_origin_bottom_left(self):
        m = _make_machine()
        vp = ViewportConfig.from_machine(m)
        assert isinstance(vp, ViewportConfig)
        assert vp.x_right is False
        assert vp.y_down is False
        assert vp.x_negative is False
        assert vp.y_negative is False

    def test_default_no_extent_frame(self):
        m = _make_machine()
        vp = ViewportConfig.from_machine(m)
        assert vp.extent_frame is None

    def test_default_model_matrix_is_identity(self):
        m = _make_machine()
        vp = ViewportConfig.from_machine(m)
        expected = np.identity(4, dtype=np.float32)
        np.testing.assert_array_equal(vp.model_matrix, expected)

    def test_default_wcs_offset_is_zero(self):
        m = _make_machine()
        vp = ViewportConfig.from_machine(m)
        assert vp.wcs_offset_mm == (0.0, 0.0, 0.0)

    def test_default_margin_shift_is_identity(self):
        m = _make_machine()
        vp = ViewportConfig.from_machine(m)
        expected = np.identity(4, dtype=np.float32)
        np.testing.assert_array_equal(vp.margin_shift, expected)


class TestViewportConfigAxisOrientation:
    def test_y_axis_down_flips_y(self):
        m = _make_machine()
        m.origin = Origin.TOP_LEFT
        m.set_axis_extents(200.0, 100.0)
        vp = ViewportConfig.from_machine(m)
        assert vp.y_down is True
        assert vp.x_right is False
        assert vp.depth_mm == 100.0
        assert vp.width_mm == 200.0
        expected_translate = np.identity(4, dtype=np.float32)
        expected_translate[1, 3] = 100.0
        expected_scale = np.identity(4, dtype=np.float32)
        expected_scale[1, 1] = -1.0
        expected = expected_translate @ expected_scale
        np.testing.assert_array_equal(vp.model_matrix, expected)

    def test_x_axis_right_flips_x(self):
        m = _make_machine()
        m.origin = Origin.BOTTOM_RIGHT
        m.set_axis_extents(300.0, 200.0)
        vp = ViewportConfig.from_machine(m)
        assert vp.x_right is True
        assert vp.y_down is False
        expected_translate = np.identity(4, dtype=np.float32)
        expected_translate[0, 3] = 300.0
        expected_scale = np.identity(4, dtype=np.float32)
        expected_scale[0, 0] = -1.0
        expected = expected_translate @ expected_scale
        np.testing.assert_array_equal(vp.model_matrix, expected)

    def test_top_right_flips_both(self):
        m = _make_machine()
        m.origin = Origin.TOP_RIGHT
        m.set_axis_extents(400.0, 300.0)
        vp = ViewportConfig.from_machine(m)
        assert vp.x_right is True
        assert vp.y_down is True
        assert vp.width_mm == 400.0
        assert vp.depth_mm == 300.0
        translate = np.identity(4, dtype=np.float32)
        translate[0, 3] = 400.0
        translate[1, 3] = 300.0
        scale = np.identity(4, dtype=np.float32)
        scale[0, 0] = -1.0
        scale[1, 1] = -1.0
        expected = translate @ scale
        np.testing.assert_array_equal(vp.model_matrix, expected)


class TestViewportConfigMargins:
    def test_margin_shift_is_pure_translation(self):
        m = _make_machine()
        m.set_work_margins(5.0, 3.0, 7.0, 2.0)
        vp = ViewportConfig.from_machine(m)
        assert vp.margin_shift[0, 3] == -5.0
        assert vp.margin_shift[1, 3] == -2.0
        assert vp.margin_shift[2, 3] == 0.0
        upper_left = vp.margin_shift[:3, :3]
        np.testing.assert_array_equal(
            upper_left, np.identity(3, dtype=np.float32)
        )

    def test_margins_reduce_work_area(self):
        m = _make_machine()
        m.set_axis_extents(200.0, 200.0)
        m.set_work_margins(10.0, 5.0, 15.0, 8.0)
        vp = ViewportConfig.from_machine(m)
        assert vp.width_mm == 200.0 - 10.0 - 15.0
        assert vp.depth_mm == 200.0 - 5.0 - 8.0

    def test_no_margins_gives_identity_shift(self):
        m = _make_machine()
        vp = ViewportConfig.from_machine(m)
        np.testing.assert_array_equal(
            vp.margin_shift, np.identity(4, dtype=np.float32)
        )


class TestViewportConfigExtentFrame:
    def test_extent_frame_present_with_margins(self):
        m = _make_machine()
        m.set_axis_extents(200.0, 150.0)
        m.set_work_margins(10.0, 5.0, 15.0, 8.0)
        vp = ViewportConfig.from_machine(m)
        assert vp.extent_frame is not None
        x, y, w, h = vp.extent_frame
        assert w == 200.0
        assert h == 150.0
        assert x == -10.0
        assert y == -8.0

    def test_extent_frame_none_without_margins(self):
        m = _make_machine()
        vp = ViewportConfig.from_machine(m)
        assert vp.extent_frame is None

    def test_extent_frame_symmetric_margins(self):
        m = _make_machine()
        m.set_axis_extents(100.0, 100.0)
        m.set_work_margins(10.0, 10.0, 10.0, 10.0)
        vp = ViewportConfig.from_machine(m)
        assert vp.extent_frame is not None
        x, y, w, h = vp.extent_frame
        assert x == -10.0
        assert y == -10.0
        assert w == 100.0
        assert h == 100.0


class TestViewportConfigWcsOffset:
    def test_wcs_origin_is_workarea_origin(self):
        m = _make_machine()
        m.wcs_origin_is_workarea_origin = True
        m.update_wcs_offset("G54", (10.0, 20.0, 0.0))
        vp = ViewportConfig.from_machine(m)
        assert vp.wcs_offset_mm == (0.0, 0.0, 0.0)

    def test_wcs_offset_no_reverse(self):
        m = _make_machine()
        m.set_axis_extents(200.0, 200.0)
        m.set_work_margins(5.0, 3.0, 7.0, 2.0)
        m.update_wcs_offset("G54", (1.0, 2.0, 0.5))
        vp = ViewportConfig.from_machine(m)
        expected_x = -5.0 + 1.0
        expected_y = -2.0 + 2.0
        assert vp.wcs_offset_mm == pytest.approx((expected_x, expected_y, 0.5))

    def test_wcs_offset_with_reverse_x(self):
        m = _make_machine()
        m.set_axis_extents(200.0, 200.0)
        m.set_work_margins(5.0, 3.0, 7.0, 2.0)
        m.set_reverse_x_axis(True)
        m.update_wcs_offset("G54", (1.0, 2.0, 0.0))
        vp = ViewportConfig.from_machine(m)
        expected_x = -5.0 - 1.0
        expected_y = -2.0 + 2.0
        assert vp.wcs_offset_mm == pytest.approx((expected_x, expected_y, 0.0))

    def test_wcs_offset_with_reverse_y(self):
        m = _make_machine()
        m.set_axis_extents(200.0, 200.0)
        m.set_work_margins(5.0, 3.0, 7.0, 2.0)
        m.set_reverse_y_axis(True)
        m.update_wcs_offset("G54", (1.0, 2.0, 0.0))
        vp = ViewportConfig.from_machine(m)
        expected_x = -5.0 + 1.0
        expected_y = -2.0 - 2.0
        assert vp.wcs_offset_mm == pytest.approx((expected_x, expected_y, 0.0))

    def test_wcs_offset_y_down(self):
        m = _make_machine()
        m.origin = Origin.TOP_LEFT
        m.set_axis_extents(200.0, 200.0)
        m.set_work_margins(5.0, 3.0, 7.0, 2.0)
        m.update_wcs_offset("G54", (1.0, 2.0, 0.0))
        vp = ViewportConfig.from_machine(m)
        expected_x = -5.0 + 1.0
        expected_y = -3.0 + 2.0
        assert vp.wcs_offset_mm == pytest.approx((expected_x, expected_y, 0.0))


class TestViewportConfigNegativeFlags:
    def test_x_negative_reflects_reverse_x(self):
        m = _make_machine()
        m.set_reverse_x_axis(True)
        vp = ViewportConfig.from_machine(m)
        assert vp.x_negative is True
        assert vp.y_negative is False

    def test_y_negative_reflects_reverse_y(self):
        m = _make_machine()
        m.set_reverse_y_axis(True)
        vp = ViewportConfig.from_machine(m)
        assert vp.y_negative is True
        assert vp.x_negative is False

    def test_both_negative(self):
        m = _make_machine()
        m.set_reverse_x_axis(True)
        m.set_reverse_y_axis(True)
        vp = ViewportConfig.from_machine(m)
        assert vp.x_negative is True
        assert vp.y_negative is True

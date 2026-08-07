"""
Tests for the shared ColorLutProvider used by the renderers.
"""

from unittest.mock import MagicMock

import numpy as np

from rayforge.core.color import ColorSet
from rayforge.machine.models.laser import LaserHead
from rayforge.ui_gtk.shared.color_lut_provider import ColorLutProvider


def _theme_color_set() -> ColorSet:
    return ColorSet(
        {
            "cut": (1.0, 0.0, 0.0, 1.0),
            "engrave": np.full((256, 4), 0.5, dtype=np.float32),
            "travel": (0.0, 1.0, 0.0, 1.0),
            "zero_power": (0.0, 0.0, 1.0, 1.0),
        }
    )


def _laser_color_set(rgb: tuple) -> ColorSet:
    lut = np.zeros((256, 4), dtype=np.float32)
    lut[:] = (*rgb, 1.0)
    return ColorSet(
        {
            "cut": lut,
            "engrave": lut,
            "travel": (0.0, 1.0, 0.0, 1.0),
            "zero_power": (0.0, 0.0, 1.0, 1.0),
        }
    )


def test_no_laser_fallback_shapes():
    provider = ColorLutProvider(_theme_color_set(), {})
    assert provider.has_lasers is False
    assert provider.num_lasers == 1
    assert provider.cut_lut().shape == (256, 4)
    assert provider.engrave_lut_2d().shape == (256, 4)
    assert provider.ring_lut_2d().shape == (256, 4)


def test_color_set_and_laser_color_sets_properties():
    laser_sets = {
        "a": _laser_color_set((1.0, 0.0, 0.0)),
    }
    provider = ColorLutProvider(_theme_color_set(), laser_sets)
    assert provider.color_set is _theme_color_set()
    assert provider.laser_color_sets is laser_sets


def test_no_laser_ring_lut_is_white_ramp():
    provider = ColorLutProvider(_theme_color_set(), {})
    ring = provider.ring_lut_2d()
    assert np.allclose(ring[0], (0.0, 0.0, 0.0, 0.0))
    assert np.allclose(ring[-1], (1.0, 1.0, 1.0, 1.0), atol=0.05)


def test_multi_laser_2d_shapes():
    laser_sets = {
        "a": _laser_color_set((1.0, 0.0, 0.0)),
        "b": _laser_color_set((0.0, 0.0, 1.0)),
    }
    provider = ColorLutProvider(_theme_color_set(), laser_sets)
    assert provider.has_lasers is True
    assert provider.num_lasers == 2
    assert provider.cut_lut().shape == (2, 256, 4)
    assert provider.engrave_lut_2d().shape == (2, 256, 4)
    assert provider.ring_lut_2d().shape == (2, 256, 4)


def test_multi_laser_rows_follow_laser_order():
    laser_sets = {
        "a": _laser_color_set((1.0, 0.0, 0.0)),
        "b": _laser_color_set((0.0, 0.0, 1.0)),
    }
    provider = ColorLutProvider(_theme_color_set(), laser_sets)
    cut = provider.cut_lut()
    assert np.allclose(cut[0, -1], (1.0, 0.0, 0.0, 1.0))
    assert np.allclose(cut[1, -1], (0.0, 0.0, 1.0, 1.0))


def test_luts_cached_across_calls():
    provider = ColorLutProvider(_theme_color_set(), {})
    assert provider.cut_lut() is provider.cut_lut()
    assert provider.engrave_lut_2d() is provider.engrave_lut_2d()
    assert provider.ring_lut_2d() is provider.ring_lut_2d()


def test_invalidate_rebuilds_luts():
    provider = ColorLutProvider(_theme_color_set(), {})
    provider.cut_lut()
    provider.engrave_lut_2d()
    provider.ring_lut_2d()
    provider.invalidate()
    assert provider.cut_lut() is not None
    assert provider.engrave_lut_2d() is not None
    assert provider.ring_lut_2d() is not None


def test_from_machine_filters_non_laser_heads():
    head1 = LaserHead()
    head2 = LaserHead()
    head2.set_cut_color("#00ff00")
    head2.set_raster_color("#00ff00")
    machine = MagicMock(heads=[head1, MagicMock(uid="spindle"), head2])

    provider = ColorLutProvider.from_machine(machine, _theme_color_set())
    assert provider.has_lasers is True
    assert provider.num_lasers == 2
    assert provider.cut_lut().shape == (2, 256, 4)


def test_from_machine_with_none_machine():
    provider = ColorLutProvider.from_machine(None, _theme_color_set())
    assert provider.has_lasers is False
    assert provider.num_lasers == 1
    assert provider.cut_lut().shape == (256, 4)

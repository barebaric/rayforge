"""Shop mode swaps the Controls tab to the touch-friendly panels."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from rayforge.context import get_context


@pytest.fixture
def shop_panel(sync_machine):
    """A BottomPanel with only the shop-mode swap wiring."""
    from rayforge.ui_gtk.doceditor.bottom_panel import BottomPanel

    panel: Any = BottomPanel.__new__(BottomPanel)
    panel.machine = sync_machine
    panel.wcs_group = MagicMock()
    panel._jog_laser_box = MagicMock()
    panel._controls_widget = MagicMock()
    panel.shop_controls = MagicMock()
    panel.shop_jog = MagicMock()
    return panel


@pytest.mark.ui
def test_shop_mode_swaps_in_touch_panels(shop_panel):
    config = get_context().config

    config.set_shop_mode(True)
    shop_panel._sync_shop_mode()
    shop_panel._controls_widget.set_children.assert_called_with(
        shop_panel.shop_controls, shop_panel.shop_jog
    )

    config.set_shop_mode(False)
    shop_panel._sync_shop_mode()
    shop_panel._controls_widget.set_children.assert_called_with(
        shop_panel.wcs_group, shop_panel._jog_laser_box
    )


@pytest.mark.ui
def test_config_change_syncs_shop_mode(shop_panel):
    config = get_context().config

    config.set_shop_mode(True)
    shop_panel._on_config_changed(config)
    shop_panel._controls_widget.set_children.assert_called_with(
        shop_panel.shop_controls, shop_panel.shop_jog
    )


@pytest.mark.ui
def test_no_shop_panels_is_a_noop(sync_machine):
    from rayforge.ui_gtk.doceditor.bottom_panel import BottomPanel

    panel: Any = BottomPanel.__new__(BottomPanel)
    panel.machine = sync_machine
    panel.shop_controls = None
    panel.shop_jog = None
    panel._sync_shop_mode()


@pytest.mark.ui
def test_shop_steppers_share_jog_values(shop_panel, sync_machine):
    """A stepper tap on the shop panel updates the machine, the jog
    widget and the regular spin rows alike."""
    from rayforge.ui_gtk.machine.shop_mode_widget import ShopControlPanel

    shop_panel.jog_widget = MagicMock()
    shop_panel.speed_row = MagicMock()
    shop_panel.distance_row = MagicMock()
    shop_panel.shop_controls = ShopControlPanel()
    shop_panel.shop_controls.set_machine(sync_machine, None)
    shop_panel.shop_controls.speed_changed.connect(
        shop_panel._on_shop_speed_changed
    )
    shop_panel.shop_controls.distance_changed.connect(
        shop_panel._on_shop_distance_changed
    )

    shop_panel.shop_controls._on_speed_plus(None)
    assert sync_machine.jog_speed == 1100
    assert shop_panel.jog_widget.jog_speed == 1100
    assert shop_panel.shop_jog.jog_speed == 1100
    shop_panel.speed_row.set_value_in_base_units.assert_called_with(1100)

    shop_panel.shop_controls._on_distance_plus(None)
    assert shop_panel.jog_widget.jog_distance == 11.0
    assert shop_panel.shop_jog.jog_distance == 11.0
    shop_panel.distance_row.set_value_in_base_units.assert_called_with(11.0)

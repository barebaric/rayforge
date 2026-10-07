"""
Tests for the BottomPanel click-to-move mode state.

The panel coordinate shortcuts (lower-left / center / upper-right /
origin) moved into the MoveToPopover and are covered by
tests/ui_gtk/machine/test_move_to_popover.py.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from blinker import Signal


@pytest.fixture
def bottom_panel_modes(sync_machine):
    """A BottomPanel with mode state and signals wired, Gtk bypassed."""
    from rayforge.ui_gtk.doceditor.bottom_panel import BottomPanel

    bottom: Any = BottomPanel.__new__(BottomPanel)
    bottom.machine = sync_machine
    bottom.machine_cmd = MagicMock()
    bottom._click_to_zero_mode = False
    bottom._move_to_mode = False
    bottom.click_to_zero_mode_changed = Signal()
    bottom.move_to_mode_changed = Signal()
    bottom._update_wcs_ui = MagicMock()
    return bottom


@pytest.mark.ui
def test_set_move_to_mode_emits_signal(bottom_panel_modes):
    emissions = []

    def on_move_to_mode_changed(sender, **kw):
        emissions.append(kw["active"])

    bottom_panel_modes.move_to_mode_changed.connect(on_move_to_mode_changed)

    bottom_panel_modes.set_move_to_mode(True)

    assert bottom_panel_modes._move_to_mode is True
    assert emissions == [True]


@pytest.mark.ui
def test_set_move_to_mode_idempotent(bottom_panel_modes):
    emissions = []

    def on_move_to_mode_changed(sender, **kw):
        emissions.append(kw["active"])

    bottom_panel_modes.move_to_mode_changed.connect(on_move_to_mode_changed)

    bottom_panel_modes.set_move_to_mode(True)
    bottom_panel_modes.set_move_to_mode(True)

    assert emissions == [True]


@pytest.mark.ui
def test_click_to_move_toggled_flips_mode(bottom_panel_modes):
    emissions = []

    def on_move_to_mode_changed(sender, **kw):
        emissions.append(kw["active"])

    bottom_panel_modes.move_to_mode_changed.connect(on_move_to_mode_changed)

    bottom_panel_modes._on_click_to_move_toggled(None)

    assert bottom_panel_modes._move_to_mode is True
    assert emissions == [True]


@pytest.mark.ui
def test_toggle_move_to_mode_flips_state(bottom_panel_modes):
    emissions = []

    def on_move_to_mode_changed(sender, **kw):
        emissions.append(kw["active"])

    bottom_panel_modes.move_to_mode_changed.connect(on_move_to_mode_changed)

    bottom_panel_modes.toggle_move_to_mode()
    bottom_panel_modes.toggle_move_to_mode()

    assert bottom_panel_modes._move_to_mode is False
    assert emissions == [True, False]


@pytest.mark.ui
def test_move_to_mode_deactivates_click_to_zero(bottom_panel_modes):
    zero_emissions = []

    def on_click_to_zero_changed(sender, **kw):
        zero_emissions.append(kw["active"])

    bottom_panel_modes.click_to_zero_mode_changed.connect(
        on_click_to_zero_changed
    )
    bottom_panel_modes.set_click_to_zero_mode(True)
    zero_emissions.clear()

    bottom_panel_modes.set_move_to_mode(True)

    assert bottom_panel_modes._move_to_mode is True
    assert bottom_panel_modes._click_to_zero_mode is False
    assert zero_emissions == [False]


@pytest.mark.ui
def test_click_to_zero_mode_deactivates_move_to_mode(bottom_panel_modes):
    bottom_panel_modes.set_move_to_mode(True)

    bottom_panel_modes.set_click_to_zero_mode(True)

    assert bottom_panel_modes._click_to_zero_mode is True
    assert bottom_panel_modes._move_to_mode is False

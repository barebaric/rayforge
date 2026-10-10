"""
The bottom panel's Pointer Alignment toggle button mirrors the
machine's alignment state bidirectionally: clicking it toggles
alignment on the machine, and machine-side changes (e.g. from the
Move-To popover's switch) update the button.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def bottom_panel_toggle(sync_machine):
    """A BottomPanel skeleton with a real alignment toggle button."""
    from gi.repository import Gtk

    from rayforge.ui_gtk.doceditor.bottom_panel import BottomPanel

    bottom: Any = BottomPanel.__new__(BottomPanel)
    bottom.machine = sync_machine
    bottom._sync_wcs_model = MagicMock()
    bottom.job_origin_rows = MagicMock()
    bottom._updating_wcs_ui = False
    bottom.wcs_list = ["G54", "G55"]
    bottom.wcs_row = MagicMock()
    bottom.wcs_row.get_selected.return_value = 0
    bottom.zero_row = MagicMock()
    bottom.position_row = MagicMock()
    bottom.zero_x_btn = MagicMock()
    bottom.zero_y_btn = MagicMock()
    bottom.zero_z_btn = MagicMock()
    bottom.zero_here_btn = MagicMock()
    bottom.edit_offsets_btn = MagicMock()
    bottom.update_position_menu_sensitivity = MagicMock()
    bottom.doc = None
    bottom.pointer_alignment_btn = Gtk.ToggleButton()
    bottom._pointer_alignment_handler_id = (
        bottom.pointer_alignment_btn.connect(
            "toggled", bottom._on_pointer_alignment_toggled
        )
    )
    return bottom


def _enable_pointer_offset(machine):
    head = machine.get_default_laser_head()
    assert head is not None
    head.set_pointer_offset(10.0, 20.0)
    head.set_pointer_offset_enabled(True)


@pytest.mark.ui
def test_toggle_drives_machine(bottom_panel_toggle):
    bottom = bottom_panel_toggle
    _enable_pointer_offset(bottom.machine)

    bottom.pointer_alignment_btn.set_active(True)
    assert bottom.machine.pointer_alignment_enabled is True

    bottom.pointer_alignment_btn.set_active(False)
    assert bottom.machine.pointer_alignment_enabled is False


@pytest.mark.ui
def test_machine_state_syncs_to_button(bottom_panel_toggle):
    bottom = bottom_panel_toggle
    _enable_pointer_offset(bottom.machine)

    bottom.machine.set_pointer_alignment(True)
    bottom._on_pointer_alignment_changed(bottom.machine)
    assert bottom.pointer_alignment_btn.get_active() is True

    bottom.machine.set_pointer_alignment(False)
    bottom._on_pointer_alignment_changed(bottom.machine)
    assert bottom.pointer_alignment_btn.get_active() is False


@pytest.mark.ui
def test_button_insensitive_without_offset(bottom_panel_toggle):
    bottom = bottom_panel_toggle

    bottom._update_wcs_ui()

    assert bottom.pointer_alignment_btn.get_sensitive() is False
    assert bottom.pointer_alignment_btn.get_active() is False


@pytest.mark.ui
def test_button_sensitive_and_synced_with_offset(bottom_panel_toggle):
    bottom = bottom_panel_toggle
    _enable_pointer_offset(bottom.machine)
    bottom.machine.set_pointer_alignment(True)

    bottom._update_wcs_ui()

    assert bottom.pointer_alignment_btn.get_sensitive() is True
    assert bottom.pointer_alignment_btn.get_active() is True

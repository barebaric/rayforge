"""
The DRO position readout must stay truthful while pointer alignment
is on: it reports the cutting beam's WCS position and is never shifted
by the pointer offset (the move popover prefill is what shifts).
"""

from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def bottom_panel_dro(sync_machine):
    """A BottomPanel with the widgets _update_wcs_ui touches mocked."""
    from rayforge.ui_gtk.doceditor.bottom_panel import BottomPanel

    bottom: Any = BottomPanel.__new__(BottomPanel)
    bottom.machine = sync_machine
    # Force controller creation up front so its initial driver-state
    # sync does not clobber the machine position mid-update.
    assert sync_machine.driver is not None
    bottom.doc = None
    bottom._sync_wcs_model = MagicMock()
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
    return bottom


def _read_dro(bottom_panel_dro) -> str:
    bottom_panel_dro._update_wcs_ui()
    return bottom_panel_dro.position_row.set_subtitle.call_args.args[0]


@pytest.mark.ui
def test_dro_unaffected_by_pointer_alignment(bottom_panel_dro):
    machine = bottom_panel_dro.machine
    machine.update_wcs_offset("G54", (10.0, 20.0, 3.0))
    machine.device_state.machine_pos = (110.0, 120.0, 8.0)
    head = machine.get_default_laser_head()
    assert head is not None
    head.set_pointer_offset(12.0, -3.5)
    head.set_pointer_offset_enabled(True)

    plain = _read_dro(bottom_panel_dro)

    machine.set_pointer_alignment(True)

    aligned = _read_dro(bottom_panel_dro)

    assert plain == "X: 100.00   Y: 100.00   Z: 5.00"
    assert aligned == plain

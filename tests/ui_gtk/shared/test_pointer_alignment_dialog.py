"""
Tests for the PointerAlignmentDialog response handling: jobs burn with
the unshifted beam, so the user picks between a pointer dry-run,
turning alignment off and burning, or cancelling.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def alignment_dialog(sync_machine):
    """A dialog over a real machine with alignment enabled."""
    from gi.repository import Gtk

    from rayforge.ui_gtk.shared.pointer_alignment_dialog import (
        PointerAlignmentDialog,
    )

    machine = sync_machine
    head = machine.get_default_laser_head()
    assert head is not None
    head.set_pointer_offset(10.0, 20.0)
    head.set_pointer_offset_enabled(True)
    machine.set_pointer_alignment(True)

    parent = Gtk.Window()
    on_proceed = MagicMock()
    dialog = PointerAlignmentDialog(
        parent=parent, machine=machine, on_proceed=on_proceed
    )
    return dialog, machine, on_proceed


@pytest.mark.ui
def test_turn_off_and_burn(alignment_dialog):
    dialog, machine, on_proceed = alignment_dialog

    dialog._on_response(dialog, "turn-off-burn")

    assert machine.pointer_alignment_enabled is False
    on_proceed.assert_called_once_with(False)


@pytest.mark.ui
def test_dry_run_with_pointer(alignment_dialog):
    dialog, machine, on_proceed = alignment_dialog

    dialog._on_response(dialog, "dry-run")

    assert machine.pointer_alignment_enabled is True
    on_proceed.assert_called_once_with(True)


@pytest.mark.ui
def test_cancel(alignment_dialog):
    dialog, machine, on_proceed = alignment_dialog

    dialog._on_response(dialog, "cancel")

    assert machine.pointer_alignment_enabled is True
    on_proceed.assert_not_called()

"""
Tests for the unified win.move-head-here action.

The context menu item and the Ctrl+M shortcut both activate the same
action: with a fresh right-click position it moves immediately, without
one it enters the click-canvas-to-move-head mode.
"""

from unittest.mock import MagicMock

import pytest
from gi.repository import GLib

from rayforge.ui_gtk.canvas2d import context_menu


def _process_events():
    """Runs pending GLib main context events, including idle callbacks."""
    ctx = GLib.MainContext.default()
    while ctx.pending():
        ctx.iteration(False)


@pytest.fixture
def move_head_window(monkeypatch, sync_machine):
    """A MainWindow with only the move-head handler's dependencies."""
    from rayforge.ui_gtk import mainwindow

    config = MagicMock()
    config.machine = sync_machine
    monkeypatch.setattr(
        mainwindow, "get_context", lambda: MagicMock(config=config)
    )

    win = mainwindow.MainWindow.__new__(mainwindow.MainWindow)
    win.surface = MagicMock()
    win.surface.right_click_machine_pos = None
    win.machine_cmd = MagicMock()
    win.bottom_panel = MagicMock()
    win.bottom_panel.jog_speed = 1000
    return win


@pytest.mark.ui
def test_menu_invocation_moves_immediately(move_head_window, sync_machine):
    move_head_window.surface.right_click_machine_pos = (30.0, 40.0)

    move_head_window.on_move_head_here_clicked(None, None)

    move_head_window.machine_cmd.move_to.assert_called_once_with(
        sync_machine, 30.0, 40.0, speed=1000
    )
    assert move_head_window.surface.right_click_machine_pos is None


@pytest.mark.ui
def test_shortcut_invocation_enters_pick_mode(move_head_window):
    move_head_window.surface.right_click_machine_pos = None

    move_head_window.on_move_head_here_clicked(None, None)

    move_head_window.machine_cmd.move_to.assert_not_called()
    move_head_window.bottom_panel.toggle_move_to_mode.assert_called_once()


@pytest.mark.ui
def test_menu_invocation_consumes_stale_position(move_head_window):
    """A second (keyboard) invocation after a move must not reuse the
    consumed click position."""
    move_head_window.surface.right_click_machine_pos = (30.0, 40.0)
    move_head_window.on_move_head_here_clicked(None, None)

    move_head_window.machine_cmd.reset_mock()
    move_head_window.on_move_head_here_clicked(None, None)

    move_head_window.machine_cmd.move_to.assert_not_called()
    move_head_window.bottom_panel.toggle_move_to_mode.assert_called_once()


@pytest.mark.ui
def test_context_menu_close_clears_click_position():
    surface = MagicMock()
    surface.right_click_machine_pos = (30.0, 40.0)

    context_menu._on_context_menu_closed(MagicMock(), surface)
    assert surface.right_click_machine_pos == (30.0, 40.0)

    _process_events()
    assert surface.right_click_machine_pos is None


@pytest.mark.ui
def test_menu_invocation_moves_when_close_fires_first(
    move_head_window, sync_machine
):
    """GTK pops the menu down (emitting 'closed') before activating the
    item's action, so the close-time clearing must not eat the fresh
    click position before the action reads it."""
    move_head_window.surface.right_click_machine_pos = (30.0, 40.0)

    context_menu._on_context_menu_closed(MagicMock(), move_head_window.surface)
    move_head_window.on_move_head_here_clicked(None, None)
    _process_events()

    move_head_window.machine_cmd.move_to.assert_called_once_with(
        sync_machine, 30.0, 40.0, speed=1000
    )
    move_head_window.bottom_panel.toggle_move_to_mode.assert_not_called()

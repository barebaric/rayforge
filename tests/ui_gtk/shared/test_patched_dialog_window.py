import sys

import pytest
from gi.repository import Gtk

from rayforge.ui_gtk.shared.patched_dialog_window import (
    PatchedDialogWindow,
    _can_reactivate,
)


@pytest.mark.ui
def test_reactivation_skipped_on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _can_reactivate() is False


@pytest.mark.ui
def test_reactivation_kept_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert _can_reactivate() is True


@pytest.mark.ui
def test_reactivation_kept_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert _can_reactivate() is True


@pytest.mark.ui
def test_transient_for_kept_on_linux():
    parent = Gtk.Window()
    dialog = PatchedDialogWindow(transient_for=parent)
    assert dialog.get_transient_for() is parent


@pytest.mark.ui
def test_transient_for_detached_on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    parent = Gtk.Window()
    dialog = PatchedDialogWindow(transient_for=parent)
    assert dialog.get_transient_for() is None

"""Tests for the update settings on the general preferences page."""

import pytest

from rayforge.ui_gtk.settings.general_preferences_page import (
    GeneralPreferencesPage,
)

pytestmark = pytest.mark.ui


def _make_page():
    return GeneralPreferencesPage()


def test_auto_download_row_reflects_config(ui_context_initializer):
    ui_context_initializer.config.auto_download_app_updates = False

    page = _make_page()

    assert not page.auto_download_updates_row.get_active()


def test_toggling_auto_download_updates_config(ui_context_initializer):
    page = _make_page()

    page.auto_download_updates_row.set_active(False)

    assert ui_context_initializer.config.auto_download_app_updates is False


def test_auto_download_row_hidden_without_self_update(monkeypatch):
    monkeypatch.setattr(
        "rayforge.ui_gtk.settings.general_preferences_page."
        "supports_self_update",
        lambda: False,
    )

    page = _make_page()

    assert not page.auto_download_updates_row.get_visible()


def test_auto_download_row_shown_with_self_update(monkeypatch):
    monkeypatch.setattr(
        "rayforge.ui_gtk.settings.general_preferences_page."
        "supports_self_update",
        lambda: True,
    )

    page = _make_page()

    assert page.auto_download_updates_row.get_visible()


def test_check_updates_row_hidden_with_msix(monkeypatch):
    monkeypatch.setattr(
        "rayforge.ui_gtk.settings.general_preferences_page.is_msix",
        lambda: True,
    )

    page = _make_page()

    assert not page.check_updates_row.get_visible()


def test_check_updates_row_shown_without_msix(monkeypatch):
    monkeypatch.setattr(
        "rayforge.ui_gtk.settings.general_preferences_page.is_msix",
        lambda: False,
    )

    page = _make_page()

    assert page.check_updates_row.get_visible()

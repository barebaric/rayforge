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

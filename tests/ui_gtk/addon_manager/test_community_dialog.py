"""Tests for the community addon confirmation dialog."""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from gi.repository import Gio, Gtk

from rayforge.addon_mgr.addon import (
    Addon,
    AddonAuthor,
    AddonMetadata,
    AddonProvides,
)
from rayforge.addon_mgr.addon_manager import AddonState
from rayforge.ui_gtk.addon_manager.addon_list import AddonRow
from rayforge.ui_gtk.addon_manager.community_dialog import CommunityAddonDialog

pytestmark = pytest.mark.ui


def _addon(name: str, url: str = "") -> Addon:
    metadata = AddonMetadata(
        name=name,
        display_name=name,
        description="Test addon",
        version="1.0.0",
        depends=[],
        author=AddonAuthor(name="Test Author", email="test@test.com"),
        provides=AddonProvides(),
        url=url,
    )
    return Addon(path=Path(name), metadata=metadata)


def test_dialog_sets_heading_and_body():
    """The dialog warns that the addon is not provided by Rayforge."""
    dialog = CommunityAddonDialog(addon_name="Test Addon")
    assert dialog.get_heading() == "Install Community Addon?"
    body = dialog.get_body()
    assert "Test Addon" in body
    assert "not provided by the Rayforge team" in body
    assert "trust" in body


def test_dialog_body_names_the_author():
    """The author is named so the user knows whom to trust."""
    dialog = CommunityAddonDialog(
        addon_name="Test Addon", author="Test Author"
    )
    assert "Test Author" in dialog.get_body()


def test_dialog_defaults_to_cancel():
    """Cancel is the safe default and close response."""
    dialog = CommunityAddonDialog(addon_name="Test Addon")
    assert dialog.get_default_response() == "cancel"
    assert dialog.get_close_response() == "cancel"


def test_dialog_requests_width_for_horizontal_responses():
    """libadwaita stacks responses below the requested width."""
    dialog = CommunityAddonDialog(addon_name="Test Addon")
    width, _height = dialog.get_default_size()
    assert width >= 480


def test_dialog_install_response_runs_callback():
    """Confirming runs the install callback."""
    on_install = MagicMock()
    dialog = CommunityAddonDialog(
        addon_name="Test Addon", on_install=on_install
    )
    dialog._on_response(dialog, "install")
    on_install.assert_called_once()


def test_dialog_cancel_response_runs_cancel_callback():
    """Cancelling runs the cancel callback."""
    on_cancel = MagicMock()
    dialog = CommunityAddonDialog(addon_name="Test Addon", on_cancel=on_cancel)
    dialog._on_response(dialog, "cancel")
    on_cancel.assert_called_once()


def test_dialog_shows_repository_link():
    """The repository URL is shown as a clickable link."""
    url = "https://github.com/someone/rayforge-test-addon"
    dialog = CommunityAddonDialog(addon_name="Test Addon", repository_url=url)
    link = cast(Gtk.Label, dialog.get_extra_child())
    assert url in link.get_text()


def test_dialog_omits_link_without_repository():
    """No link child is added when no URL is known."""
    dialog = CommunityAddonDialog(addon_name="Test Addon")
    assert dialog.get_extra_child() is None


def test_dialog_link_opens_browser():
    """Clicking the repository link opens it in the user's browser."""
    dialog = CommunityAddonDialog(
        addon_name="Test Addon",
        repository_url="https://github.com/someone/rayforge-test-addon",
    )
    link = dialog.get_extra_child()
    with patch(
        "rayforge.ui_gtk.addon_manager.community_dialog.webbrowser.open"
    ) as mock_open:
        assert dialog._on_activate_link(link, "https://example.com/repo")
    mock_open.assert_called_once_with("https://example.com/repo")


def _row_icon_file(row: AddonRow) -> str:
    """Return the basename of the icon file used by the row."""
    icon = cast(Gtk.Image, row.get_first_child())
    gicon = cast(Gio.FileIcon, icon.get_gicon())
    return gicon.get_file().get_basename() or ""


def test_community_addon_row_uses_community_icon():
    """Installed community addons get the dedicated community icon."""
    addon = _addon("community_addon", url="https://github.com/x/y")
    row = AddonRow(addon, AddonState.ENABLED.value, lambda a: None)
    assert _row_icon_file(row) == "addon-community-symbolic.svg"


def test_official_addon_row_uses_default_icon():
    """Addons from the Rayforge team keep the default addon icon."""
    addon = _addon(
        "official_addon", url="https://github.com/barebaric/rayforge-addon-x"
    )
    row = AddonRow(addon, AddonState.ENABLED.value, lambda a: None)
    assert _row_icon_file(row) == "addon-symbolic.svg"

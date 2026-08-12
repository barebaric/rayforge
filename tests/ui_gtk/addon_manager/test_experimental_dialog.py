"""Tests for the experimental addon confirmation dialog."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rayforge.addon_mgr.addon import (
    Addon,
    AddonAuthor,
    AddonMaturity,
    AddonMetadata,
    AddonProvides,
)
from rayforge.ui_gtk.addon_manager.addon_list import AddonListWidget
from rayforge.ui_gtk.addon_manager.experimental_dialog import (
    ExperimentalAddonDialog,
)

pytestmark = pytest.mark.ui


def _addon(name: str, maturity: AddonMaturity = AddonMaturity.STABLE) -> Addon:
    metadata = AddonMetadata(
        name=name,
        display_name=name,
        description="Test addon",
        version="1.0.0",
        depends=[],
        author=AddonAuthor(name="Test", email="test@test.com"),
        provides=AddonProvides(),
        maturity=maturity,
    )
    return Addon(path=Path(name), metadata=metadata)


def test_dialog_sets_heading_and_body():
    """The dialog warns about the experimental addon by name."""
    dialog = ExperimentalAddonDialog(addon_name="Test Addon")
    assert dialog.get_heading() == "Enable Experimental Addon?"
    body = dialog.get_body()
    assert "Test Addon" in body
    assert "experimental" in body


def test_dialog_defaults_to_cancel():
    """Cancel is the safe default and close response."""
    dialog = ExperimentalAddonDialog(addon_name="Test Addon")
    assert dialog.get_default_response() == "cancel"
    assert dialog.get_close_response() == "cancel"


def test_dialog_enable_response_runs_callback():
    """Confirming calls the enable callback and closes the dialog."""
    on_enable = MagicMock()
    dialog = ExperimentalAddonDialog(
        addon_name="Test Addon", on_enable=on_enable
    )
    dialog._on_response(dialog, "enable")
    on_enable.assert_called_once()


def test_dialog_cancel_response_runs_cancel_callback():
    """Cancelling calls the cancel callback and closes the dialog."""
    on_cancel = MagicMock()
    dialog = ExperimentalAddonDialog(
        addon_name="Test Addon", on_cancel=on_cancel
    )
    dialog._on_response(dialog, "cancel")
    on_cancel.assert_called_once()


def test_toggle_experimental_shows_dialog(ui_context_initializer):
    """Enabling an experimental addon asks for confirmation first."""
    addon = _addon("test_exp", AddonMaturity.EXPERIMENTAL)
    am = ui_context_initializer.addon_mgr
    am.disabled_addons["test_exp"] = addon
    am.enable_addon = MagicMock(return_value=True)

    widget = AddonListWidget()
    with patch(
        "rayforge.ui_gtk.addon_manager.addon_list.ExperimentalAddonDialog"
    ) as mock_dialog:
        widget._on_toggle_addon("test_exp", True)
        mock_dialog.assert_called_once()
        mock_dialog.return_value.present.assert_called_once()
        am.enable_addon.assert_not_called()
        mock_dialog.call_args.kwargs["on_cancel"]()
        am.enable_addon.assert_not_called()
        mock_dialog.call_args.kwargs["on_enable"]()
        am.enable_addon.assert_called_once_with("test_exp")


def test_toggle_stable_enables_directly(ui_context_initializer):
    """Enabling a stable addon does not show the confirmation dialog."""
    addon = _addon("test_stable")
    am = ui_context_initializer.addon_mgr
    am.disabled_addons["test_stable"] = addon
    am.enable_addon = MagicMock(return_value=True)

    widget = AddonListWidget()
    with patch(
        "rayforge.ui_gtk.addon_manager.addon_list.ExperimentalAddonDialog"
    ) as mock_dialog:
        widget._on_toggle_addon("test_stable", True)
        am.enable_addon.assert_called_once_with("test_stable")
        mock_dialog.assert_not_called()

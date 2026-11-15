# flake8: noqa: E402
"""UI tests for the addon detail dialog and row activation."""

import time
from typing import cast
from unittest.mock import MagicMock, patch

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import GLib, Gtk

from rayforge.addon_mgr.addon import (
    AddonAuthor,
    AddonMetadata,
    AddonProvides,
)
from rayforge.shared.util.versioning import UnknownVersion
from rayforge.ui_gtk.addon_manager.addon_detail_dialog import (
    AddonDetailDialog,
    resolve_screenshots,
)

pytestmark = pytest.mark.ui


def _pump_until(condition, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        while GLib.MainContext.default().iteration(False):
            pass
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def _meta(name: str = "test_addon", **kwargs) -> AddonMetadata:
    defaults = {
        "name": name,
        "display_name": "Test Addon",
        "description": "Short description",
        "version": "1.0.0",
        "depends": [],
        "author": AddonAuthor(name="Test Author", email="t@example.com"),
        "provides": AddonProvides(),
        "url": "https://github.com/example/rayforge-test-addon",
    }
    defaults.update(kwargs)
    return AddonMetadata(**defaults)


def test_dialog_shows_local_metadata(ui_context_initializer):
    context = ui_context_initializer
    with patch.object(
        context.addon_mgr, "fetch_registry_cached", return_value=[]
    ):
        dialog = AddonDetailDialog(None, _meta())
    try:
        assert dialog.get_title() == "Test Addon"
        assert dialog._description_label.get_label() == "Short description"
        assert dialog._description_label.get_visible() is True
        assert dialog._markdown_view.get_visible() is False
        assert "Test Author" in dialog._meta_label.get_label()
        assert "1.0.0" in dialog._meta_label.get_label()
    finally:
        dialog.destroy()


def test_dialog_backfills_screenshots_and_description(
    ui_context_initializer,
):
    context = ui_context_initializer
    entry = _meta(
        long_description="A much longer description.",
        screenshots=["https://example.com/shot.png"],
    )
    with (
        patch.object(
            context.addon_mgr, "fetch_registry_cached", return_value=[entry]
        ),
        patch(
            "rayforge.ui_gtk.shared.async_image.fetch_image_bytes",
            return_value=None,
        ),
    ):
        dialog = AddonDetailDialog(None, _meta())
        try:
            assert _pump_until(dialog._carousel_box.get_visible)
            assert dialog._markdown_view.get_visible() is True
            assert dialog._markdown_view._text == (
                "A much longer description."
            )
            assert dialog._description_label.get_visible() is False
        finally:
            dialog.destroy()


def test_dialog_renders_long_description_as_markdown(
    ui_context_initializer,
):
    context = ui_context_initializer
    meta = _meta(long_description="**Bold** text")
    with patch.object(
        context.addon_mgr, "fetch_registry_cached", return_value=[]
    ):
        dialog = AddonDetailDialog(None, meta)
    try:
        assert dialog._markdown_view.get_visible() is True
        assert dialog._description_label.get_visible() is False
        label = cast(Gtk.Label, dialog._markdown_view.get_first_child())
        assert "<b>Bold</b>" in label.get_label()
    finally:
        dialog.destroy()


def test_dialog_without_screenshots_hides_carousel(ui_context_initializer):
    context = ui_context_initializer
    with patch.object(
        context.addon_mgr, "fetch_registry_cached", return_value=[]
    ):
        dialog = AddonDetailDialog(None, _meta())
    try:
        assert dialog._carousel_box.get_visible() is False
    finally:
        dialog.destroy()


def test_dialog_without_callbacks_hides_action(ui_context_initializer):
    context = ui_context_initializer
    with patch.object(
        context.addon_mgr, "fetch_registry_cached", return_value=[]
    ):
        dialog = AddonDetailDialog(None, _meta())
    try:
        assert dialog._action_box.get_visible() is False
    finally:
        dialog.destroy()


def test_dialog_install_button_runs_callback(ui_context_initializer):
    context = ui_context_initializer
    on_install = MagicMock()
    meta = _meta()
    with patch.object(
        context.addon_mgr, "fetch_registry_cached", return_value=[]
    ):
        dialog = AddonDetailDialog(None, meta, on_install=on_install)
    try:
        btn = cast(Gtk.Button, dialog._action_box.get_first_child())
        assert btn.get_label() == "Install"
        btn.emit("clicked")
        on_install.assert_called_once_with(meta, False)
    finally:
        dialog.destroy()


def test_registry_row_activation_opens_detail(ui_context_initializer):
    from rayforge.ui_gtk.addon_manager import addon_dialog

    context = ui_context_initializer
    meta = _meta()
    with patch.object(
        context.addon_mgr, "fetch_registry", return_value=[meta]
    ):
        dialog = addon_dialog.AddonRegistryDialog(None, MagicMock())
        try:
            assert _pump_until(
                lambda: dialog.stack.get_visible_child_name() == "list"
            )
            row = dialog.list_box.get_row_at_index(0)
            with patch.object(addon_dialog, "AddonDetailDialog") as detail:
                dialog._on_row_activated(dialog.list_box, row)
            detail.assert_called_once()
            assert detail.call_args.args[1] is meta
        finally:
            dialog.destroy()


def test_installed_row_activation_opens_detail(ui_context_initializer):
    from rayforge.ui_gtk.addon_manager import addon_list

    widget = addon_list.AddonListWidget()
    row = widget.list_box.get_row_at_index(0)
    assert row is not None
    with patch.object(addon_list, "AddonDetailDialog") as detail:
        widget._on_row_activated(widget.list_box, row)
    detail.assert_called_once()


def test_resolve_screenshots_passes_urls_through():
    meta = _meta(screenshots=["https://example.com/a.png"])
    assert resolve_screenshots(meta) == [("https://example.com/a.png", None)]


def test_resolve_screenshots_pins_release_version():
    meta = _meta(
        version="1.2.3",
        screenshots=["docs/shot.png"],
    )
    base = "https://raw.githubusercontent.com/example/rayforge-test-addon"
    assert resolve_screenshots(meta) == [
        (f"{base}/1.2.3/docs/shot.png", f"{base}/main/docs/shot.png")
    ]


def test_resolve_screenshots_keeps_v_prefix():
    meta = _meta(version="v1.5.1", screenshots=["docs/shot.png"])
    url, _fallback = resolve_screenshots(meta)[0]
    assert "/v1.5.1/docs/shot.png" in url


def test_resolve_screenshots_builtin_uses_app_repo():
    meta = _meta(
        url="",
        version=UnknownVersion,
        screenshots=["website/static/screenshots/x.webp"],
    )
    with patch(
        "rayforge.ui_gtk.addon_manager.addon_detail_dialog.__version__",
        "1.2.3",
    ):
        url, fallback = resolve_screenshots(meta)[0]
    assert url == (
        "https://raw.githubusercontent.com/barebaric/rayforge/1.2.3/"
        "website/static/screenshots/x.webp"
    )
    assert fallback == (
        "https://raw.githubusercontent.com/barebaric/rayforge/main/"
        "website/static/screenshots/x.webp"
    )


def test_resolve_screenshots_dev_version_uses_main():
    meta = _meta(
        version="1.2.3.post4+git.abc123",
        screenshots=["docs/shot.png"],
    )
    base = "https://raw.githubusercontent.com/example/rayforge-test-addon"
    assert resolve_screenshots(meta) == [(f"{base}/main/docs/shot.png", None)]


def test_resolve_screenshots_skips_non_github_repos():
    meta = _meta(
        url="https://gitlab.com/example/addon",
        screenshots=["docs/shot.png"],
    )
    assert resolve_screenshots(meta) == []


def test_dialog_shows_local_screenshots(ui_context_initializer):
    context = ui_context_initializer
    meta = _meta(url="", screenshots=["website/static/screenshots/x.webp"])
    with (
        patch.object(
            context.addon_mgr, "fetch_registry_cached", return_value=[]
        ),
        patch(
            "rayforge.ui_gtk.shared.async_image.fetch_image_bytes",
            return_value=None,
        ),
    ):
        dialog = AddonDetailDialog(None, meta)
    try:
        assert dialog._carousel_box.get_visible() is True
    finally:
        dialog.destroy()

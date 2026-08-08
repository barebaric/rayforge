"""
UI tests: the tool-manager settings page is contributed and rendered.
"""

from typing import Any, cast

import pytest
from gi.repository import Gtk
from tool_library.edit_dialog import AddEditToolDialog
from tool_library.frontend import ToolManagerPage, register_settings_pages
from tool_library.tool import CATEGORY_NAMES, CATEGORY_PARAMS, Tool

from rayforge.ui_gtk.settings.registry import settings_page_registry
from rayforge.ui_gtk.settings.settings_dialog import SettingsWindow

pytestmark = pytest.mark.ui


def test_tool_page_registered_via_hook(ui_context):
    """The frontend hook registers ToolManagerPage when addons load."""
    before = len(settings_page_registry.get_pages())
    register_settings_pages(settings_page_registry)
    register_settings_pages(settings_page_registry)
    pages = settings_page_registry.get_pages()

    assert ToolManagerPage in pages
    assert len(pages) == before + 1  # deduped


def test_tool_page_present_in_settings_dialog(ui_context):
    """SettingsWindow renders a 'Tools' page from the addon registry."""
    win = SettingsWindow()
    pages = cast(Any, win.content_stack.get_pages())
    titles = [
        pages.get_item(i).get_title() for i in range(pages.get_n_items())
    ]
    assert "Tools" in titles
    win.destroy()


def test_dialog_geometry_adapts_to_category(ui_context):
    """Switching category rebuilds the geometry rows for that shape."""
    win = SettingsWindow()
    root = win.get_root()
    dlg = AddEditToolDialog(cast(Gtk.Window, root) if root else None)
    try:
        assert "corner_radius" not in dlg.param_keys()
        dlg._category.set_selected(CATEGORY_NAMES.index("BULL_NOSE"))
        assert "corner_radius" in dlg.param_keys()
        dlg._category.set_selected(CATEGORY_NAMES.index("DRILL"))
        assert "corner_radius" not in dlg.param_keys()
        assert "shank_diameter" not in dlg.param_keys()
        cat = CATEGORY_NAMES[dlg._category.get_selected()]
        assert set(dlg.param_keys()) == {s.key for s in CATEGORY_PARAMS[cat]}
    finally:
        dlg.destroy()
        win.destroy()


def test_dialog_length_rows_use_unit_helper(ui_context):
    """Length fields are unit-aware and round-trip base mm."""
    win = SettingsWindow()
    root = win.get_root()
    tool = cast(Any, Tool.create_default("Mine"))
    dlg = AddEditToolDialog(
        cast(Gtk.Window, root) if root else None, tool=tool
    )
    try:
        assert dlg.is_length_param("diameter")
        assert not dlg.is_length_param("flute_count")
        assert dlg.get_tool().diameter() == pytest.approx(6.0)
    finally:
        dlg.destroy()
        win.destroy()

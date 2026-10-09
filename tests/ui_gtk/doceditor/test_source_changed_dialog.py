"""Tests for the dialog and monitor that reload changed source files."""

from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from rayforge.context import get_context
from rayforge.core.source_asset import SourceAsset
from rayforge.image.svg.renderer import SVG_RENDERER
from rayforge.ui_gtk.canvas2d import context_menu
from rayforge.ui_gtk.doceditor.source_changed_dialog import (
    SourceChangedDialog,
)
from rayforge.ui_gtk.doceditor.source_reload import SourceReloadMonitor
from rayforge.ui_gtk.main_menu import MainMenu
from rayforge.ui_gtk.settings.general_preferences_page import (
    GeneralPreferencesPage,
)

pytestmark = pytest.mark.ui


def _asset(name: str, auto_reload: bool = False) -> SourceAsset:
    return SourceAsset(
        source_file=Path("/tmp") / name,
        original_data=b"<svg/>",
        renderer=SVG_RENDERER,
        auto_reload=auto_reload,
    )


class TestSourceChangedDialog:
    def test_single_file_names_it(self):
        dialog = SourceChangedDialog([_asset("part.svg")])
        assert dialog.get_heading() == "Imported File Changed"
        assert "part.svg" in dialog.get_body()

    def test_several_files_are_listed(self):
        assets = [_asset("a.svg"), _asset("b.svg")]
        dialog = SourceChangedDialog(assets)
        assert dialog.get_heading() == "Imported Files Changed"
        assert dialog.selected_assets() == assets

    def test_defaults(self):
        dialog = SourceChangedDialog([_asset("part.svg")])
        assert dialog.get_default_response() == "reload"
        assert dialog.get_close_response() == "ignore"

    def test_reload_passes_selection_and_choice(self):
        assets = [_asset("a.svg"), _asset("b.svg")]
        on_reload = MagicMock()
        dialog = SourceChangedDialog(assets, on_reload=on_reload)
        dialog.set_selected(assets[0], False)
        dialog.set_always(True)

        dialog._on_response(dialog, "reload")

        on_reload.assert_called_once_with([assets[1]], True, [assets[0]])

    def test_ignore_passes_all_assets(self):
        assets = [_asset("a.svg"), _asset("b.svg")]
        on_ignore = MagicMock()
        dialog = SourceChangedDialog(assets, on_ignore=on_ignore)

        dialog._on_response(dialog, "ignore")

        on_ignore.assert_called_once_with(assets)


class _FakeDialog:
    instances: ClassVar[list["_FakeDialog"]] = []

    def __init__(self, assets, on_reload, on_ignore):
        self.assets = assets
        self.on_reload: Any = on_reload
        self.on_ignore: Any = on_ignore
        self.presented = False
        _FakeDialog.instances.append(self)

    def present(self, parent):
        self.presented = True


@pytest.fixture
def monitor():
    _FakeDialog.instances = []
    editor = MagicMock()
    watcher = MagicMock()
    monitor = SourceReloadMonitor(
        MagicMock(),
        editor,
        watcher=watcher,
        dialog_factory=_FakeDialog,
        enabled=lambda: True,
    )
    return monitor, editor, watcher


class TestSourceReloadMonitor:
    def test_changed_files_open_one_dialog(self, monitor):
        mon, _editor, watcher = monitor
        assets = [_asset("a.svg"), _asset("b.svg")]
        watcher.poll.return_value = assets

        assert mon.tick() is True

        (dialog,) = _FakeDialog.instances
        assert dialog.presented
        assert dialog.assets == assets

    def test_auto_reload_files_reload_without_asking(self, monitor):
        mon, editor, watcher = monitor
        asset = _asset("a.svg", auto_reload=True)
        watcher.poll.return_value = [asset]

        mon.tick()

        editor.reload.reload_source.assert_called_once_with(asset)
        watcher.acknowledge.assert_called_once_with(asset)
        assert _FakeDialog.instances == []

    def test_further_changes_wait_for_the_open_dialog(self, monitor):
        mon, _editor, watcher = monitor
        first, second = _asset("a.svg"), _asset("b.svg")
        watcher.poll.return_value = [first]
        mon.tick()
        watcher.poll.return_value = [second]
        mon.tick()
        assert len(_FakeDialog.instances) == 1

        _FakeDialog.instances[0].on_ignore([first])

        assert len(_FakeDialog.instances) == 2
        assert _FakeDialog.instances[1].assets == [second]

    def test_reload_choice_is_applied(self, monitor):
        mon, editor, watcher = monitor
        keep, skip = _asset("a.svg"), _asset("b.svg")
        watcher.poll.return_value = [keep, skip]
        mon.tick()

        _FakeDialog.instances[0].on_reload([keep], True, [skip])

        editor.reload.reload_source.assert_called_once_with(keep)
        assert keep.auto_reload is True
        assert skip.auto_reload is False
        editor.mark_as_unsaved.assert_called_once()
        watcher.acknowledge.assert_any_call(keep)

    def test_check_now_uses_immediate_check(self, monitor):
        mon, _editor, watcher = monitor
        asset = _asset("a.svg")
        watcher.check_now.return_value = [asset]

        mon.check_now()

        assert _FakeDialog.instances[0].assets == [asset]


class TestMonitorPreferenceAndActions:
    def test_disabled_watching_does_nothing(self):
        _FakeDialog.instances = []
        watcher = MagicMock()
        mon = SourceReloadMonitor(
            MagicMock(),
            MagicMock(),
            watcher=watcher,
            dialog_factory=_FakeDialog,
            enabled=lambda: False,
        )

        assert mon.tick() is True
        mon.check_now()

        watcher.poll.assert_not_called()
        watcher.check_now.assert_not_called()
        assert _FakeDialog.instances == []

    def test_reload_assets_reloads_and_acknowledges(self, monitor):
        mon, editor, watcher = monitor
        a, b = _asset("a.svg"), _asset("b.svg")

        mon.reload_assets([a, b])

        assert editor.reload.reload_source.call_count == 2
        watcher.acknowledge.assert_any_call(a)
        watcher.acknowledge.assert_any_call(b)

    def test_relink_choice_relinks_and_acknowledges(self, monitor):
        mon, editor, watcher = monitor
        asset = _asset("a.svg")
        path = Path("/tmp/elsewhere/a.svg")

        mon.relink_to(asset, path)

        editor.reload.relink_source.assert_called_once_with(asset, path)
        watcher.acknowledge.assert_called_once_with(asset)


class TestMenus:
    def _actions(self, model) -> set[str]:
        found = set()
        for i in range(model.get_n_items()):
            action = model.get_item_attribute_value(i, "action", None)
            if action is not None:
                found.add(action.get_string())
            for link in ("section", "submenu"):
                sub = model.get_item_link(i, link)
                if sub is not None:
                    found |= self._actions(sub)
        return found

    def test_canvas_item_menu_offers_reload_and_relink(self):
        actions = self._actions(context_menu._MENU_MODELS["item"])

        assert {"win.reload-source", "win.relink-source"} <= actions

    def test_file_menu_offers_reload_and_relink(self):
        actions = self._actions(MainMenu())

        assert {"win.reload-source", "win.relink-source"} <= actions


class TestWatchPreference:
    def test_switch_reflects_and_updates_config(self, context_initializer):
        config = get_context().config
        page = GeneralPreferencesPage()
        assert page.watch_sources_row.get_active() is True

        page.watch_sources_row.set_active(False)

        assert config.watch_source_files is False

    def test_default_monitor_follows_config(self, context_initializer):
        watcher = MagicMock()
        watcher.poll.return_value = []
        mon = SourceReloadMonitor(MagicMock(), MagicMock(), watcher=watcher)
        get_context().config.set_watch_source_files(False)

        mon.tick()

        watcher.poll.assert_not_called()

"""Tests for the dialog and monitor that reload changed source files."""

from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from rayforge.core.source_asset import SourceAsset
from rayforge.image.svg.renderer import SVG_RENDERER
from rayforge.ui_gtk.doceditor.source_changed_dialog import (
    SourceChangedDialog,
)
from rayforge.ui_gtk.doceditor.source_reload import SourceReloadMonitor

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
        MagicMock(), editor, watcher=watcher, dialog_factory=_FakeDialog
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

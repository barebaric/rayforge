import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rayforge.core.doc import Doc
from rayforge.core.source_asset import SourceAsset
from rayforge.core.vectorization_spec import (
    LayerImportMode,
    PassthroughSpec,
)
from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.source_watcher import SourceWatcher
from rayforge.image import importer_registry
from rayforge.shared.tasker.manager import TaskManager

SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="80mm" '
    'viewBox="0 0 100 80"><rect x="10" y="10" width="{w}" height="10" '
    'fill="none" stroke="#ff0000"/></svg>'
)


@pytest.fixture
def editor(context_initializer):
    task_manager = MagicMock(spec=TaskManager)
    editor = DocEditor(task_manager, context_initializer, Doc())
    yield editor
    editor.cleanup()


def _write(path: Path, width: int, mtime_ns: int) -> None:
    path.write_text(SVG.format(w=width))
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _import(editor: DocEditor, path: Path) -> SourceAsset:
    importer_cls = importer_registry.get_for_file(path)
    assert importer_cls is not None
    spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)
    result = importer_cls(path.read_bytes(), path).get_doc_items(spec)
    assert result is not None and result.payload is not None
    editor.file._finalize_import_on_main_thread(
        result.payload, path, None, spec
    )
    assert result.payload.source is not None
    return result.payload.source


@pytest.fixture
def imported(editor, tmp_path):
    path = tmp_path / "part.svg"
    _write(path, 20, 1_000_000_000)
    asset = _import(editor, path)
    watcher = SourceWatcher(editor)
    assert watcher.poll() == []
    assert watcher.poll() == []
    return watcher, asset, path


class TestSourceWatcher:
    def test_unchanged_file_is_not_reported(self, imported):
        watcher, _asset, _path = imported
        assert watcher.poll() == []

    def test_change_is_reported_once_the_file_is_stable(self, imported):
        watcher, asset, path = imported
        _write(path, 30, 2_000_000_000)

        assert watcher.poll() == []
        assert watcher.poll() == [asset]
        assert watcher.poll() == []

    def test_file_still_being_written_is_not_reported(self, imported):
        watcher, _asset, path = imported
        _write(path, 30, 2_000_000_000)
        assert watcher.poll() == []
        _write(path, 35, 3_000_000_000)
        assert watcher.poll() == []

    def test_touch_without_new_content_is_not_reported(self, imported):
        watcher, _asset, path = imported
        os.utime(path, ns=(5_000_000_000, 5_000_000_000))

        assert watcher.poll() == []
        assert watcher.poll() == []

    def test_acknowledged_content_is_not_reported_again(self, imported):
        watcher, asset, path = imported
        _write(path, 30, 2_000_000_000)
        watcher.poll()
        assert watcher.poll() == [asset]
        watcher.acknowledge(asset)
        os.utime(path, ns=(4_000_000_000, 4_000_000_000))

        assert watcher.poll() == []
        assert watcher.poll() == []

        _write(path, 40, 6_000_000_000)
        watcher.poll()
        assert watcher.poll() == [asset]

    def test_missing_file_is_not_reported(self, imported):
        watcher, _asset, path = imported
        path.unlink()

        assert watcher.poll() == []
        assert watcher.poll() == []

    def test_check_now_reports_without_waiting(self, editor, tmp_path):
        path = tmp_path / "part.svg"
        _write(path, 20, 1_000_000_000)
        asset = _import(editor, path)
        _write(path, 30, 2_000_000_000)

        watcher = SourceWatcher(editor)

        assert watcher.check_now() == [asset]
        assert watcher.poll() == []

    def test_unused_source_is_not_watched(self, editor, tmp_path):
        path = tmp_path / "part.svg"
        _write(path, 20, 1_000_000_000)
        asset = _import(editor, path)
        for wp in list(editor.doc.all_workpieces):
            wp.parent.remove_child(wp)
        _write(path, 30, 2_000_000_000)

        assert SourceWatcher(editor).check_now() == []
        assert asset.uid in editor.doc.source_assets

    def test_reloaded_and_undone_content_is_not_reported(
        self, editor, imported
    ):
        watcher, asset, path = imported
        _write(path, 30, 2_000_000_000)
        watcher.poll()
        assert watcher.poll() == [asset]
        editor.reload.reload_source(asset)
        watcher.acknowledge(asset)
        editor.history_manager.undo()

        assert watcher.poll() == []
        assert watcher.poll() == []

    def test_follows_document_switch(self, editor, imported, tmp_path):
        watcher, _asset, _path = imported
        editor.set_doc(Doc())
        other = tmp_path / "other.svg"
        _write(other, 20, 1_000_000_000)
        asset = _import(editor, other)
        _write(other, 30, 2_000_000_000)

        assert watcher.check_now() == [asset]


class TestAutoReloadFlag:
    def test_defaults_to_false(self, editor, tmp_path):
        path = tmp_path / "part.svg"
        _write(path, 20, 1_000_000_000)
        assert _import(editor, path).auto_reload is False

    def test_round_trips_through_dict(self, editor, tmp_path):
        path = tmp_path / "part.svg"
        _write(path, 20, 1_000_000_000)
        asset = _import(editor, path)
        asset.auto_reload = True

        data = asset.to_dict()
        restored = SourceAsset.from_dict(data)

        assert data["auto_reload"] is True
        assert restored.auto_reload is True
        assert "auto_reload" not in restored.extra

    def test_old_projects_load_without_flag(self, editor, tmp_path):
        path = tmp_path / "part.svg"
        _write(path, 20, 1_000_000_000)
        data = _import(editor, path).to_dict()
        del data["auto_reload"]

        assert SourceAsset.from_dict(data).auto_reload is False

# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import Gtk

from rayforge.doceditor.editor import DocEditor
from rayforge.image.base_importer import ImporterFeature
from rayforge.image.postscript import ghostscript
from rayforge.ui_gtk.doceditor.import_dialog import ImportDialog

EPS = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n"
FEATURES = {ImporterFeature.DIRECT_VECTOR, ImporterFeature.BITMAP_TRACING}


@pytest.fixture
def editor(ui_context_initializer, ui_task_mgr):
    editor = DocEditor(
        task_manager=ui_task_mgr, context=ui_context_initializer
    )
    yield editor
    editor.cleanup()


@pytest.mark.ui
def test_scan_error_hides_switch_to_trace_hint(editor, tmp_path, monkeypatch):
    monkeypatch.setattr(ghostscript, "find_ghostscript", lambda: None)
    ghostscript.clear_cache()
    path = tmp_path / "logo.eps"
    path.write_bytes(EPS)

    dialog = ImportDialog(Gtk.Window(), editor, path, "image/x-eps", FEATURES)
    dialog._update_ui_with_preview(None)

    assert dialog.error_banner.get_revealed()
    assert "Ghostscript" in dialog.error_banner.get_title()
    assert not dialog.warning_banner.get_revealed()
    dialog.destroy()

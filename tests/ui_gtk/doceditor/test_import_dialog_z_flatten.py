# flake8: noqa: E402
import io

import ezdxf
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import Gtk

from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.doceditor.editor import DocEditor
from rayforge.image.base_importer import ImporterFeature
from rayforge.image.dxf.importer import DxfImporter
from rayforge.ui_gtk.doceditor.import_dialog import ImportDialog

# The DXF importer's real feature set: notably no BITMAP_TRACING, so
# the "Import Mode" group stays hidden for DXF files.
FEATURES = DxfImporter.features


@pytest.fixture
def editor(ui_context_initializer, ui_task_mgr):
    editor = DocEditor(
        task_manager=ui_task_mgr, context=ui_context_initializer
    )
    yield editor
    editor.cleanup()


def _dxf_bytes(z_variation: bool) -> bytes:
    doc = ezdxf.new()  # type: ignore
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    msp.add_line((0, 0, 0), (100, 0, 0))
    if z_variation:
        msp.add_line((0, 0, -10), (100, 0, -10))
    buffer = io.StringIO()
    doc.write(buffer)
    return buffer.getvalue().encode("utf-8")


def _make_dialog(editor, tmp_path, z_variation: bool) -> ImportDialog:
    path = tmp_path / "part.dxf"
    path.write_bytes(_dxf_bytes(z_variation))
    dialog = ImportDialog(
        Gtk.Window(), editor, path, "image/vnd.dxf", FEATURES
    )
    return dialog


@pytest.mark.ui
def test_flatten_z_row_visible_and_default_on(editor, tmp_path):
    dialog = _make_dialog(editor, tmp_path, z_variation=True)

    assert dialog.z_flatten_group.get_visible()
    assert dialog.flatten_z_switch.get_active()
    assert not dialog.z_flatten_banner.get_revealed()
    dialog.destroy()


@pytest.mark.ui
def test_flatten_z_warning_revealed_when_disabled(editor, tmp_path):
    dialog = _make_dialog(editor, tmp_path, z_variation=True)

    dialog.flatten_z_switch.set_active(False)
    assert dialog.z_flatten_banner.get_revealed()

    dialog.flatten_z_switch.set_active(True)
    assert not dialog.z_flatten_banner.get_revealed()
    dialog.destroy()


@pytest.mark.ui
def test_flatten_z_row_hidden_for_flat_files(editor, tmp_path):
    dialog = _make_dialog(editor, tmp_path, z_variation=False)

    assert not dialog.z_flatten_group.get_visible()
    assert not dialog.z_flatten_banner.get_revealed()
    dialog.destroy()


@pytest.mark.ui
def test_flatten_z_row_hidden_without_feature(editor, tmp_path):
    path = tmp_path / "part.dxf"
    path.write_bytes(_dxf_bytes(z_variation=True))
    features = FEATURES - {ImporterFeature.Z_FLATTENING}

    dialog = ImportDialog(
        Gtk.Window(), editor, path, "image/vnd.dxf", features
    )

    assert not dialog.z_flatten_group.get_visible()
    dialog.destroy()


@pytest.mark.ui
def test_spec_carries_flatten_z_state(editor, tmp_path):
    dialog = _make_dialog(editor, tmp_path, z_variation=True)
    dialog.use_vectors_switch.set_active(True)

    dialog.flatten_z_switch.set_active(False)
    spec = dialog._get_current_spec()
    assert isinstance(spec, PassthroughSpec)
    assert spec.flatten_z is False

    dialog.flatten_z_switch.set_active(True)
    spec = dialog._get_current_spec()
    assert isinstance(spec, PassthroughSpec)
    assert spec.flatten_z is True
    dialog.destroy()

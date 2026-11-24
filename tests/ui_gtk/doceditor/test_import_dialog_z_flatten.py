# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import Gtk
from raygeo.formats.dxf.writer import write
from raygeo.geo import Geometry

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
    geometry = Geometry()
    geometry.move_to(0, 0, 0)
    geometry.line_to(100, 0, 0)
    if z_variation:
        geometry.move_to(0, 0, -10)
        geometry.line_to(100, 0, -10)
    return write(geometry).encode("utf-8")


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


@pytest.mark.ui
def test_repair_group_shown_for_capable_importers(editor, tmp_path):
    dialog = _make_dialog(editor, tmp_path, z_variation=False)

    assert dialog.repair_group.get_visible()
    assert dialog.repair_switch.get_active() is False
    assert not dialog.repair_tolerance_row.get_sensitive()
    dialog.destroy()


@pytest.mark.ui
def test_repair_group_hidden_without_feature(editor, tmp_path):
    path = tmp_path / "part.dxf"
    path.write_bytes(_dxf_bytes(z_variation=False))
    features = FEATURES - {ImporterFeature.GEOMETRY_REPAIR}

    dialog = ImportDialog(
        Gtk.Window(), editor, path, "image/vnd.dxf", features
    )

    assert not dialog.repair_group.get_visible()
    dialog.destroy()


@pytest.mark.ui
def test_repair_tolerance_follows_switch(editor, tmp_path):
    dialog = _make_dialog(editor, tmp_path, z_variation=False)

    dialog.repair_switch.set_active(True)
    assert dialog.repair_tolerance_row.get_sensitive()

    dialog.repair_switch.set_active(False)
    assert not dialog.repair_tolerance_row.get_sensitive()
    dialog.destroy()


@pytest.mark.ui
def test_spec_carries_repair_state(editor, tmp_path):
    dialog = _make_dialog(editor, tmp_path, z_variation=False)
    dialog.use_vectors_switch.set_active(True)

    dialog.repair_switch.set_active(True)
    spec = dialog._get_current_spec()
    assert isinstance(spec, PassthroughSpec)
    assert spec.repair_outlines is True
    assert spec.repair_tolerance_mm is None

    dialog.repair_tolerance_row.set_value(0.25)
    spec = dialog._get_current_spec()
    assert isinstance(spec, PassthroughSpec)
    assert spec.repair_tolerance_mm == pytest.approx(0.25)
    dialog.destroy()

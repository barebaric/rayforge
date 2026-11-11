# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
import yaml
from gi.repository import Gtk

from rayforge import config as config_module
from rayforge.core.vectorization_spec import LayerSource, PassthroughSpec
from rayforge.doceditor.editor import DocEditor
from rayforge.image.svg.importer import SvgImporter
from rayforge.ui_gtk.doceditor.import_dialog import ImportDialog

FEATURES = SvgImporter.features

# An SVG with distinct stroke colors but no layer structure: this is
# the shape of file that makes the dialog default to color grouping.
SVG = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'width="100mm" height="50mm" viewBox="0 0 100 50">'
    '<path d="M 10,10 L 40,10" stroke="#ff0000" fill="none" '
    'stroke-width="2"/>'
    '<path d="M 10,30 L 40,30" stroke="#0000ff" fill="none" '
    'stroke-width="2"/>'
    "</svg>"
)


@pytest.fixture
def editor(ui_context_initializer, ui_task_mgr):
    editor = DocEditor(
        task_manager=ui_task_mgr, context=ui_context_initializer
    )
    yield editor
    editor.cleanup()


def _make_dialog(editor, tmp_path) -> ImportDialog:
    path = tmp_path / "colors.svg"
    path.write_text(SVG, encoding="utf-8")
    return ImportDialog(Gtk.Window(), editor, path, "image/svg+xml", FEATURES)


def _selected_source(dialog: ImportDialog) -> LayerSource:
    return (
        LayerSource.COLORS
        if dialog.layer_source_row.get_selected() == 1
        else LayerSource.SVG_LAYERS
    )


@pytest.mark.ui
def test_defaults_to_colors_without_preference(editor, tmp_path):
    dialog = _make_dialog(editor, tmp_path)

    assert _selected_source(dialog) == LayerSource.COLORS
    dialog.destroy()


@pytest.mark.ui
def test_remembered_choice_wins_over_heuristic(
    editor, ui_context_initializer, tmp_path
):
    ui_context_initializer.config.set_import_layer_source(
        LayerSource.SVG_LAYERS
    )
    dialog = _make_dialog(editor, tmp_path)

    assert _selected_source(dialog) == LayerSource.SVG_LAYERS
    dialog.destroy()


@pytest.mark.ui
def test_remembered_colors_choice_is_kept(
    editor, ui_context_initializer, tmp_path
):
    ui_context_initializer.config.set_import_layer_source(LayerSource.COLORS)
    dialog = _make_dialog(editor, tmp_path)

    assert _selected_source(dialog) == LayerSource.COLORS
    dialog.destroy()


@pytest.mark.ui
def test_user_choice_is_remembered(editor, ui_context_initializer, tmp_path):
    dialog = _make_dialog(editor, tmp_path)
    assert _selected_source(dialog) == LayerSource.COLORS

    dialog.layer_source_row.set_selected(0)

    assert ui_context_initializer.config.import_layer_source == (
        LayerSource.SVG_LAYERS
    )
    dialog.destroy()


@pytest.mark.ui
def test_user_choice_is_persisted_to_disk(
    editor, ui_context_initializer, tmp_path
):
    dialog = _make_dialog(editor, tmp_path)
    dialog.layer_source_row.set_selected(0)
    dialog.destroy()

    data = yaml.safe_load(config_module.CONFIG_FILE.read_text())
    assert data["import_layer_source"] == "svg_layers"


@pytest.mark.ui
def test_reimport_spec_does_not_change_remembered_choice(
    editor, ui_context_initializer, tmp_path
):
    dialog = _make_dialog(editor, tmp_path)
    spec = PassthroughSpec(layer_source=LayerSource.SVG_LAYERS)

    dialog._apply_spec_to_widgets(spec)

    assert _selected_source(dialog) == LayerSource.SVG_LAYERS
    assert ui_context_initializer.config.import_layer_source is None
    dialog.destroy()

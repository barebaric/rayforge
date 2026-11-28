"""UI tests for the layer settings dialog."""

import pytest

from rayforge.core.layer import Layer
from rayforge.ui_gtk.doceditor.layer_settings_dialog import (
    LayerSettingsDialog,
)


def _make_dialog():
    layer = Layer("Layer")
    dialog = LayerSettingsDialog(layer=layer, transient_for=None, editor=None)
    return dialog, layer


@pytest.mark.ui
def test_shows_layer_settings(ui_context_initializer):
    dialog, layer = _make_dialog()

    assert dialog.name_row.get_text() == layer.name
    assert dialog.visible_row.get_active() == layer.visible
    assert dialog.rotary_enabled_row.get_active() == layer.rotary_enabled
    dialog.destroy()


@pytest.mark.ui
def test_edits_name_visible_and_rotary(ui_context_initializer):
    dialog, layer = _make_dialog()

    dialog.name_row.set_text("Renamed")
    assert layer.name == "Renamed"

    dialog.visible_row.set_active(False)
    assert layer.visible is False

    dialog.rotary_enabled_row.set_active(True)
    assert layer.rotary_enabled is True
    dialog.destroy()

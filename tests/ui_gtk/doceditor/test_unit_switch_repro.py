# flake8: noqa: E402
"""Reproduction: does switching length units mutate the item model?

Switch user units mm -> in and verify a selected item's size/pos don't
change in the model.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from raygeo.geo import Matrix

from rayforge.context import get_context
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.editor import DocEditor
from rayforge.ui_gtk.doceditor.property_providers.transform import (
    TransformPropertyProvider,
)


@pytest.fixture
def editor_with_provider(ui_context_initializer, ui_task_mgr):
    editor = DocEditor(
        task_manager=ui_task_mgr, context=ui_context_initializer
    )
    yield editor
    editor.cleanup()


@pytest.mark.ui
def test_unit_switch_keeps_model_intact(editor_with_provider):
    editor = editor_with_provider
    context = get_context()

    wp = WorkPiece(name="wp.svg")
    wp.matrix = Matrix.scale(100, 50)
    editor.doc.active_layer.add_child(wp)

    provider = TransformPropertyProvider()
    provider.create_widgets()
    provider.update_widgets(editor, [wp])

    before_size = wp.size
    before_pos = wp.pos

    context.config.unit_preferences["length"] = "in"
    context.config.changed.send(context.config)

    assert wp.size == pytest.approx(before_size)
    assert wp.pos == pytest.approx(before_pos)

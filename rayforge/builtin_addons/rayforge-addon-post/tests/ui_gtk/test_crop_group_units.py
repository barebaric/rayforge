"""UI tests: post-processor transformer settings groups show units.

The crop offset (and other length rows) must display and convert in the
user's length unit via LengthSpinRow.
"""

# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from typing import cast

import pytest
from blinker import Signal
from gi.repository import Adw
from post_processors.transformers import CropTransformer
from post_processors.widgets.crop_group import CropSettingsGroup

from rayforge.core.step import Step
from rayforge.doceditor.editor import DocEditor


class _Page:
    use_expanders = True


class _StepStub:
    def __init__(self):
        self.per_workpiece_transformers_dicts = []
        self.per_step_transformers_dicts = []
        self.updated = Signal()


class _EditorStub:
    def __init__(self):
        self.history_manager = None
        self.step = object()


def _build_group(ui_context):
    step = _StepStub()
    transformer = CropTransformer(offset=25.4)
    step.per_step_transformers_dicts = [transformer.to_dict()]
    editor = _EditorStub()
    group = CropSettingsGroup(
        cast(DocEditor, editor),
        "Crop",
        transformer,
        cast(Adw.PreferencesPage, _Page()),
        cast(Step, step),
    )
    return group, transformer


@pytest.mark.ui
def test_crop_offset_shows_mm_by_default(ui_context):
    group, transformer = _build_group(ui_context)

    assert group.offset_row.get_value_in_base_units() == pytest.approx(25.4)
    assert group.offset_row.get_value() == pytest.approx(25.4, abs=1e-2)


@pytest.mark.ui
def test_crop_offset_shows_inches_when_imperial(ui_context):
    ui_context.config.unit_preferences["length"] = "in"
    group, transformer = _build_group(ui_context)

    assert group.offset_row.get_value_in_base_units() == pytest.approx(25.4)
    assert group.offset_row.get_value() == pytest.approx(1.0, abs=1e-2)

# flake8: noqa: E402
"""UI tests: post-processor transformer settings groups show units.

The crop offset (and other length rows) must display and convert in the
user's length unit via LengthSpinRow.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from post_processors.transformers import CropTransformer
from post_processors.widgets.crop_group import CropSettingsGroup


class _Page:
    use_expanders = True


def _build_group(ui_context):
    transformer = CropTransformer(offset=25.4)
    group = CropSettingsGroup("Crop", transformer, _Page())
    return group, transformer


@pytest.mark.ui
def test_crop_offset_shows_mm_by_default(ui_context):
    group, _transformer = _build_group(ui_context)

    assert group.offset_row.get_value_in_base_units() == pytest.approx(25.4)
    assert group.offset_row.get_value() == pytest.approx(25.4, abs=1e-2)


@pytest.mark.ui
def test_crop_offset_shows_inches_when_imperial(ui_context):
    ui_context.config.unit_preferences["length"] = "in"
    group, _transformer = _build_group(ui_context)

    assert group.offset_row.get_value_in_base_units() == pytest.approx(25.4)
    assert group.offset_row.get_value() == pytest.approx(1.0, abs=1e-2)

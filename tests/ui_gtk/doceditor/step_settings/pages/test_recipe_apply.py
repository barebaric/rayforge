# flake8: noqa: E402
"""Tests for applying a recipe and syncing the settings page widgets."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.core.recipe import Recipe
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.doceditor.step_settings.rows import SliderRow, SpinRow


@pytest.mark.ui
def test_applying_recipe_updates_page_widgets(editor, step):
    page = StepSettingsPage(editor, step)
    count_row = SpinRow(
        editor, step, "count", "Count", None, 1, 10, 1, 0, is_int=True
    )
    power_row = SliderRow(
        editor, step, "power", "Power", None, 0.0, 1.0, 0.01, 2
    )
    page.add_section("Params", count_row, power_row)

    recipe = Recipe(
        name="Test Recipe",
        setting_dicts=[
            {"name": "count", "value": 9, "recipe_apply": True},
            {"name": "power", "value": 0.8, "recipe_apply": True},
        ],
    )

    updated = []
    step.updated.connect(lambda *_: updated.append(1), weak=False)
    page.recipe_control._apply_recipe(recipe)

    assert step.applied_recipe_uid == recipe.uid
    assert step.count == 9
    assert step.power == pytest.approx(0.8)
    assert updated, "applying a recipe must emit step.updated"
    assert count_row.widget.get_value() == 9
    assert power_row._adj.get_value() == pytest.approx(0.8)


@pytest.mark.ui
def test_applying_recipe_uses_setters(editor, step):
    page = StepSettingsPage(editor, step)
    recipe = Recipe(
        name="Setter Recipe",
        setting_dicts=[{"name": "count", "value": 5, "recipe_apply": True}],
    )

    page.recipe_control._apply_recipe(recipe)

    assert step.count == 5


@pytest.mark.ui
def test_resync_overrides_pending_edit(editor, step):
    row = SpinRow(
        editor, step, "count", "Count", None, 1, 10, 1, 0, is_int=True
    )
    row.widget.get_adjustment().set_value(9)
    assert row._debounce_timer != 0

    step.count = 4
    row.resync()

    assert row.widget.get_value() == 4
    assert row._debounce_timer == 0

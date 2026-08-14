# flake8: noqa: E402
"""Tests for applying a recipe and syncing the settings page widgets."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.core.recipe import Recipe
from rayforge.core.varset import FloatVar, IntVar, VarSet
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.varset.varsetwidget import VarSetWidget


def _make_page(editor, step) -> tuple[StepSettingsPage, VarSetWidget]:
    page = StepSettingsPage(editor, step)
    var_set = VarSet(
        vars=[
            IntVar(key="count", label="Count", default=3, min_val=1),
            FloatVar(
                key="power",
                label="Power",
                default=0.5,
                min_val=0.0,
                max_val=1.0,
                digits=2,
            ),
        ]
    )
    widget = page.add_varset_section("Params", var_set)
    return page, widget


@pytest.mark.ui
def test_applying_recipe_updates_page_widgets(editor, step):
    page, widget = _make_page(editor, step)

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
    assert widget.get_values()["count"] == 9
    assert widget.get_values()["power"] == pytest.approx(0.8)


@pytest.mark.ui
def test_applying_recipe_uses_setters(editor, step):
    page, _widget = _make_page(editor, step)
    recipe = Recipe(
        name="Setter Recipe",
        setting_dicts=[{"name": "count", "value": 5, "recipe_apply": True}],
    )

    page.recipe_control._apply_recipe(recipe)

    assert step.count == 5


@pytest.mark.ui
def test_resync_overrides_pending_edit(editor, step):
    """Applying a recipe overrides a pending (debounced) user edit."""
    page, widget = _make_page(editor, step)

    count_adapter = widget.adapter_for("count")
    assert count_adapter is not None
    count_adapter.set_value(9)
    widget._on_data_changed("count")
    assert widget._pending_keys

    recipe = Recipe(
        name="Override Recipe",
        setting_dicts=[{"name": "count", "value": 4, "recipe_apply": True}],
    )
    page.recipe_control._apply_recipe(recipe)

    assert widget.get_values()["count"] == 4
    assert not widget._pending_keys


def test_sync_from_model_preserves_pending_edit(editor, step):
    """A model resync skips keys with pending (debounced) edits."""
    _page, widget = _make_page(editor, step)

    count_adapter = widget.adapter_for("count")
    assert count_adapter is not None
    count_adapter.set_value(9)
    widget._on_data_changed("count")

    widget.sync_from_model({"count": 4, "power": 0.5})

    assert widget.get_values()["count"] == 9
    assert widget.get_values()["power"] == pytest.approx(0.5)

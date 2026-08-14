# flake8: noqa: E402
"""Integration tests for AddEditRecipeDialog step-type targeting."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.core.recipe import Recipe
from rayforge.machine.models.laser import Laser
from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.doceditor.recipes.edit_recipe_dialog import (
    AddEditRecipeDialog,
)

pytestmark = pytest.mark.ui


@pytest.fixture
def laser_machine(ui_context_initializer):
    """A machine with one laser head, set as the active machine."""
    context = ui_context_initializer
    machine = Machine(context)
    machine.set_axis_extents(200, 150)
    machine.max_cut_speed = 5000
    machine.max_travel_speed = 10000

    laser = Laser()
    laser.name = "Laser 1"
    laser.spot_size_mm = (0.1, 0.2)
    machine.heads.clear()
    machine.add_head(laser)

    context.machine_mgr.machines.clear()
    context.machine_mgr.add_machine(machine)
    context.config.set_machine(machine)
    return machine


def _settings_keys(dialog):
    """Collect widget keys across all settings pages."""
    keys = []
    for page in dialog._settings_pages.values():
        keys.extend(page.keys)
    return keys


def test_default_selection_is_generic(laser_machine):
    """Opening the editor with no recipe defaults to a generic (any step)
    selection and base Step settings."""
    dialog = AddEditRecipeDialog(parent=None, recipe=None)
    page = dialog.applicability_page

    assert page.get_step_types() == []

    keys = _settings_keys(dialog)
    assert "cut_speed" in keys
    assert "travel_speed" in keys
    assert "power" not in keys
    assert "cut_side" not in keys

    data = dialog.get_recipe_data()
    assert data["target_step_types"] == []

    dialog.close()


def test_single_step_type_shows_step_specific_settings(laser_machine):
    """Selecting one step type shows that step's full settings groups."""
    recipe = Recipe(
        name="Contour Recipe",
        target_step_types=["ContourStep"],
        setting_dicts=[
            {"name": "power", "value": 0.8, "recipe_apply": True},
            {"name": "cut_side", "value": "OUTSIDE", "recipe_apply": True},
        ],
    )
    dialog = AddEditRecipeDialog(parent=None, recipe=recipe)

    # Two settings pages: "Laser" and "Step Settings".
    titles = [p.group_title for p in dialog._settings_pages.values()]
    assert any("Laser" in t for t in titles)
    assert any("Step Settings" in t for t in titles)

    all_keys = set(_settings_keys(dialog))
    assert "cut_side" in all_keys
    assert "cut_order" in all_keys
    assert "power" in all_keys

    data = dialog.get_recipe_data()
    assert data["target_step_types"] == ["ContourStep"]
    assert any(
        d["name"] == "power" and d["value"] == 0.8
        for d in data["setting_dicts"]
    )
    assert any(
        d["name"] == "cut_side" and d["value"] == "OUTSIDE"
        for d in data["setting_dicts"]
    )

    dialog.close()


def test_multi_step_types_show_common_settings(laser_machine):
    """Multiple step types show only the settings common to all of them.

    ContourStep and EngraveStep only share the inherited Laser group;
    their step-specific settings (cut_side / scan_angle) are excluded.
    """
    recipe = Recipe(
        name="Multi",
        target_step_types=["ContourStep", "EngraveStep"],
    )
    dialog = AddEditRecipeDialog(parent=None, recipe=recipe)

    keys = _settings_keys(dialog)
    assert "cut_speed" in keys
    assert "travel_speed" in keys
    assert "power" in keys
    assert "cut_side" not in keys  # ContourStep-specific
    assert "scan_angle" not in keys  # EngraveStep-specific

    data = dialog.get_recipe_data()
    assert data["target_step_types"] == ["ContourStep", "EngraveStep"]

    dialog.close()


def test_selecting_single_step_type_rebuilds_settings(laser_machine):
    """Changing the selection to a single step type rebuilds the settings."""
    dialog = AddEditRecipeDialog(parent=None, recipe=None)
    page = dialog.applicability_page

    # Simulate the StepTypeSelectionDialog callback.
    page._on_step_types_selected(["ContourStep"])

    assert page.get_step_types() == ["ContourStep"]
    all_keys = set(_settings_keys(dialog))
    assert "cut_side" in all_keys
    assert "power" in all_keys

    dialog.close()


def test_selecting_multi_step_types_shows_common(laser_machine):
    """Changing to multiple step types shows the common settings only."""
    dialog = AddEditRecipeDialog(parent=None, recipe=None)
    page = dialog.applicability_page

    page._on_step_types_selected(["ContourStep", "EngraveStep"])

    keys = _settings_keys(dialog)
    assert "power" in keys  # shared laser setting
    assert "cut_side" not in keys
    assert "scan_angle" not in keys

    dialog.close()


# --- POST-PROCESSING TAB ---


def test_post_processing_tab_hidden_for_generic_selection(laser_machine):
    """The base Step has no transformers, so no tab appears."""
    dialog = AddEditRecipeDialog(parent=None, recipe=None)
    assert dialog._post_processing_page is None
    assert "post-processing" not in dialog._tab_buttons
    assert "transformer_dicts" in dialog.get_recipe_data()
    assert dialog.get_recipe_data()["transformer_dicts"] == []
    dialog.close()


def test_post_processing_tab_appears_for_contour(laser_machine):
    """ContourStep has transformers, so the tab appears with them."""
    recipe = Recipe(name="Contour", target_step_types=["ContourStep"])
    dialog = AddEditRecipeDialog(parent=None, recipe=recipe)

    assert dialog._post_processing_page is not None
    assert "post-processing" in dialog._tab_buttons
    names = [
        d.get("name")
        for d in dialog._post_processing_page.get_transformer_dicts()
    ]
    assert "Optimize" in names
    assert "CropTransformer" in names

    # All dicts default to "Toggle off" (recipe_apply=False).
    assert all(
        not d.get("recipe_apply")
        for d in dialog._post_processing_page.get_transformer_dicts()
    )

    dialog.close()


def test_post_processing_tab_restores_recipe_values(laser_machine):
    """Stored recipe values are overlaid onto the common transformers."""
    recipe = Recipe(
        name="Contour",
        target_step_types=["ContourStep"],
        transformer_dicts=[
            {
                "name": "Optimize",
                "recipe_apply": True,
                "enabled": False,
            }
        ],
    )
    dialog = AddEditRecipeDialog(parent=None, recipe=recipe)
    assert dialog._post_processing_page is not None

    dicts = {
        d.get("name"): d
        for d in dialog._post_processing_page.get_transformer_dicts()
    }
    assert dicts["Optimize"]["recipe_apply"] is True
    assert dicts["Optimize"]["enabled"] is False
    # Other transformers default to Toggle off.
    assert dicts["CropTransformer"]["recipe_apply"] is False

    data = dialog.get_recipe_data()
    data_dicts = {d.get("name"): d for d in data["transformer_dicts"]}
    assert data_dicts["Optimize"]["recipe_apply"] is True
    assert data_dicts["Optimize"]["enabled"] is False

    dialog.close()


def test_apply_toggle_and_enable_switch_update_dict(laser_machine):
    """Flipping the apply toggle writes recipe_apply; the group's native
    enable switch writes enabled."""
    recipe = Recipe(name="Contour", target_step_types=["ContourStep"])
    dialog = AddEditRecipeDialog(parent=None, recipe=recipe)
    assert dialog._post_processing_page is not None
    page = dialog._post_processing_page

    group = _group_for_transformer(page, "Optimize")
    dicts = {d.get("name"): d for d in page.get_transformer_dicts()}
    optimize_dict = dicts["Optimize"]
    assert optimize_dict["recipe_apply"] is False

    # Flip the apply toggle on.
    _set_apply_toggle(group, True)
    assert optimize_dict["recipe_apply"] is True

    # Flip the group's native enable switch off.
    enable_switch = group.enable_switch
    assert enable_switch is not None
    enable_switch.set_active(False)
    assert optimize_dict["enabled"] is False

    # Flip the apply toggle off again.
    _set_apply_toggle(group, False)
    assert optimize_dict["recipe_apply"] is False

    dialog.close()


def test_post_processing_tab_rebuilds_with_selection(laser_machine):
    """Changing the step-type selection rebuilds the post-processing tab."""
    dialog = AddEditRecipeDialog(parent=None, recipe=None)
    assert dialog._post_processing_page is None

    dialog.applicability_page._on_step_types_selected(["ContourStep"])
    assert dialog._post_processing_page is not None

    dialog.applicability_page._on_step_types_selected(["EngraveStep"])
    assert dialog._post_processing_page is not None

    dialog.applicability_page._on_step_types_selected([])
    assert dialog._post_processing_page is None

    dialog.close()


def _group_for_transformer(page, name):
    """The settings group backing the named transformer, if any."""
    for group, t_dict in page._group_dicts.items():
        if t_dict.get("name") == name:
            return group
    raise AssertionError(f"No group for transformer '{name}'")


def _set_apply_toggle(group, active):
    """Flip the group's apply toggle, mirroring a click."""
    toggle = group.apply_toggle
    assert toggle is not None
    toggle.set_active(active)

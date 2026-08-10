"""Integration tests for AddEditRecipeDialog step-type targeting."""

import pytest

from rayforge.core.recipe import Recipe
from rayforge.machine.models.laser import Laser
from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.doceditor.edit_recipe_dialog import AddEditRecipeDialog

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
        settings={"power": 0.8, "cut_side": "OUTSIDE"},
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
    assert data["settings"]["power"] == 0.8
    assert data["settings"]["cut_side"] == "OUTSIDE"

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

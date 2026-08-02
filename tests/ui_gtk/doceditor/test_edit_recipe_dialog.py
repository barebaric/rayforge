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


def _capability_index(dialog, cap_name):
    # Capabilities are offset by 1 in the row because index 0 is "Any".
    page = dialog.applicability_page
    for i, cap in enumerate(page._ui_capabilities):
        if cap.name == cap_name:
            return i + 1
    raise AssertionError(f"Capability {cap_name} not found")


def _settings_keys(dialog):
    """Collect widget keys across all settings pages."""
    keys = []
    for page in dialog._settings_pages.values():
        keys.extend(page.keys)
    return keys


def test_default_task_type_is_any_shows_all_steps(laser_machine):
    """Opening the editor defaults to 'Any' task type, listing all steps."""
    dialog = AddEditRecipeDialog(parent=None, recipe=None)
    page = dialog.applicability_page

    # Default task type is "Any" (index 0).
    assert page.capability_row.get_selected() == 0
    assert page.get_capability() is None

    # All non-hidden step types are visible, including EngraveStep.
    step_types = page._ui_step_types
    assert step_types[0] is None  # "Any Type"
    assert "EngraveStep" in step_types
    assert "ContourStep" in step_types

    dialog.close()


def test_step_type_dropdown_filters_by_capability(laser_machine):
    """Selecting a specific task type filters the step list to it."""
    dialog = AddEditRecipeDialog(parent=None, recipe=None)
    page = dialog.applicability_page

    page.capability_row.set_selected(_capability_index(dialog, "CUT"))

    step_types = page._ui_step_types
    assert step_types[0] is None  # "Any Type"
    assert "ContourStep" in step_types
    assert "FrameStep" in step_types
    # EngraveStep is ENGRAVE-only, so not under CUT.
    assert "EngraveStep" not in step_types

    dialog.close()


def test_any_task_type_shows_base_step_settings(laser_machine):
    """With 'Any' task type and no step type, base Step settings appear."""
    recipe = Recipe(name="Generic", target_capability_name="")
    dialog = AddEditRecipeDialog(parent=None, recipe=recipe)

    keys = _settings_keys(dialog)
    assert "cut_speed" in keys
    assert "travel_speed" in keys
    assert "power" not in keys
    assert "cut_side" not in keys

    data = dialog.get_recipe_data()
    assert data["target_step_type"] is None
    assert data["target_capability_name"] == ""

    dialog.close()


def test_capability_task_type_uses_capability_varset(laser_machine):
    """A specific task type with no step type uses the capability varset."""
    recipe = Recipe(name="Generic Cut", target_capability_name="CUT")
    dialog = AddEditRecipeDialog(parent=None, recipe=recipe)

    keys = _settings_keys(dialog)
    # The capability varset has no step-specific keys like cut_side.
    assert "cut_side" not in keys
    assert "power" in keys

    data = dialog.get_recipe_data()
    assert data["target_step_type"] is None
    assert data["target_capability_name"] == "CUT"

    dialog.close()


def test_step_type_splits_into_laser_and_step_pages(laser_machine):
    """Selecting ContourStep splits settings into Laser + Step Settings."""
    recipe = Recipe(
        name="Contour Recipe",
        target_capability_name="CUT",
        target_step_type="ContourStep",
        settings={"power": 0.8, "cut_side": "OUTSIDE"},
    )
    dialog = AddEditRecipeDialog(parent=None, recipe=recipe)

    # Two settings pages: "Laser" and "Step Settings".
    titles = [p.group_title for p in dialog._settings_pages.values()]
    assert any("Laser" in t for t in titles)
    assert any("Step Settings" in t for t in titles)

    # cut_side lives on the Step Settings page; power on the Laser page.
    all_keys = set(_settings_keys(dialog))
    assert "cut_side" in all_keys
    assert "cut_order" in all_keys
    assert "power" in all_keys

    data = dialog.get_recipe_data()
    assert data["target_step_type"] == "ContourStep"
    assert data["settings"]["power"] == 0.8
    assert data["settings"]["cut_side"] == "OUTSIDE"

    dialog.close()


def test_changing_capability_resets_step_type(laser_machine):
    """Changing capability resets the step type to 'Any Type'."""
    recipe = Recipe(
        name="Contour Recipe",
        target_capability_name="CUT",
        target_step_type="ContourStep",
    )
    dialog = AddEditRecipeDialog(parent=None, recipe=recipe)
    page = dialog.applicability_page
    assert page.step_type_row.get_selected() != 0  # a step type selected

    # Switch to ENGRAVE capability.
    page.capability_row.set_selected(_capability_index(dialog, "ENGRAVE"))
    # Step type should reset to "Any Type" (index 0).
    assert page.step_type_row.get_selected() == 0

    dialog.close()

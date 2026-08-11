"""Tests for the recipe list row subtitle."""

import pytest

from rayforge.core.recipe import Recipe
from rayforge.ui_gtk.doceditor.recipes.recipe_list import RecipeRow

pytestmark = pytest.mark.ui


def _row_for(recipe: Recipe) -> RecipeRow:
    return RecipeRow(
        recipe,
        on_delete=lambda recipe: None,
        on_edit=lambda recipe: None,
    )


def test_step_types_shown(ui_context_initializer):
    """A step-scoped recipe shows its step type, not 'Any'."""
    recipe = Recipe(
        name="Contour Only",
        target_step_types=["ContourStep"],
    )
    subtitle = _row_for(recipe)._get_subtitle()
    assert "Contour" in subtitle
    assert "Any" not in subtitle


def test_multiple_step_types_joined(ui_context_initializer):
    """Multiple step types are joined in the subtitle."""
    recipe = Recipe(
        name="Multi",
        target_step_types=["ContourStep", "FrameStep"],
    )
    subtitle = _row_for(recipe)._get_subtitle()
    assert "Contour" in subtitle
    assert "Frame" in subtitle


def test_generic_recipe_shows_any(ui_context_initializer):
    """A generic recipe (no step types) shows 'Any'."""
    recipe = Recipe(name="Generic", target_step_types=[])
    subtitle = _row_for(recipe)._get_subtitle()
    assert subtitle == "Any"

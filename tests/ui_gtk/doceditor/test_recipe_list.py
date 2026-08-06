"""Tests for the recipe list row subtitle."""

import pytest

from rayforge.core.recipe import Recipe
from rayforge.ui_gtk.doceditor.recipe_list import RecipeRow

pytestmark = pytest.mark.ui


def _row_for(recipe: Recipe) -> RecipeRow:
    return RecipeRow(
        recipe,
        on_delete=lambda recipe: None,
        on_edit=lambda recipe: None,
    )


def test_step_type_shown_instead_of_any(ui_context_initializer):
    """A step-scoped recipe shows its step type, not 'Any'."""
    recipe = Recipe(
        name="Contour Only",
        target_capability_name="",
        target_step_type="ContourStep",
    )
    subtitle = _row_for(recipe)._get_subtitle()
    assert "Contour" in subtitle
    assert "Any" not in subtitle


def test_capability_label_when_no_step_type(ui_context_initializer):
    """Without a step type, the capability label is shown."""
    recipe = Recipe(
        name="Generic Cut",
        target_capability_name="CUT",
        target_step_type=None,
    )
    subtitle = _row_for(recipe)._get_subtitle()
    assert subtitle == "Cut"


def test_capability_and_step_type_shows_step_type(ui_context_initializer):
    """A recipe with both criteria shows the more specific step type."""
    recipe = Recipe(
        name="Contour",
        target_capability_name="CUT",
        target_step_type="ContourStep",
    )
    subtitle = _row_for(recipe)._get_subtitle()
    assert "Contour" in subtitle
    assert "Any" not in subtitle

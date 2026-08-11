"""Dedicated page widgets for the recipe editor dialog."""

from .applicability import RecipeApplicabilityPage
from .general import RecipeGeneralPage
from .post_processing import RecipePostProcessingPage
from .settings import RecipeSettingsPage

__all__ = [
    "RecipeApplicabilityPage",
    "RecipeGeneralPage",
    "RecipePostProcessingPage",
    "RecipeSettingsPage",
]

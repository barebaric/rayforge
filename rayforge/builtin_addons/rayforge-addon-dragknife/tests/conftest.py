"""
Pytest configuration for the dragknife builtin addon tests.

This conftest ensures that steps, transformers, and settings pages
are registered before tests run, mirroring the cnc_essentials addon
conftest.
"""

import pytest
from dragknife.frontend import register_step_settings_pages
from dragknife.steps import DragKnifeCutStep, TangentialKnifeCutStep
from dragknife.transformers import (
    DragKnifeTransformer,
    TangentialKnifeTransformer,
)

from rayforge.core.step_registry import step_registry
from rayforge.pipeline.transformer.registry import transformer_registry
from rayforge.ui_gtk.doceditor.step_settings.page_registry import (
    step_settings_page_registry,
)


def _register_addon():
    """Register all dragknife addon components."""
    step_registry.register(DragKnifeCutStep, addon_name="dragknife")
    step_registry.register(TangentialKnifeCutStep, addon_name="dragknife")
    transformer_registry.register(DragKnifeTransformer, addon_name="dragknife")
    transformer_registry.register(
        TangentialKnifeTransformer, addon_name="dragknife"
    )
    register_step_settings_pages(step_settings_page_registry)


@pytest.fixture(scope="session", autouse=True)
def register_dragknife():
    """
    Automatically register the dragknife addon components for all
    tests in this addon.
    """
    _register_addon()
    yield

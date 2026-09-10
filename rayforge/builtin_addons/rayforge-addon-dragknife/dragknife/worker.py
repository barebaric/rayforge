"""
Backend entry point for the dragknife addon.

Registers steps and transformers with the main application.
"""

from rayforge.core.hooks import hookimpl

from .steps import DragKnifeCutStep, TangentialKnifeCutStep
from .transformers import DragKnifeTransformer, TangentialKnifeTransformer

ADDON_NAME = "dragknife"


@hookimpl
def register_steps(step_registry):
    """Register knife steps with the step registry."""
    step_registry.register(DragKnifeCutStep, addon_name=ADDON_NAME)
    step_registry.register(TangentialKnifeCutStep, addon_name=ADDON_NAME)


@hookimpl
def register_transformers(transformer_registry):
    """Register the knife compensation transformers."""
    transformer_registry.register(DragKnifeTransformer, addon_name=ADDON_NAME)
    transformer_registry.register(
        TangentialKnifeTransformer, addon_name=ADDON_NAME
    )

"""
Backend entry point for the dragknife addon.

Registers steps and transformers with the main application.
"""

import logging

from rayforge.core.hooks import hookimpl

from .compat import KNIFE_TRANSFORMS_AVAILABLE
from .steps import DragKnifeCutStep, TangentialKnifeCutStep
from .transformers import DragKnifeTransformer, TangentialKnifeTransformer

logger = logging.getLogger(__name__)

ADDON_NAME = "dragknife"


@hookimpl
def register_steps(step_registry):
    """Register knife steps with the step registry."""
    if not KNIFE_TRANSFORMS_AVAILABLE:
        logger.warning(
            "The dragknife addon requires a raygeo release with the "
            "knife transforms; skipping step registration."
        )
        return
    step_registry.register(DragKnifeCutStep, addon_name=ADDON_NAME)
    step_registry.register(TangentialKnifeCutStep, addon_name=ADDON_NAME)


@hookimpl
def register_transformers(transformer_registry):
    """Register the knife compensation transformers."""
    if not KNIFE_TRANSFORMS_AVAILABLE:
        logger.warning(
            "The dragknife addon requires a raygeo release with the "
            "knife transforms; skipping transformer registration."
        )
        return
    transformer_registry.register(DragKnifeTransformer, addon_name=ADDON_NAME)
    transformer_registry.register(
        TangentialKnifeTransformer, addon_name=ADDON_NAME
    )

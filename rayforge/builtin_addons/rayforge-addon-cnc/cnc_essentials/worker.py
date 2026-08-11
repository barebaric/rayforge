"""
Backend entry point for cnc-essentials addon.

Registers steps with the main application.
"""

from rayforge.core.hooks import hookimpl

from .steps import (
    AdaptiveClearStep,
    FlatSpiralStep,
    HelixPlungeStep,
    ProfileInnerStep,
    ProfileOuterStep,
    RampEntryStep,
    SlotStep,
    ToroidalClearStep,
)

ADDON_NAME = "cnc_essentials"


@hookimpl
def register_steps(step_registry):
    """Register CNC steps with the step registry."""
    step_registry.register(AdaptiveClearStep, addon_name=ADDON_NAME)
    step_registry.register(HelixPlungeStep, addon_name=ADDON_NAME)
    step_registry.register(FlatSpiralStep, addon_name=ADDON_NAME)
    step_registry.register(RampEntryStep, addon_name=ADDON_NAME)
    step_registry.register(ToroidalClearStep, addon_name=ADDON_NAME)
    step_registry.register(SlotStep, addon_name=ADDON_NAME)
    step_registry.register(ProfileInnerStep, addon_name=ADDON_NAME)
    step_registry.register(ProfileOuterStep, addon_name=ADDON_NAME)

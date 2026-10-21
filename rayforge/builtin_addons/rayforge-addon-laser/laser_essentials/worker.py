"""
Backend entry point for laser-essentials addon.

Registers steps with the main application.
"""

from rayforge.core.hooks import hookimpl

from .steps import (
    ContourStep,
    EngraveStep,
    FrameStep,
    MaterialTestStep,
    OffsetFillStep,
    ShrinkWrapStep,
    WavefrontStep,
)

ADDON_NAME = "laser_essentials"


@hookimpl
def register_steps(step_registry):
    """Register steps with the step registry."""
    step_registry.register(ContourStep, addon_name=ADDON_NAME)
    step_registry.register(EngraveStep, addon_name=ADDON_NAME)
    step_registry.register(FrameStep, addon_name=ADDON_NAME)
    step_registry.register(MaterialTestStep, addon_name=ADDON_NAME)
    step_registry.register(OffsetFillStep, addon_name=ADDON_NAME)
    step_registry.register(ShrinkWrapStep, addon_name=ADDON_NAME)
    step_registry.register(WavefrontStep, addon_name=ADDON_NAME)


@hookimpl
def register_asset_types(asset_type_registry):
    """Register the record that keeps a calibration test's settings."""
    from .calibration.record import CalibrationRecord

    asset_type_registry.register(
        CalibrationRecord, CalibrationRecord.asset_type_name, ADDON_NAME
    )

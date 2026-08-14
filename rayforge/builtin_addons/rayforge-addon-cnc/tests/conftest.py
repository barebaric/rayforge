"""
Pytest configuration for cnc_essentials builtin addon tests.

This conftest ensures that steps are registered with the step registry
before tests run, mirroring the laser_essentials addon conftest.
"""

from unittest.mock import MagicMock

import pytest
from cnc_essentials.frontend import register_step_settings_pages
from cnc_essentials.steps import (
    AdaptiveClearStep,
    FlatSpiralStep,
    HelixPlungeStep,
    ProfileInnerStep,
    ProfileOuterStep,
    RampEntryStep,
    SlotStep,
    ToroidalClearStep,
)

from rayforge.core.step_registry import step_registry
from rayforge.machine.models.spindle import SpindleHead
from rayforge.ui_gtk.doceditor.step_settings.page_registry import (
    step_settings_page_registry,
)


@pytest.fixture
def machine():
    """A machine with a spindle head for CNC steps."""
    m = MagicMock()
    m.heads = [SpindleHead()]
    return m


def _register_steps():
    """Register all steps from cnc_essentials addon."""
    step_registry.register(AdaptiveClearStep, addon_name="cnc_essentials")
    step_registry.register(HelixPlungeStep, addon_name="cnc_essentials")
    step_registry.register(FlatSpiralStep, addon_name="cnc_essentials")
    step_registry.register(RampEntryStep, addon_name="cnc_essentials")
    step_registry.register(ToroidalClearStep, addon_name="cnc_essentials")
    step_registry.register(SlotStep, addon_name="cnc_essentials")
    step_registry.register(ProfileInnerStep, addon_name="cnc_essentials")
    step_registry.register(ProfileOuterStep, addon_name="cnc_essentials")


@pytest.fixture(scope="session", autouse=True)
def register_cnc_essentials():
    """
    Automatically register cnc_essentials steps and settings pages
    for all tests in this addon.
    """
    _register_steps()
    register_step_settings_pages(step_settings_page_registry)
    yield

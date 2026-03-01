"""
Pytest configuration for laser_essentials builtin addon tests.

This conftest ensures that producers and steps are registered
with their respective registries before tests run.
"""

import pytest
from rayforge.pipeline.producer.registry import producer_registry
from rayforge.core.step_registry import step_registry


def _register_producers():
    """Register all producers from laser_essentials addon."""
    from laser_essentials.producers import (
        ContourProducer,
        FrameProducer,
        MaterialTestGridProducer,
        Rasterizer,
        ShrinkWrapProducer,
    )

    producer_registry.register(ContourProducer, addon_name="laser_essentials")
    producer_registry.register(Rasterizer, addon_name="laser_essentials")
    producer_registry.register(
        Rasterizer, name="DepthEngraver", addon_name="laser_essentials"
    )
    producer_registry.register(
        Rasterizer,
        name="DitherRasterizer",
        addon_name="laser_essentials",
    )
    producer_registry.register(FrameProducer, addon_name="laser_essentials")
    producer_registry.register(
        MaterialTestGridProducer, addon_name="laser_essentials"
    )
    producer_registry.register(
        ShrinkWrapProducer, addon_name="laser_essentials"
    )


def _register_steps():
    """Register all steps from laser_essentials addon."""
    from laser_essentials.steps import (
        ContourStep,
        EngraveStep,
        FrameStep,
        MaterialTestStep,
        ShrinkWrapStep,
    )

    step_registry.register(ContourStep, addon_name="laser_essentials")
    step_registry.register(EngraveStep, addon_name="laser_essentials")
    step_registry.register(FrameStep, addon_name="laser_essentials")
    step_registry.register(MaterialTestStep, addon_name="laser_essentials")
    step_registry.register(ShrinkWrapStep, addon_name="laser_essentials")


@pytest.fixture(scope="session", autouse=True)
def register_laser_essentials():
    """
    Automatically register laser_essentials producers and steps
    for all tests in this addon.

    This also prevents ensure_addons_loaded() from loading via
    AddonManager, which would register classes from a different
    module path (rayforge_plugins.*) causing isinstance() checks
    to fail in tests.
    """
    from rayforge import worker_init

    worker_init._worker_addons_loaded = True

    _register_producers()
    _register_steps()
    yield

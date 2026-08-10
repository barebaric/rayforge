"""
Pytest configuration for laser_essentials builtin addon tests.

This conftest ensures that producers, steps, and assemblers are registered
with their respective registries before tests run.
"""

from unittest.mock import MagicMock

import pluggy
import pytest
from laser_essentials.steps import (
    ContourStep,
    EngraveStep,
    FrameStep,
    MaterialTestStep,
    ShrinkWrapStep,
)

from rayforge.addon_mgr.addon_manager import AddonManager
from rayforge.config import BUILTIN_ADDONS_DIR
from rayforge.core.hooks import RayforgeSpecs
from rayforge.core.step_registry import step_registry
from rayforge.machine.models.laser import LaserHead
from rayforge.pipeline.transformer.registry import transformer_registry


@pytest.fixture
def machine():
    """A machine whose resolved laser defaults mirror the historical
    ``machine_defaults`` fixture: laser head spot (0.1, 0.1), arc
    tolerance 0.03, arcs supported, curves not supported."""
    m = MagicMock()
    m.arc_tolerance = 0.03
    m.supports_arcs = True
    m.supports_curves = False
    head = MagicMock(spec=LaserHead)
    head.uid = "laser-1"
    head.spot_size_mm = (0.1, 0.1)
    m.heads = [head]
    return m


def _register_steps():
    """Register all steps from laser_essentials addon."""
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
    module path (rayforge_addons.*) causing isinstance() checks
    to fail in tests.
    """
    plugin_mgr = pluggy.PluginManager("rayforge")
    plugin_mgr.add_hookspecs(RayforgeSpecs)

    mgr = AddonManager(
        [BUILTIN_ADDONS_DIR], BUILTIN_ADDONS_DIR, plugin_mgr, MagicMock()
    )
    mgr.set_registries({"transformer_registry": transformer_registry})
    mgr.load_addon_by_name("post_processors", worker_only=True)

    _register_steps()
    yield

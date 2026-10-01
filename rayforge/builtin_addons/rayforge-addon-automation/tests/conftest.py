"""
Pytest configuration for the automation builtin addon tests.

This conftest ensures that the CommandStep is registered with the
step registry before tests run.
"""

from unittest.mock import MagicMock

import pluggy
import pytest
from automation.steps import CommandStep

from rayforge.addon_mgr.addon_manager import AddonManager
from rayforge.config import BUILTIN_ADDONS_DIR
from rayforge.core.hooks import RayforgeSpecs
from rayforge.core.step_registry import step_registry
from rayforge.pipeline.transformer.registry import transformer_registry


@pytest.fixture
def machine():
    """A minimal machine mock: G-code capable, no special features."""
    m = MagicMock()
    m.driver_name = "GrblSerialDriver"
    m.max_cut_speed = 5000
    m.max_travel_speed = 10000
    m.heads = []
    return m


def _register_steps():
    """Register all steps from the automation addon."""
    step_registry.register(CommandStep, addon_name="automation")


@pytest.fixture(scope="session", autouse=True)
def register_automation():
    """
    Automatically register the automation addon's steps for all
    tests in this addon.

    This also prevents ensure_addons_loaded() from loading via
    AddonManager, which would register classes from a different
    module path causing isinstance() checks to fail in tests.
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

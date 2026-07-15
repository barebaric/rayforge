"""
Pytest configuration for laser_essentials builtin addon tests.

This conftest ensures that producers, steps, and assemblers are registered
with their respective registries before tests run.
"""

import pytest
from unittest.mock import MagicMock

from rayforge.core.step_registry import step_registry
from rayforge.pipeline.assembler.registry import assembler_registry
from rayforge.pipeline.producer.registry import producer_registry
from rayforge.pipeline.transformer.registry import transformer_registry


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


def _register_assemblers():
    """Register all assembler functions from laser_essentials addon."""
    from raygeo.ops.assembly.contour import contour
    from raygeo.ops.assembly.frame import frame
    from raygeo.ops.assembly.material_test_grid import (
        generate_material_test_grid,
    )
    from raygeo.ops.assembly.raster import raster
    from raygeo.ops.assembly.shrinkwrap import shrinkwrap
    from raygeo.ops.assembly.wavefront import (
        adaptive_wavefronts_multi_pocket,
    )

    assembler_registry.register(
        "contour", contour, addon_name="laser_essentials"
    )
    assembler_registry.register(
        "frame", frame, addon_name="laser_essentials"
    )
    assembler_registry.register(
        "shrinkwrap", shrinkwrap, addon_name="laser_essentials"
    )
    assembler_registry.register(
        "raster", raster, addon_name="laser_essentials"
    )
    assembler_registry.register(
        "wavefront", adaptive_wavefronts_multi_pocket,
        addon_name="laser_essentials",
    )
    assembler_registry.register(
        "material_test_grid", generate_material_test_grid,
        addon_name="laser_essentials",
    )


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
    from rayforge import worker_init
    from rayforge.addon_mgr.addon_manager import AddonManager
    from rayforge.config import BUILTIN_ADDONS_DIR

    worker_init._worker_addons_loaded = True

    import pluggy

    from rayforge.core.hooks import RayforgeSpecs

    plugin_mgr = pluggy.PluginManager("rayforge")
    plugin_mgr.add_hookspecs(RayforgeSpecs)

    mgr = AddonManager(
        [BUILTIN_ADDONS_DIR], BUILTIN_ADDONS_DIR, plugin_mgr, MagicMock()
    )
    mgr.set_registries({"transformer_registry": transformer_registry})
    mgr.load_addon_by_name("post_processors", worker_only=True)

    _register_producers()
    _register_steps()
    _register_assemblers()
    yield

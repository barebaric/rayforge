"""
Backend entry point for laser-essentials addon.

Registers producers, steps, and assemblers with the main application.
"""

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

from rayforge.core.hooks import hookimpl

from .steps import (
    ContourStep,
    EngraveStep,
    FrameStep,
    MaterialTestStep,
    ShrinkWrapStep,
    WavefrontStep,
)

ADDON_NAME = "laser_essentials"


@hookimpl
def register_assemblers(assembler_registry):
    """Register assembler functions with the assembler registry."""
    assembler_registry.register("contour", contour, addon_name=ADDON_NAME)
    assembler_registry.register("frame", frame, addon_name=ADDON_NAME)
    assembler_registry.register(
        "shrinkwrap", shrinkwrap, addon_name=ADDON_NAME
    )
    assembler_registry.register("raster", raster, addon_name=ADDON_NAME)
    assembler_registry.register(
        "wavefront",
        adaptive_wavefronts_multi_pocket,
        addon_name=ADDON_NAME,
    )
    assembler_registry.register(
        "material_test_grid",
        generate_material_test_grid,
        addon_name=ADDON_NAME,
    )


@hookimpl
def register_steps(step_registry):
    """Register steps with the step registry."""
    step_registry.register(ContourStep, addon_name=ADDON_NAME)
    step_registry.register(EngraveStep, addon_name=ADDON_NAME)
    step_registry.register(FrameStep, addon_name=ADDON_NAME)
    step_registry.register(MaterialTestStep, addon_name=ADDON_NAME)
    step_registry.register(ShrinkWrapStep, addon_name=ADDON_NAME)
    step_registry.register(WavefrontStep, addon_name=ADDON_NAME)

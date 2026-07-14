"""
Backend entry point for laser-essentials addon.

Registers producers, steps, and assemblers with the main application.
"""

import gettext
from pathlib import Path

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
from rayforge.image.tracing import prepare_surface
from rayforge.pipeline.assembler.registry import AssemblerMeta

from .producers import (
    ContourProducer,
    FrameProducer,
    MaterialTestGridProducer,
    Rasterizer,
    ShrinkWrapProducer,
    WavefrontProducer,
)
from .steps import (
    ContourStep,
    EngraveStep,
    FrameStep,
    MaterialTestStep,
    ShrinkWrapStep,
    WavefrontStep,
)

_localedir = Path(__file__).parent.parent / "locale"
_t = gettext.translation(
    "laser_essentials", localedir=_localedir, fallback=True
)
_ = _t.gettext

ADDON_NAME = "laser_essentials"


@hookimpl
def register_producers(producer_registry):
    """Register producers with the producer registry."""
    producer_registry.register(WavefrontProducer, addon_name=ADDON_NAME)
    producer_registry.register(ContourProducer, addon_name=ADDON_NAME)
    producer_registry.register(Rasterizer, addon_name=ADDON_NAME)
    # DepthEngraver and DitherRasterizer are aliases for Rasterizer
    producer_registry.register(
        Rasterizer, name="DepthEngraver", addon_name=ADDON_NAME
    )
    producer_registry.register(
        Rasterizer, name="DitherRasterizer", addon_name=ADDON_NAME
    )
    producer_registry.register(FrameProducer, addon_name=ADDON_NAME)
    producer_registry.register(MaterialTestGridProducer, addon_name=ADDON_NAME)
    producer_registry.register(ShrinkWrapProducer, addon_name=ADDON_NAME)


@hookimpl
def register_assemblers(assembler_registry):
    """Register assembler functions with the assembler registry."""
    assembler_registry.register(
        "contour",
        AssemblerMeta(
            assemble=contour,
            is_vector=True,
            requires_full_render=False,
            param_keys=(
                "kerf_mm",
                "path_offset_mm",
                "cut_side",
                "overcut",
                "cut_order",
                "remove_inner",
                "arc_tolerance",
                "allow_arcs",
                "supports_curves",
            ),
            split_contours=True,
        ),
        addon_name=ADDON_NAME,
    )
    assembler_registry.register(
        "frame",
        AssemblerMeta(
            assemble=frame,
            is_vector=True,
            requires_full_render=False,
            param_keys=("kerf_mm", "path_offset_mm", "cut_side"),
            set_power=True,
        ),
        addon_name=ADDON_NAME,
    )
    assembler_registry.register(
        "shrinkwrap",
        AssemblerMeta(
            assemble=shrinkwrap,
            is_vector=True,
            requires_full_render=True,
            param_keys=(
                "gravity",
                "kerf_mm",
                "path_offset_mm",
                "cut_side",
                "arc_tolerance",
                "allow_arcs",
                "supports_curves",
            ),
            prepare_surface=prepare_surface,
            set_power=True,
        ),
        addon_name=ADDON_NAME,
    )
    assembler_registry.register(
        "raster",
        AssemblerMeta(
            assemble=raster,
            is_vector=False,
            requires_full_render=False,
            param_keys=(
                "alpha",
                "mode",
                "line_interval_mm",
                "sample_interval_mm",
                "min_power",
                "max_power",
                "step_power",
                "num_power_levels",
                "angle",
                "offset_x_mm",
                "offset_y_mm",
                "scan_mode",
                "cross_hatch",
                "num_depth_levels",
                "z_step_down",
                "angle_increment",
            ),
            build_part_mode="raster",
            always_wrap=True,
            section_type="RASTER_FILL",
        ),
        addon_name=ADDON_NAME,
    )
    assembler_registry.register(
        "wavefront",
        AssemblerMeta(
            assemble=adaptive_wavefronts_multi_pocket,
            is_vector=True,
            requires_full_render=False,
            param_keys=(
                "step_over",
                "offset_mm",
                "area_tolerance",
                "precision",
                "cut_feed_rate",
                "cut_power",
            ),
            normalize_windings=True,
        ),
        addon_name=ADDON_NAME,
    )
    assembler_registry.register(
        "material_test_grid",
        AssemblerMeta(
            assemble=generate_material_test_grid,
            is_vector=True,
            requires_full_render=False,
            param_keys=(
                "size_mm",
                "cols",
                "rows",
                "min_speed",
                "max_speed",
                "min_power",
                "max_power",
                "min_passes",
                "max_passes",
                "fixed_speed",
                "fixed_power",
                "shape_size",
                "spacing",
                "line_interval_mm",
                "mode",
                "grid_mode",
                "include_labels",
            ),
            build_part_mode="none",
            normalize_windings=True,
        ),
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

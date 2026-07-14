"""
Registry for assembler functions.

Provides a name-based lookup for assembler callables, allowing producers
to delegate to registered assemblers without importing raygeo directly.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from rayforge.worker_init import ensure_addons_loaded

logger = logging.getLogger(__name__)


@dataclass
class AssemblerMeta:
    """Metadata for a registered assembler function.

    In addition to the assembler callable itself, this carries all the
    orchestration hints the stage needs to replace per-producer
    ``run()`` methods with a single generic dispatch.
    """

    assemble: Callable[..., Any]
    is_vector: bool
    requires_full_render: bool = False
    param_keys: Tuple[str, ...] = field(default_factory=tuple)

    build_part_mode: str = "vector"
    """How to build the Part: ``"vector"``, ``"raster"``, or ``"none"``."""

    normalize_windings: bool = False
    """If True, normalise winding orders on the vector Part."""

    override_threshold: bool = False
    """If True, trace the surface instead of using source vectors."""

    prepare_surface: Optional[Callable[..., Any]] = None
    """Optional callable to pre-process a surface before Part building
    (e.g. ``prepare_surface`` for shrinkwrap).  Called as
    ``prepare_surface(surface)`` and the result stored on
    ``part.image``."""

    split_contours: bool = False
    """If True, split the result ops into individual contours."""

    set_power: bool = False
    """If True, inject ``settings["power"]`` into each section."""

    always_wrap: bool = False
    """If True, wrap even when the assembler returns no ops."""

    section_type: str = "VECTOR_OUTLINE"
    """The ``SectionType`` name for section markers."""


class AssemblerRegistry:
    """
    Registry for assembler functions.

    Maps string names to AssemblerMeta instances so that producers
    can look up and call assemblers without importing raygeo directly.
    """

    def __init__(self):
        self._assemblers: Dict[str, AssemblerMeta] = {}

    def register(
        self,
        name: str,
        meta: AssemblerMeta,
        addon_name: Optional[str] = None,
    ) -> None:
        """
        Register an assembler.

        Args:
            name: The name to register the assembler under.
            meta: The AssemblerMeta for this assembler.
            addon_name: Optional addon name for cleanup tracking.
        """
        if name in self._assemblers:
            logger.warning(
                "Assembler '%s' already registered, overwriting", name
            )
        self._assemblers[name] = meta

    def unregister(self, name: str) -> bool:
        """
        Unregister an assembler by name.

        Returns:
            True if the assembler was unregistered, False if not found.
        """
        if name in self._assemblers:
            del self._assemblers[name]
            return True
        return False

    def get(self, name: str) -> Optional[AssemblerMeta]:
        """
        Look up an assembler by name.

        Returns:
            The AssemblerMeta, or None if not found.
        """
        meta = self._assemblers.get(name)
        if meta is None:
            ensure_addons_loaded()
            meta = self._assemblers.get(name)
        if meta is None and not self._assemblers:
            self._register_laser_assemblers()
            meta = self._assemblers.get(name)
        return meta

    @staticmethod
    def _register_laser_assemblers():
        """Fallback: register laser assemblers directly.

        Called when the normal addon loading path (``ensure_addons_loaded``)
        did not populate the registry — for example inside a subprocess
        worker that lacks the addon manifest in shared state.
        """
        try:
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
            from rayforge.image.tracing import prepare_surface
        except ImportError:
            return

        _reg = assembler_registry
        _reg.register(
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
        )
        _reg.register(
            "frame",
            AssemblerMeta(
                assemble=frame,
                is_vector=True,
                requires_full_render=False,
                param_keys=(
                    "kerf_mm",
                    "path_offset_mm",
                    "cut_side",
                ),
                set_power=True,
            ),
        )
        _reg.register(
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
        )
        _reg.register(
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
        )
        _reg.register(
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
        )
        _reg.register(
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
        )

    def assemble(self, name: str, part: Any = None, **kwargs: Any) -> Any:
        """
        Look up an assembler by name and call it.

        Filters kwargs to only include keys defined in the assembler's
        param_keys before calling the function. If part is None, it is
        not passed to the function.

        Args:
            name: The registered name of the assembler.
            part: The Part to pass as the first positional argument,
                  or None for assemblers that don't need one.
            **kwargs: Parameters to forward to the assembler function.

        Returns:
            The result of calling the assembler function.

        Raises:
            KeyError: If no assembler is registered under the given name.
        """
        meta = self._assemblers.get(name)
        if meta is None:
            ensure_addons_loaded()
            meta = self._assemblers[name]
        if meta.param_keys:
            filtered = {
                k: v for k, v in kwargs.items() if k in meta.param_keys
            }
        else:
            filtered = kwargs
        if part is not None:
            return meta.assemble(part, **filtered)
        return meta.assemble(**filtered)

    def all_assemblers(self) -> Dict[str, AssemblerMeta]:
        """Return a copy of all registered assemblers."""
        return self._assemblers.copy()


assembler_registry = AssemblerRegistry()

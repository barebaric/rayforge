"""
Helper functions for the assembler-based pipeline.

These functions absorb the Part-construction, image-preprocessing,
and result-wrapping logic that currently lives inside each producer's
``run()`` method, so that the stage can call raygeo assemblers
directly without needing producer class instances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Tuple,
)

import numpy as np
from raygeo.geo import Geometry, Matrix
from raygeo.image.grayscale import (
    compute_auto_levels,
    normalize_grayscale,
)
from raygeo.ops import Ops
from raygeo.ops.assembly import AssemblyResult
from raygeo.ops.part import Part
from raygeo.ops.types import RasterMode, SectionType

from ...core.vectorization_spec import TraceSpec
from ...image.dither import DitherAlgorithm, surface_to_dithered_array
from ...image.tracing import trace_surface
from ...image.util.grayscale import surface_to_binary, surface_to_grayscale
from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem

if TYPE_CHECKING:
    import cairo

    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser

logger = logging.getLogger(__name__)


class DepthMode(Enum):
    """Rasterisation depth mode.

    Each mode controls how pixel intensity maps to laser output:

    * ``POWER_MODULATION`` — variable power proportional to darkness.
    * ``CONSTANT_POWER`` — binary mask, constant-power scan lines.
    * ``DITHER`` — Floyd-Steinberg / ordered dither to binary.
    * ``MULTI_PASS`` — repeated Z-stepped passes through the depth.
    """

    POWER_MODULATION = auto()
    CONSTANT_POWER = auto()
    DITHER = auto()
    MULTI_PASS = auto()

    @property
    def display_name(self) -> str:
        names = {
            DepthMode.POWER_MODULATION: _("Variable Power"),
            DepthMode.CONSTANT_POWER: _("Constant Power"),
            DepthMode.DITHER: _("Dither"),
            DepthMode.MULTI_PASS: _("Multiple Depths"),
        }
        return names[self]

    @property
    def short_name(self) -> str:
        names = {
            DepthMode.POWER_MODULATION: _("Variable"),
            DepthMode.CONSTANT_POWER: _("Constant"),
            DepthMode.DITHER: _("Dither"),
            DepthMode.MULTI_PASS: _("Multi-Pass"),
        }
        return names[self]

    @property
    def raygeo_name(self) -> str:
        """Return the string expected by the raygeo ``raster()`` call."""
        names = {
            DepthMode.POWER_MODULATION: "power_modulated",
            DepthMode.CONSTANT_POWER: "mask_scan",
            DepthMode.DITHER: "dither",
            DepthMode.MULTI_PASS: "multi_pass",
        }
        return names[self]

    @property
    def raster_mode(self) -> RasterMode:
        """Return the :class:`RasterMode` for this depth mode."""
        _raster_mode_map = {
            DepthMode.POWER_MODULATION: RasterMode.VARIABLE_POWER,
            DepthMode.CONSTANT_POWER: RasterMode.CONSTANT_POWER,
            DepthMode.DITHER: RasterMode.CONSTANT_POWER,
            DepthMode.MULTI_PASS: RasterMode.DEPTH_MAP,
        }
        return _raster_mode_map[self]


@dataclass(frozen=True)
class MachineDefaults:
    """Resolved machine-level defaults for assembler parameters.

    Every producer currently inlines its own resolution of these
    values from the ``Laser`` model and the step ``settings`` dict.
    This dataclass centralises that logic so callers can resolve
    once and pass the result through.
    """

    kerf_mm: float
    arc_tolerance: float
    allow_arcs: bool
    supports_curves: bool
    line_interval_mm: float
    step_power: float
    tool_radius: float
    step_over: float
    cut_speed: int


def resolve_machine_defaults(
    laser: Laser,
    settings: Optional[Dict[str, Any]] = None,
) -> MachineDefaults:
    """Resolve machine defaults from a Laser model and step settings.

    Resolution order for each field mirrors the existing per-producer
    logic:

    * ``kerf_mm`` — ``settings["kerf_mm"]`` → ``laser.spot_size_mm[0]``
    * ``arc_tolerance`` — ``settings["arc_tolerance"]`` → ``0.03``
    * ``allow_arcs`` — ``settings["machine_supports_arcs"]`` →
      ``settings["output_arcs"]`` → ``True``
    * ``supports_curves`` — ``settings["machine_supports_curves"]`` →
      ``False``
    * ``line_interval_mm`` — ``laser.spot_size_mm[1]``
    * ``step_power`` — ``settings["power"]`` → ``1.0``
    * ``tool_radius`` — ``laser.spot_size_mm[0] / 2``
    * ``step_over`` — ``laser.spot_size_mm[0]``
    * ``cut_speed`` — ``settings["cut_speed"]`` → ``500``
    """
    s = settings or {}

    spot_x = laser.spot_size_mm[0]
    spot_y = laser.spot_size_mm[1]

    return MachineDefaults(
        kerf_mm=s.get("kerf_mm", spot_x),
        arc_tolerance=s.get("arc_tolerance", 0.03),
        allow_arcs=s.get("machine_supports_arcs", s.get("output_arcs", True)),
        supports_curves=s.get("machine_supports_curves", False),
        line_interval_mm=spot_y,
        step_power=s.get("power", 1.0),
        tool_radius=spot_x / 2.0,
        step_over=spot_x,
        cut_speed=s.get("cut_speed", 500),
    )


def _trace_surface_to_mm_geometry(
    surface: cairo.ImageSurface,
    workpiece: WorkPiece,
    threshold: float = 0.5,
    auto_threshold: bool = True,
    invert: bool = False,
) -> Optional[Geometry]:
    """Trace a rendered surface into a single Geometry in mm-space.

    The traced contours come back in pixel space (Y-down, origin
    top-left).  They are transformed to mm-space (Y-up, origin
    bottom-left) at the workpiece's physical size.

    Returns ``None`` if tracing yields no contours.
    """
    spec = TraceSpec(
        threshold=threshold,
        auto_threshold=auto_threshold,
        invert=invert,
    )
    traced = trace_surface(surface, vectorization_spec=spec)
    if not traced:
        return None

    width_mm, height_mm = workpiece.size
    px_w = surface.get_width()
    px_h = surface.get_height()
    if px_w <= 0 or px_h <= 0:
        return None

    scale_x = width_mm / px_w
    scale_y = height_mm / px_h
    transform = Matrix.translation(0, height_mm) @ Matrix.scale(
        scale_x, -scale_y
    )

    merged = Geometry()
    for geo in traced:
        geo.transform(transform)
        merged.extend(geo)
    return merged


def build_part_vector(
    workpiece: WorkPiece,
    surface: Optional[cairo.ImageSurface] = None,
    *,
    override_threshold: bool = False,
    threshold: float = 0.5,
    normalize_windings: bool = False,
) -> Optional[Part]:
    """Build a ``Part`` carrying vector geometry for an assembler.

    This absorbs the Part-construction logic shared by
    ``ContourProducer``, ``FrameProducer``, ``ShrinkWrapProducer``,
    and ``WavefrontProducer``.

    Resolution order:

    1. **Vector source** — if the workpiece has boundaries and
       ``override_threshold`` is False, use ``workpiece.to_part()``.
       When ``normalize_windings`` is True (e.g. WavefrontProducer),
       the geometry is re-scaled manually so that
       ``normalize_winding_orders()`` can be applied before
       constructing the Part.

    2. **Raster fallback** — if a ``surface`` is available (either
       because there are no boundaries or because
       ``override_threshold`` is True), trace the surface into
       mm-space geometry and build a Part from that.

    3. Returns ``None`` if no geometry could be obtained.

    Args:
        workpiece: The WorkPiece to derive geometry from.
        surface: Optional rendered Cairo surface for raster tracing.
        override_threshold: If True, ignore vector boundaries and
            trace the surface instead.
        threshold: Brightness threshold (0–1) for raster tracing.
        normalize_windings: If True, normalize winding orders
            (outer CCW, inner CW) on the scaled geometry.  Required
            by wavefront/pocketing assemblers.
    """
    boundaries = workpiece.boundaries
    has_vector_source = boundaries is not None and not boundaries.is_empty()

    # 1. Vector source — preferred path.
    if has_vector_source and not override_threshold:
        if normalize_windings:
            assert boundaries is not None
            scaled = boundaries.copy()
            w, h = workpiece.size
            if w > 0 and h > 0:
                scaled.transform(Matrix.scale(w, h))
            scaled.normalize_winding_orders()
            return Part(geometry=scaled, size_mm=(w, h))
        return workpiece.to_part()

    # 2. Raster fallback — trace the surface.
    if surface is not None:
        geo = _trace_surface_to_mm_geometry(
            surface,
            workpiece,
            threshold=threshold,
            auto_threshold=not override_threshold,
        )
        if geo is not None and not geo.is_empty():
            if normalize_windings:
                geo.normalize_winding_orders()
            return Part(geometry=geo, size_mm=workpiece.size)

    return None


def build_part_raster(
    workpiece: WorkPiece,
    pixels_per_mm: Tuple[float, float],
) -> Part:
    """Build a ``Part`` for a raster assembler.

    Raster assemblers (``raster()``, ``shrinkwrap()``) do not need
    vector geometry — they work from image data passed separately.
    The Part carries only physical metadata: ``size_mm`` and
    ``pixels_per_mm``.

    Args:
        workpiece: The WorkPiece being processed.
        pixels_per_mm: The (x, y) resolution of the rendered surface.
    """
    return Part(
        size_mm=workpiece.size,
        pixels_per_mm=pixels_per_mm,
    )


def make_artifact(
    ops: Ops,
    workpiece: WorkPiece,
    generation_id: int,
    *,
    is_vector: bool = True,
    source_dimensions: Optional[Tuple[float, float]] = None,
) -> WorkPieceArtifact:
    """Create a ``WorkPieceArtifact`` from already-assembled ops.

    This absorbs the common artifact-creation code shared by every
    producer's ``run()`` return statement.

    Args:
        ops: The final, fully-wrapped Ops sequence.
        workpiece: The WorkPiece that was processed.
        generation_id: The generation ID for staleness tracking.
        is_vector: If True, the coordinate system is
            ``MILLIMETER_SPACE`` and ``source_dimensions`` defaults
            to ``workpiece.size``.  If False, ``PIXEL_SPACE`` and
            ``source_dimensions`` must be provided.
        source_dimensions: Override for ``source_dimensions``.  For
            raster producers, pass ``(width_px, height_px)``.
    """
    if source_dimensions is None:
        source_dimensions = workpiece.size

    coord_sys = (
        CoordinateSystem.MILLIMETER_SPACE
        if is_vector
        else CoordinateSystem.PIXEL_SPACE
    )

    return WorkPieceArtifact(
        ops=ops,
        is_scalable=False,
        source_coordinate_system=coord_sys,
        source_dimensions=source_dimensions,
        generation_size=workpiece.size,
        generation_id=generation_id,
    )


def wrap_assembler_result(
    result: AssemblyResult,
    workpiece: WorkPiece,
    laser: Laser,
    generation_id: int,
    *,
    section_type: SectionType = SectionType.VECTOR_OUTLINE,
    split_contours: bool = False,
    set_power: Optional[float] = None,
    raster_mode: Optional[RasterMode] = None,
    is_vector: bool = True,
    source_dimensions: Optional[Tuple[float, float]] = None,
    always_wrap: bool = False,
) -> WorkPieceArtifact:
    """Wrap an ``AssemblyResult`` into a ``WorkPieceArtifact``.

    This absorbs the post-processing logic shared by
    ``ContourProducer``, ``FrameProducer``, ``ShrinkWrapProducer``,
    and ``Rasterizer`` — building an ``Ops`` sequence with section
    markers, head assignment, and optional power, then wrapping it
    in a ``WorkPieceArtifact``.

    Args:
        result: The ``AssemblyResult`` from a raygeo assembler.
        workpiece: The WorkPiece being processed.
        laser: The Laser model (used for ``laser.uid`` head
            assignment).
        generation_id: The generation ID for staleness tracking.
        section_type: The ``SectionType`` for section markers.
            Defaults to ``VECTOR_OUTLINE``.
        split_contours: If True, split the result ops into
            individual contours and wrap each in its own section
            (used by ContourProducer).  If False, all ops go in a
            single section.
        set_power: If not None, emit a ``set_power`` command
            inside each section (used by Frame/ShrinkWrap).
        is_vector: If True, use ``MILLIMETER_SPACE`` coordinate
            system.  If False, ``PIXEL_SPACE`` (used by Rasterizer).
        source_dimensions: Override for ``source_dimensions``.
            For raster producers, pass
            ``(width_px, height_px)``.
    """
    final_ops = Ops()

    has_ops = result.ops.len() > 0
    if has_ops:
        final_ops.set_head(laser.uid)

    if always_wrap or has_ops:
        if split_contours:
            contour_geo = result.ops.to_geometry()
            contour_list = contour_geo.split_into_contours()
            for c in contour_list:
                final_ops.ops_section_start(
                    section_type, workpiece.uid, raster_mode=raster_mode
                )
                final_ops.extend(Ops.from_geometry(c))
                final_ops.ops_section_end(
                    section_type, raster_mode=raster_mode
                )
        else:
            final_ops.ops_section_start(
                section_type, workpiece.uid, raster_mode=raster_mode
            )
            if set_power is not None:
                final_ops.set_power(set_power)
            final_ops.extend(result.ops)
            final_ops.ops_section_end(section_type, raster_mode=raster_mode)

    return make_artifact(
        final_ops,
        workpiece,
        generation_id,
        is_vector=is_vector,
        source_dimensions=source_dimensions,
    )


def preprocess_raster_image(
    surface: "cairo.ImageSurface",
    *,
    mode: DepthMode,
    invert: bool = False,
    auto_levels: bool = True,
    computed_auto_levels: Optional[Tuple[int, int]] = None,
    black_point: int = 0,
    white_point: int = 255,
    threshold: int = 128,
    dither_algorithm: Optional[DitherAlgorithm] = None,
    laser_spot_x_mm: float = 0.1,
    pixels_per_mm_x: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert a Cairo surface into an image array for a raster assembler.

    Handles depth-mode preprocessing: grayscale with optional levels,
    dithering, or binary thresholding.

    Args:
        surface: The rendered Cairo ARGB32 surface.
        mode: The rasterisation depth mode.
        invert: If True, invert grayscale before further processing.
        auto_levels: For grayscale modes, if True, auto-compute
            black/white points from the image histogram.
        computed_auto_levels: Pre-computed ``(black, white)`` points
            from a low-resolution preview (e.g. from
            ``compute_raster_auto_levels``).  When provided, skips
            per-chunk auto-level computation.
        black_point: Manual black-point (0–255) when
            ``auto_levels`` is False.
        white_point: Manual white-point (0–255) when
            ``auto_levels`` is False.
        threshold: Brightness threshold (0–255) for binary
            ``CONSTANT_POWER`` mode.
        dither_algorithm: Dithering algorithm for ``DITHER`` mode.
        laser_spot_x_mm: Laser spot X diameter in mm, used to
            compute minimum feature size for dithering.
        pixels_per_mm_x: Rendered pixels-per-mm in X, used with
            ``laser_spot_x_mm`` for dither minimum feature size.

    Returns:
        ``(image, alpha)`` where *image* is a 2-D ``uint8`` array
        (grayscale or binary depending on mode) and *alpha* is a
        ``float32`` alpha array (grayscale modes) or ``None``
        (dither / constant-power modes).
    """
    if mode in (DepthMode.MULTI_PASS, DepthMode.POWER_MODULATION):
        gray_image, alpha = surface_to_grayscale(surface)
        if invert:
            alpha_mask = alpha > 0
            gray_image[alpha_mask] = 255 - gray_image[alpha_mask]
        gray_image = _apply_raster_levels(
            gray_image,
            alpha,
            auto_levels=auto_levels,
            computed_auto_levels=computed_auto_levels,
            black_point=black_point,
            white_point=white_point,
        )
        return gray_image, alpha

    if mode == DepthMode.DITHER:
        min_feature_px = max(
            1,
            int(round(laser_spot_x_mm * pixels_per_mm_x)),
        )
        algo = dither_algorithm or DitherAlgorithm.FLOYD_STEINBERG
        image = surface_to_dithered_array(
            surface,
            algo,
            invert=invert,
            min_feature_px=min_feature_px,
        )
        return image, None

    if mode == DepthMode.CONSTANT_POWER:
        image = surface_to_binary(
            surface,
            threshold=threshold,
            invert=invert,
        )
        return image, None

    return None, None


def _apply_raster_levels(
    gray_image: np.ndarray,
    alpha: np.ndarray,
    *,
    auto_levels: bool = True,
    computed_auto_levels: Optional[Tuple[int, int]] = None,
    black_point: int = 0,
    white_point: int = 255,
) -> np.ndarray:
    """Apply auto-levels or manual black/white-point normalization.

    This is the standalone equivalent of ``Rasterizer._apply_levels``.
    """
    if auto_levels:
        if computed_auto_levels is not None:
            bp, wp = computed_auto_levels
        else:
            bp, wp = compute_auto_levels(gray_image[alpha > 0])
    else:
        bp, wp = black_point, white_point

    if bp > 0 or wp < 255:
        gray_image = normalize_grayscale(gray_image, bp, wp)
    return gray_image


def compute_raster_auto_levels(
    workpiece: "WorkPiece",
    pixels_per_mm: Tuple[float, float],
    *,
    invert: bool = False,
    max_preview_pixels: int = 512,
) -> Optional[Tuple[int, int]]:
    """Compute auto-levels from a low-resolution preview render.

    Renders a small preview of the workpiece, converts it to
    grayscale, and returns the ``(black_point, white_point)`` tuple
    suitable for passing to ``preprocess_raster_image`` as
    ``computed_auto_levels``.

    This ensures consistent black/white points across all chunks
    when processing large images.

    Args:
        workpiece: The WorkPiece to render a preview of.
        pixels_per_mm: The ``(x, y)`` resolution of the full render.
        invert: If True, invert the grayscale before computing
            levels.
        max_preview_pixels: Maximum preview dimension in pixels.

    Returns:
        ``(black_point, white_point)`` tuple, or ``None`` if the
        preview render failed or auto-levels are not applicable.
    """
    px_per_mm_x, px_per_mm_y = pixels_per_mm
    size = workpiece.size

    scale = min(
        1.0,
        max_preview_pixels / (size[0] * px_per_mm_x),
        max_preview_pixels / (size[1] * px_per_mm_y),
    )

    preview_width = max(1, int(size[0] * px_per_mm_x * scale))
    preview_height = max(1, int(size[1] * px_per_mm_y * scale))

    surface = workpiece.render_to_pixels(preview_width, preview_height)
    if not surface:
        return None

    gray_image, alpha = surface_to_grayscale(surface)

    if invert:
        alpha_mask = alpha > 0
        gray_image[alpha_mask] = 255 - gray_image[alpha_mask]

    surface.flush()

    return compute_auto_levels(gray_image[alpha > 0])

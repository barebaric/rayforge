"""
Helper functions for the assembler-based pipeline.

These functions absorb the Part-construction, image-preprocessing,
and result-wrapping logic that currently lives inside each producer's
``run()`` method, so that the stage can call raygeo assemblers
directly without needing producer class instances.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
)

import numpy as np
from raygeo.geo import Geometry, Matrix
from raygeo.image.grayscale import (
    compute_auto_levels,
    normalize_grayscale,
)
from raygeo.ops.part import Part
from raygeo.ops.types import RasterMode

from ...core.vectorization_spec import TraceSpec
from ...image.dither import DitherAlgorithm, surface_to_dithered_array
from ...image.tracing import trace_surface
from ...image.util.grayscale import surface_to_binary, surface_to_grayscale

if TYPE_CHECKING:
    import cairo

    from ...core.workpiece import WorkPiece

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


def _trace_surface_to_mm_geometry(
    surface: cairo.ImageSurface,
    workpiece: WorkPiece,
    threshold: float = 0.5,
    auto_threshold: bool = True,
    invert: bool = False,
) -> Geometry | None:
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
    surface: cairo.ImageSurface | None = None,
    *,
    override_threshold: bool = False,
    threshold: float = 0.5,
    normalize_windings: bool = False,
) -> Part | None:
    """Build a ``Part`` carrying vector geometry for an assembler.

    This absorbs the Part-construction logic shared by
    ``ContourProducer``, ``FrameProducer``, ``ShrinkWrapProducer``,
    and ``WavefrontProducer``.

    Resolution order:

    1. **Vector source** — if the workpiece has boundaries and
       ``override_threshold`` is False, use ``workpiece.to_part()``.
       When ``normalize_windings`` is True (e.g. WavefrontStep),
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
            return Part.from_geometry_multi_face(
                geometry=scaled, size_mm=(w, h)
            )
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
                return Part.from_geometry_multi_face(
                    geometry=geo, size_mm=workpiece.size
                )
            return Part(geometry=geo, size_mm=workpiece.size)

    return None


MAX_VECTOR_TRACE_PIXELS = 16 * 1024 * 1024


def build_part_vector_with_raster_fallback(
    workpiece: WorkPiece,
    pixels_per_mm: tuple[float, float],
    *,
    override_threshold: bool = False,
    threshold: float = 0.5,
    normalize_windings: bool = False,
) -> Part:
    """Build a vector :class:`Part`, rendering the workpiece source to a
    raster surface and tracing it when no vector boundaries are
    available.

    This mirrors the old ``_execute_vector`` pipeline path that
    fell back to render-and-trace when a workpiece had no boundaries
    (e.g. an SVG whose ``pristine_geometry`` is empty).

    Returns a :class:`Part` with at least ``size_mm`` set.  The
    geometry may be empty (``None``) if neither the vector source
    nor the raster trace yields any contours.
    """
    boundaries = workpiece.boundaries
    has_vector = boundaries is not None and not boundaries.is_empty()
    if has_vector and not override_threshold:
        part = build_part_vector(
            workpiece,
            surface=None,
            override_threshold=False,
            normalize_windings=normalize_windings,
        )
        if part is not None and part.has_geometry():
            return part

    size_mm = workpiece.size
    if not size_mm or size_mm[0] <= 0 or size_mm[1] <= 0:
        return Part(size_mm=size_mm)

    target_w = int(size_mm[0] * pixels_per_mm[0])
    target_h = int(size_mm[1] * pixels_per_mm[1])
    num_pixels = target_w * target_h
    if num_pixels > MAX_VECTOR_TRACE_PIXELS:
        scale = (MAX_VECTOR_TRACE_PIXELS / num_pixels) ** 0.5
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)

    if target_w <= 0 or target_h <= 0:
        return Part(size_mm=size_mm)

    surface = workpiece.render_to_pixels(target_w, target_h)
    if surface is None:
        return Part(size_mm=size_mm)

    part = build_part_vector(
        workpiece,
        surface=surface,
        override_threshold=True,
        threshold=threshold,
        normalize_windings=normalize_windings,
    )
    if part is not None:
        return part
    return Part(size_mm=size_mm)


def preprocess_raster_image(
    surface: cairo.ImageSurface,
    *,
    mode: DepthMode,
    invert: bool = False,
    auto_levels: bool = True,
    computed_auto_levels: tuple[int, int] | None = None,
    black_point: int = 0,
    white_point: int = 255,
    threshold: int = 128,
    dither_algorithm: DitherAlgorithm | None = None,
    laser_spot_x_mm: float = 0.1,
    pixels_per_mm_x: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
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
    computed_auto_levels: tuple[int, int] | None = None,
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
    workpiece: WorkPiece,
    pixels_per_mm: tuple[float, float],
    *,
    invert: bool = False,
    max_preview_pixels: int = 512,
) -> tuple[int, int] | None:
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

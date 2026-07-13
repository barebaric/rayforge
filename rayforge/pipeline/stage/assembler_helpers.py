"""
Helper functions for the assembler-based pipeline.

These functions absorb the Part-construction and result-wrapping
logic that currently lives inside each producer's ``run()`` method,
so that the stage can call raygeo assemblers directly without
needing producer class instances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from raygeo.geo import Geometry, Matrix
from raygeo.ops import Ops
from raygeo.ops.assembly import AssemblyResult
from raygeo.ops.part import Part
from raygeo.ops.types import SectionType

from ...core.vectorization_spec import TraceSpec
from ...image.tracing import trace_surface
from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem

if TYPE_CHECKING:
    import cairo

    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser

logger = logging.getLogger(__name__)


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
    """
    s = settings or {}

    spot_x = laser.spot_size_mm[0]
    spot_y = laser.spot_size_mm[1]

    return MachineDefaults(
        kerf_mm=s.get("kerf_mm", spot_x),
        arc_tolerance=s.get("arc_tolerance", 0.03),
        allow_arcs=s.get(
            "machine_supports_arcs", s.get("output_arcs", True)
        ),
        supports_curves=s.get("machine_supports_curves", False),
        line_interval_mm=spot_y,
        step_power=s.get("power", 1.0),
        tool_radius=spot_x / 2.0,
        step_over=spot_x,
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
                final_ops.ops_section_start(section_type, workpiece.uid)
                final_ops.extend(Ops.from_geometry(c))
                final_ops.ops_section_end(section_type)
        else:
            final_ops.ops_section_start(section_type, workpiece.uid)
            if set_power is not None:
                final_ops.set_power(set_power)
            final_ops.extend(result.ops)
            final_ops.ops_section_end(section_type)

    return make_artifact(
        final_ops,
        workpiece,
        generation_id,
        is_vector=is_vector,
        source_dimensions=source_dimensions,
    )

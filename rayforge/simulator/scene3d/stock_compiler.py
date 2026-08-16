"""
Stock mesh compilation (CPU, background thread).

Turns the document's ``StockItem`` definitions into solid prism meshes
suitable for the PBR renderer.  Triangulation and extrusion run in
Rust (:func:`raygeo.mesh.build.build_prism_mesh`); this module only
extracts plain-data specs, validates them and packs the GPU buffers
into :class:`StockLayer` objects.

Mesh layout (all coordinates in world / machine mm):

- Top face on the z=0 engrave plane, CCW-outward normal ``+z``.
- Bottom cap at ``z=-thickness``, flipped winding, normal ``-z``.
- Side walls for every boundary ring (outer contours and inner
  islands), with outward-facing normals.
- UVs are ``world_xy / texture_size_mm`` so one texture repeat always
  spans ``texture_size_mm`` world millimeters (physical density,
  ``GL_REPEAT`` at render time).
"""

from __future__ import annotations

import logging

import numpy as np
from raygeo.mesh.build import build_prism_mesh

from .compiled_scene import StockLayer

logger = logging.getLogger(__name__)

# Top face of the stock sits on the engrave plane (z=0 in world space).
Z_TOP = 0.0

# Fallback thickness when the asset has none configured.
DEFAULT_THICKNESS_MM = 18.0

# Fallback texture tile size when the material has none configured.
DEFAULT_TEXTURE_SIZE_MM = 300.0

# Fallback albedo (linear-ish RGBA) for materials without a color.
DEFAULT_RGBA = (1.0, 1.0, 1.0, 1.0)


def _parse_rgba(color: object) -> tuple[float, float, float, float]:
    """Best-effort parse of a hex color into an RGBA tuple."""
    if not isinstance(color, str) or not color:
        return DEFAULT_RGBA
    from ...core.color import hex_to_rgba

    try:
        rgba = hex_to_rgba(color)
    except ValueError:
        logger.warning("Invalid material color %r, using default.", color)
        return DEFAULT_RGBA
    return (
        float(rgba[0]),
        float(rgba[1]),
        float(rgba[2]),
        float(rgba[3]),
    )


def _parse_rgba_optional(
    color: object,
) -> tuple[float, float, float, float] | None:
    """Parse a hex color, returning None for absent/invalid values."""
    if not isinstance(color, str) or not color:
        return None
    from ...core.color import hex_to_rgba

    try:
        rgba = hex_to_rgba(color)
    except ValueError:
        logger.warning("Invalid tint color %r, ignoring.", color)
        return None
    return (
        float(rgba[0]),
        float(rgba[1]),
        float(rgba[2]),
        float(rgba[3]),
    )


def _positive_float(value: object, default: float) -> float:
    """Coerce *value* to a positive float, falling back to *default*."""
    if not isinstance(value, (int, float, str)):
        return default
    try:
        result = float(value)
    except ValueError:
        return default
    return result if result > 0 else default


def _compile_stock_spec(spec: dict, w2v: np.ndarray) -> StockLayer | None:
    """Compile a single stock spec dict into a mesh layer."""
    outers = [
        ring
        for ring in (
            [(float(x), float(y)) for x, y in r]
            for r in spec.get("outers", [])
        )
        if len(ring) >= 3
    ]
    holes = [
        ring
        for ring in (
            [(float(x), float(y)) for x, y in r] for r in spec.get("holes", [])
        )
        if len(ring) >= 3
    ]
    if not outers:
        return None

    thickness = _positive_float(spec.get("thickness"), DEFAULT_THICKNESS_MM)
    texture_size_mm = _positive_float(
        spec.get("texture_size_mm"), DEFAULT_TEXTURE_SIZE_MM
    )

    pos_parts: list[np.ndarray] = []
    norm_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    base = 0
    for outer in outers:
        try:
            mesh = build_prism_mesh(
                outer,
                holes,
                thickness=thickness,
                uv_scale=texture_size_mm,
                z_top=Z_TOP,
            )
        except ValueError as e:
            logger.warning(
                "Skipping stock ring of %r: %s", spec.get("name"), e
            )
            continue
        pos = np.asarray(mesh.positions, dtype=np.float32)
        if pos.shape[0] == 0:
            continue
        pos_parts.append(pos)
        norm_parts.append(np.asarray(mesh.normals, dtype=np.float32))
        uv_parts.append(np.asarray(mesh.uvs, dtype=np.float32))
        idx_parts.append(np.asarray(mesh.indices, dtype=np.uint32) + base)
        base += pos.shape[0]

    if not idx_parts:
        return None

    return StockLayer(
        positions=np.concatenate(pos_parts),
        normals=np.concatenate(norm_parts),
        uvs=np.concatenate(uv_parts),
        indices=np.concatenate(idx_parts),
        transform=np.asarray(w2v, dtype=np.float32),
        texture_path=spec.get("texture_path"),
        texture_size_mm=texture_size_mm,
        roughness=float(spec.get("roughness") or 0.8),
        metallic=float(spec.get("metallic") or 0.0),
        fallback_rgba=_parse_rgba(spec.get("color")),
        tint_rgba=_parse_rgba_optional(spec.get("tint")),
    )


# ── Public API ───────────────────────────────────────────────────


def compile_stock_layers(
    stock_specs: list[dict],
    world_to_visual: np.ndarray,
) -> list[StockLayer]:
    """Compile stock specs into a list of prism mesh layers.

    ``stock_specs`` are plain-data dicts produced by the scene presenter
    (geometry rings in world mm, thickness, material parameters) so the
    CPU-heavy triangulation runs on the background compile thread.
    """
    if not stock_specs:
        return []
    w2v = np.asarray(world_to_visual, dtype=np.float32)
    layers: list[StockLayer] = []
    for spec in stock_specs:
        try:
            layer = _compile_stock_spec(spec, w2v)
        except (ValueError, TypeError) as e:
            logger.warning(
                "Failed to compile stock %r: %s", spec.get("name"), e
            )
            continue
        if layer is not None:
            layers.append(layer)
    return layers

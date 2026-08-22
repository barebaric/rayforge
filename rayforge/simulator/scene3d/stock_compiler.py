"""
Stock mesh compilation (CPU, background thread).

Turns the document's stock definitions into solid meshes suitable for
the PBR renderer.  Flat stocks are prism meshes built in Rust
(:func:`raygeo.mesh.build.build_prism_mesh`); rotary stocks are
cylinder shells built in numpy.  This module only extracts plain-data
specs, validates them and packs the GPU buffers into
:class:`StockLayer` objects.

Mesh layout (all coordinates in world / machine mm):

- Flat: bottom face on the bed (z=0), top face at ``z=+thickness``.
  Top-face normal ``+z``, bottom cap flipped winding, normal ``-z``.
  Side walls for every boundary ring (outer contours and inner
  islands), with outward-facing normals.
- Rotary: single-layer cylindrical shell (no caps) along the local X
  axis (the chuck axis), spanning ``0..length`` axially, positioned by
  the per-frame cylinder kinematics at render time.
- UVs are ``world_xy / texture_size_mm`` (flat) or
  ``axial/circumference distance / texture_size_mm`` (rotary) so one
  texture repeat always spans ``texture_size_mm`` world millimeters
  (physical density, ``GL_REPEAT`` at render time).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from raygeo.mesh.build import build_prism_mesh
from raygeo.ops.material.grid import compute_power_uvs

from .compiled_scene import StockLayer

if TYPE_CHECKING:
    from raygeo.compressed_array import CompressedArray

logger = logging.getLogger(__name__)

# Fallback thickness when the asset has none configured.
DEFAULT_THICKNESS_MM = 18.0

# Fallback texture tile size when the material has none configured.
DEFAULT_TEXTURE_SIZE_MM = 300.0

# Fallback albedo (linear-ish RGBA) for materials without a color.
DEFAULT_RGBA = (1.0, 1.0, 1.0, 1.0)

# Cylinder shell tessellation: angular segments and axial segments.
CYLINDER_RINGS = 48
CYLINDER_LENGTH_SEGMENTS = 16


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


def _build_cylinder_shell(
    diameter: float,
    length: float,
    texture_size_mm: float,
    rings: int = CYLINDER_RINGS,
    length_segments: int = CYLINDER_LENGTH_SEGMENTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a solid cylinder mesh in chuck-local space.

    The axis runs along local X (the chuck axis), spanning ``0..length``
    axially with the cross-section centered on the Y/Z origin — the
    same frame the wireframe cylinder and the rotary texture meshes
    use, so the per-frame cylinder kinematics place it.  The lateral
    surface is a shell whose angular seam is duplicated with continued
    UVs so the texture wraps physically via ``GL_REPEAT``; both ends
    are closed with flat cap fans so the stock reads as a solid rod
    instead of a pipe.

    UV mapping: U follows the circumference and V follows the axis, so
    the source textures' vertical grain (their V direction) runs along
    the cylinder instead of around it.  The caps sample the texture
    at their axial stripe (the cross-section at each end).

    Returns ``(positions, normals, uvs, indices)`` as flat float32 /
    uint32 arrays.
    """
    radius = diameter / 2.0
    circumference = math.pi * diameter

    i_vals = np.linspace(0.0, 1.0, length_segments + 1, dtype=np.float64)
    j_vals = np.linspace(0.0, 1.0, rings + 1, dtype=np.float64)
    ii, jj = np.meshgrid(i_vals, j_vals, indexing="ij")

    # The angular parametrization starts at pi (the bottom of the
    # roller) so the duplicated seam sits opposite the beam-home
    # direction: the burn grid's unrolled domain is centered on the
    # machine origin (y = 0 at the top), and a power-uv v of 0..1
    # then spans it without crossing the texture cut mid-surface.
    theta = math.pi + 2.0 * math.pi * jj
    x = ii * length
    y = radius * np.sin(theta)
    z = radius * np.cos(theta)

    positions = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1).astype(
        np.float32
    )
    normals = np.stack(
        [
            np.zeros(theta.size),
            np.sin(theta).ravel(),
            np.cos(theta).ravel(),
        ],
        axis=-1,
    ).astype(np.float32)
    uvs = np.stack(
        [
            (jj * circumference / texture_size_mm).ravel(),
            (x / texture_size_mm).ravel(),
        ],
        axis=-1,
    ).astype(np.float32)

    v = (
        np.arange(length_segments + 1, dtype=np.int64)[:, None] * (rings + 1)
        + np.arange(rings + 1, dtype=np.int64)[None, :]
    )
    tri1 = np.stack([v[:-1, :-1], v[1:, :-1], v[1:, 1:]], axis=-1).reshape(-1)
    tri2 = np.stack([v[:-1, :-1], v[1:, 1:], v[:-1, 1:]], axis=-1).reshape(-1)
    side_indices = np.concatenate([tri1, tri2])

    # Cap fans: one center vertex per end, reusing the ring vertices
    # of the first/last axial segment.  Normals point along ±X so the
    # flat ends light correctly.
    near_center = positions.shape[0]
    far_center = near_center + 1
    centers = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]], dtype=np.float32)
    center_normals = np.array(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )
    center_uvs = np.array(
        [[0.0, 0.0], [0.0, length / texture_size_mm]], dtype=np.float32
    )

    ring_near = v[0, :rings]
    ring_near_wrap = np.roll(ring_near, -1)
    ring_far = v[length_segments, :rings]
    ring_far_wrap = np.roll(ring_far, -1)

    near_fan = np.stack(
        [
            np.full(rings, near_center),
            ring_near,
            ring_near_wrap,
        ],
        axis=-1,
    ).reshape(-1)
    far_fan = np.stack(
        [
            np.full(rings, far_center),
            ring_far_wrap,
            ring_far,
        ],
        axis=-1,
    ).reshape(-1)

    positions = np.concatenate([positions, centers])
    normals = np.concatenate([normals, center_normals])
    uvs = np.concatenate([uvs, center_uvs])
    indices = np.concatenate([side_indices, near_fan, far_fan]).astype(
        np.uint32
    )
    return positions, normals, uvs, indices


def _rotary_power_uvs(
    diameter: float,
    length: float,
    power_grid: tuple[
        tuple[float, float], tuple[float, float], tuple[int, int]
    ],
    rings: int = CYLINDER_RINGS,
    length_segments: int = CYLINDER_LENGTH_SEGMENTS,
) -> np.ndarray:
    """Power UVs for a cylinder shell built by ``_build_cylinder_shell``.

    Derived from the same parametrization as the mesh so the vertex
    order matches exactly (side vertices, then the two cap centers).
    U maps axial position through the burn grid like the flat path;
    V follows the angular parameter, with the duplicated seam mapping
    to 0 and 1 so no quad interpolates across it.
    """
    origin_mm, px_per_mm, size_px = power_grid
    i_vals = np.linspace(0.0, 1.0, length_segments + 1, dtype=np.float64)
    j_vals = np.linspace(0.0, 1.0, rings + 1, dtype=np.float64)
    ii, jj = np.meshgrid(i_vals, j_vals, indexing="ij")
    u = ((ii * length - origin_mm[0]) * px_per_mm[0]) / size_px[0]
    side = np.stack([u.ravel(), jj.ravel()], axis=-1).astype(np.float32)
    caps = np.zeros((2, 2), dtype=np.float32)
    return np.concatenate([side, caps], axis=0)


class _BurnData(NamedTuple):
    """A stock's folded burn surface map and its world placement."""

    surface_map: CompressedArray
    size_px: tuple[int, int]
    aabb: tuple[float, float, float, float]
    power_grid: tuple[
        tuple[float, float], tuple[float, float], tuple[int, int]
    ]


def _parse_burn_spec(spec: dict) -> _BurnData | None:
    """Parse a stock spec's optional ``burn`` entry.

    The presenter attaches it when the material fold produced a burn
    surface map for this stock:
    ``{"surface_map": CompressedArray, "origin_mm": (x, y),
    "px_per_mm": (x, y), "size_px": (w, h)}``.

    Returns the parsed burn data (with the grid's world-mm AABB and
    the ``power_grid`` argument for ``build_prism_mesh``), or ``None``
    when the entry is absent or invalid.
    """
    burn = spec.get("burn")
    if not isinstance(burn, dict):
        return None
    surface_map = burn.get("surface_map")
    origin_mm = burn.get("origin_mm")
    px_per_mm = burn.get("px_per_mm")
    size_px = burn.get("size_px")
    if (
        surface_map is None
        or not isinstance(origin_mm, tuple)
        or not isinstance(px_per_mm, tuple)
        or not isinstance(size_px, tuple)
        or len(origin_mm) != 2
        or len(px_per_mm) != 2
        or len(size_px) != 2
        or px_per_mm[0] <= 0
        or px_per_mm[1] <= 0
        or size_px[0] < 1
        or size_px[1] < 1
    ):
        logger.warning(
            "Ignoring invalid burn entry of stock %r", spec.get("name")
        )
        return None
    (ox, oy), (ppm_x, ppm_y), (w_px, h_px) = (
        origin_mm,
        px_per_mm,
        size_px,
    )
    aabb = (
        float(ox),
        float(oy),
        float(ox) + float(w_px) / float(ppm_x),
        float(oy) + float(h_px) / float(ppm_y),
    )
    return _BurnData(
        surface_map=surface_map,
        size_px=(int(w_px), int(h_px)),
        aabb=aabb,
        power_grid=(origin_mm, px_per_mm, size_px),
    )


def _compile_stock_spec(
    spec: dict, stock_w2v: np.ndarray
) -> StockLayer | None:
    """Compile a single stock spec dict into a mesh layer."""
    texture_size_mm = _positive_float(
        spec.get("texture_size_mm"), DEFAULT_TEXTURE_SIZE_MM
    )
    roughness = float(spec.get("roughness") or 0.8)
    metallic = float(spec.get("metallic") or 0.0)
    fallback_rgba = _parse_rgba(spec.get("color"))
    tint_rgba = _parse_rgba_optional(spec.get("tint"))

    if spec.get("kind") == "rotary":
        return _compile_rotary_stock_spec(
            spec,
            texture_size_mm=texture_size_mm,
            roughness=roughness,
            metallic=metallic,
            fallback_rgba=fallback_rgba,
            tint_rgba=tint_rgba,
            stock_w2v=stock_w2v,
        )

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
    burn = _parse_burn_spec(spec)

    pos_parts: list[np.ndarray] = []
    norm_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    power_uv_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    base = 0
    for outer in outers:
        try:
            mesh = build_prism_mesh(
                outer,
                holes,
                thickness=thickness,
                uv_scale=texture_size_mm,
                z_top=thickness,
            )
        except ValueError as e:
            logger.warning(
                "Skipping stock ring of %r: %s", spec.get("name"), e
            )
            continue
        pos = np.asarray(mesh.positions, dtype=np.float32)
        if pos.shape[0] == 0:
            continue
        norm = np.asarray(mesh.normals, dtype=np.float32)
        uvs = np.asarray(mesh.uvs, dtype=np.float32)
        ring_idx = np.asarray(mesh.indices, dtype=np.uint32)
        puv: np.ndarray | None = None
        if burn is not None:
            puv = np.asarray(
                compute_power_uvs(
                    pos.reshape(-1, 3),
                    burn.power_grid[0],
                    burn.power_grid[1],
                    burn.power_grid[2],
                ),
                dtype=np.float32,
            )
        pos_parts.append(pos)
        norm_parts.append(norm)
        uv_parts.append(uvs)
        if puv is not None:
            power_uv_parts.append(puv)
        idx_parts.append(ring_idx + base)
        base += pos.shape[0]

    if not idx_parts:
        return None

    if burn is not None and power_uv_parts:
        arr = burn.surface_map.to_numpy()
        logger.info(
            "Stock %r: burn-in surface map applied (%d/%d px burned, aabb=%s)",
            spec.get("name"),
            int((arr > 0).sum()),
            arr.size,
            burn.aabb,
        )

    return StockLayer(
        positions=np.concatenate(pos_parts),
        normals=np.concatenate(norm_parts),
        uvs=np.concatenate(uv_parts),
        indices=np.concatenate(idx_parts),
        transform=np.asarray(stock_w2v, dtype=np.float32),
        texture_path=spec.get("texture_path"),
        texture_size_mm=texture_size_mm,
        roughness=roughness,
        metallic=metallic,
        fallback_rgba=fallback_rgba,
        tint_rgba=tint_rgba,
        power_texture=burn.surface_map if burn is not None else None,
        power_size_px=burn.size_px if burn is not None else None,
        power_aabb=burn.aabb if burn is not None else None,
        power_uvs=(
            np.concatenate(power_uv_parts)
            if power_uv_parts
            else np.empty((0, 2), dtype=np.float32)
        ),
    )


def _compile_rotary_stock_spec(
    spec: dict,
    *,
    texture_size_mm: float,
    roughness: float,
    metallic: float,
    fallback_rgba: tuple[float, float, float, float],
    tint_rgba: tuple[float, float, float, float] | None,
    stock_w2v: np.ndarray,
) -> StockLayer | None:
    """Compile a rotary stock spec into a cylinder shell layer."""
    diameter = _positive_float(spec.get("diameter"), 0.0)
    length = _positive_float(spec.get("length"), 0.0)
    if diameter <= 0 or length <= 0:
        return None

    positions, normals, uvs, indices = _build_cylinder_shell(
        diameter, length, texture_size_mm
    )
    burn = _parse_burn_spec(spec)
    power_uvs: np.ndarray = np.empty((0, 2), dtype=np.float32)
    if burn is not None:
        power_uvs = _rotary_power_uvs(diameter, length, burn.power_grid)
    return StockLayer(
        positions=positions,
        normals=normals,
        uvs=uvs,
        indices=indices,
        transform=np.asarray(stock_w2v, dtype=np.float32),
        texture_path=spec.get("texture_path"),
        texture_size_mm=texture_size_mm,
        roughness=roughness,
        metallic=metallic,
        fallback_rgba=fallback_rgba,
        tint_rgba=tint_rgba,
        is_rotary=True,
        power_texture=(burn.surface_map if burn is not None else None),
        power_size_px=(burn.size_px if burn is not None else None),
        power_aabb=(burn.aabb if burn is not None else None),
        power_uvs=power_uvs,
    )


# ── Public API ───────────────────────────────────────────────────


def stock_burn_aabbs(
    stock_specs: list[dict],
) -> list[tuple[float, float, float, float]]:
    """World-mm AABBs of every spec's burn surface map.

    Used by the scene compiler to decide which LUT engrave quads are
    superseded by a stock's burned-in charring.
    """
    aabbs: list[tuple[float, float, float, float]] = []
    for spec in stock_specs:
        burn = _parse_burn_spec(spec)
        if burn is not None:
            aabbs.append(burn.aabb)
    return aabbs


def compile_stock_layers(
    stock_specs: list[dict],
    stock_world_to_visual: np.ndarray,
) -> list[StockLayer]:
    """Compile stock specs into a list of prism mesh layers.

    ``stock_specs`` are plain-data dicts produced by the scene presenter
    (geometry rings in world mm, thickness, material parameters) so the
    CPU-heavy triangulation runs on the background compile thread.

    ``stock_world_to_visual`` is the bed-anchored world->visual matrix
    (Z=0); stock meshes always sit on the bed regardless of WCS Z
    offsets or no-Z lifts.
    """
    if not stock_specs:
        return []
    stock_w2v = np.asarray(stock_world_to_visual, dtype=np.float32)
    layers: list[StockLayer] = []
    for spec in stock_specs:
        try:
            layer = _compile_stock_spec(spec, stock_w2v)
        except (ValueError, TypeError) as e:
            logger.warning(
                "Failed to compile stock %r: %s", spec.get("name"), e
            )
            continue
        if layer is not None:
            layers.append(layer)
    return layers

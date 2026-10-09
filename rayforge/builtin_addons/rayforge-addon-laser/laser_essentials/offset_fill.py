"""Contour-parallel fill rings for the Offset Fill step."""

from __future__ import annotations

from raygeo.geo import Geometry

MAX_RINGS = 10000


def _closed_contours(geometry: Geometry) -> Geometry:
    closed = Geometry()
    for contour in geometry.split_into_contours():
        if contour.is_closed():
            closed.extend(contour)
    return closed


def offset_fill_rings(
    geometry: Geometry,
    step: float,
    offset: float = 0.0,
    max_rings: int = MAX_RINGS,
) -> list[Geometry]:
    """Return the fill rings of the closed contours in *geometry*.

    Ring ``k`` is the contour group (outlines and holes together)
    offset inward by ``offset + k * step``. Rings are returned
    outermost first; open contours are ignored. Generation stops when
    an offset leaves nothing or after *max_rings* rings.
    """
    if step <= 0:
        raise ValueError("Offset fill step must be positive")
    closed = _closed_contours(geometry)
    rings: list[Geometry] = []
    if closed.is_empty():
        return rings
    while len(rings) < max_rings:
        distance = offset + len(rings) * step
        ring = closed.copy()
        if distance > 0:
            ring.grow(-distance)
        if ring.is_empty():
            break
        rings.append(ring)
    return rings


def offset_fill_faces(
    geometry: Geometry,
    step: float,
    offset: float = 0.0,
    inside_out: bool = False,
    max_rings: int = MAX_RINGS,
) -> list[Geometry]:
    """Return the fill rings grouped per shape, one shape after the other.

    Each shape (a connected component of *geometry*) contributes its
    rings in outside→inside order; ``inside_out`` reverses the order
    per shape. Used for cutting order decisions, so every shape is
    filled completely before the next one starts.
    """
    faces: list[Geometry] = []
    for shape in geometry.split_into_components():
        rings = offset_fill_rings(shape, step, offset, max_rings)
        if inside_out:
            rings.reverse()
        faces.extend(rings)
    return faces

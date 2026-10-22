"""Contour-parallel fill rings for the Offset Fill step."""

from __future__ import annotations

from raygeo.geo import Geometry
from raygeo.geo.algo.offset import concentric_offsets

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

    The first ring is the contour group (outlines and holes together)
    offset inward by ``offset``; every next ring lies ``step`` further
    inside. Rings are returned outermost first; open contours are
    ignored. Generation stops when an offset leaves nothing or after
    *max_rings* rings.
    """
    if step <= 0:
        raise ValueError("Offset fill step must be positive")
    first = _closed_contours(geometry)
    if offset > 0 and not first.is_empty():
        first.grow(-offset)
    if first.is_empty() or max_rings < 1:
        return []
    return [first, *concentric_offsets(first, step, max_rings - 1, 0.0)]

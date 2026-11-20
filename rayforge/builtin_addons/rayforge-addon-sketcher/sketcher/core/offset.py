"""Offset-command helpers over the compiled raygeo model."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from raygeo.sketch.contour import offset_outline, outline_frame
from raygeo.sketch.entities import Polygon as PolygonEntity

from .types import EntityID

__all__ = ["offset_outline", "outline_item"]


def outline_item(
    vertices: list[tuple[float, float]],
    closed: bool,
    allocate_id: Callable[[], EntityID],
    holes: Sequence[Sequence[tuple[float, float]]] | None = None,
) -> tuple:
    """Builds the frame points and polygon for a world outline."""
    center, handle = outline_frame(vertices, closed)
    return PolygonEntity.build(
        allocate_id(),
        allocate_id(),
        allocate_id(),
        center,
        handle,
        vertices,
        closed=closed,
        world_holes=list(holes) if holes else None,
    )

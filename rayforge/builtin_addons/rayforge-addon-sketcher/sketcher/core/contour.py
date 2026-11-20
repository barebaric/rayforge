"""Preprocessing that turns a sketch selection into offsettable items.

The selection is partitioned into connected components (shared or
coincident endpoints). Each component becomes exactly one offsettable
item: an entity that offsets on its own (polymorphic
``as_offset_item``) passes through as itself and is updated in place;
chains of endpoint-bearing entities are sampled into a single
:class:`PolygonOutline`, whose offset replaces them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from raygeo.sketch.contour import (
    chain_vertices,
    partition_components,
    walk_component,
)
from raygeo.sketch.entities import Entity, PolygonOutline

if TYPE_CHECKING:
    from .sketch import Sketch

logger = logging.getLogger(__name__)

_SAMPLE_TOLERANCE = 0.02

OffsetItem = Entity | PolygonOutline


def coincident_pairs(
    sketch: Sketch, entity_ids: list[int]
) -> list[tuple[int, int]]:
    """Collects the coincident point pairs touched by the selection."""
    pairs: set[tuple[int, int]] = set()
    registry = sketch.registry
    for eid in entity_ids:
        entity = registry.get_entity(eid)
        if entity is None:
            continue
        for pid in entity.get_endpoint_ids()[:2]:
            for other in sketch.get_coincident_points(pid):
                if other > pid:
                    pairs.add((pid, other))
    return sorted(pairs)


def build_offset_items(
    sketch: Sketch, entity_ids: list[int]
) -> list[OffsetItem] | None:
    """Partitions the selection into offsettable items."""
    registry = sketch.registry
    items: list[OffsetItem] = []
    chainable: list[int] = []
    for eid in entity_ids:
        entity = registry.get_entity(eid)
        if entity is None:
            continue
        if len(entity.get_endpoint_ids()) == 2:
            chainable.append(eid)
            continue
        item = entity.as_offset_item(sketch)
        if item is not None:
            items.append(item)

    if chainable:
        pairs = coincident_pairs(sketch, chainable)
        components = partition_components(registry, chainable, pairs)
        if components is None:
            return None
        for component in components:
            if len(component) == 1:
                entity = registry.get_entity(component[0])
                if entity is not None:
                    item = entity.as_offset_item(sketch)
                    if item is not None:
                        items.append(item)
                        continue
            edges, closed = walk_component(registry, component, pairs)
            vertices = chain_vertices(
                registry, edges, closed, _SAMPLE_TOLERANCE
            )
            if not closed and vertices:
                vertices = _orient_to_first_endpoint(
                    sketch, chainable, component, vertices
                )
            if len(vertices) >= 2:
                items.append(
                    PolygonOutline(
                        vertices,
                        closed,
                        source_ids=[eid for eid, _ in edges],
                    )
                )
    return items or None


def _orient_to_first_endpoint(
    sketch: Sketch,
    selection_order: list[int],
    component: list[int],
    vertices: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Keeps the chain start tied to the first selected entity."""
    component_ids = set(component)
    first_eid = next(
        (eid for eid in selection_order if eid in component_ids), None
    )
    if first_eid is None:
        return vertices
    entity = sketch.registry.get_entity(first_eid)
    if entity is None:
        return vertices
    endpoints = entity.get_endpoint_ids()
    if not endpoints:
        return vertices
    start = sketch.registry.get_point(endpoints[0])
    if vertices[-1] == (start.x, start.y):
        return list(reversed(vertices))
    return vertices

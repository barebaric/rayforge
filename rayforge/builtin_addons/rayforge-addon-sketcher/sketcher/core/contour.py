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
from collections import defaultdict
from typing import TYPE_CHECKING

from .entities import Entity, PolygonOutline

if TYPE_CHECKING:
    from .sketch import Sketch

logger = logging.getLogger(__name__)

_JUNCTION_EPS = 1e-6
_SAMPLE_TOLERANCE = 0.02

OffsetItem = Entity | PolygonOutline


def build_offset_items(
    sketch: Sketch, entity_ids: list[int]
) -> list[OffsetItem] | None:
    """
    Partitions the selection into offsettable items, one per connected
    component.

    Entities that offset on their own (lone circles, arcs, ellipses,
    polygons) pass through polymorphically; connected chains are
    sampled into outlines.

    Returns None when the selection cannot be offset: it contains no
    offsettable geometry, or a component branches at a junction with
    more than two connected segments.
    """
    registry = sketch.registry
    items: list[OffsetItem] = []
    chainable: list[Entity] = []
    for eid in entity_ids:
        entity = registry.get_entity(eid)
        if entity is None:
            continue
        if len(entity.get_endpoint_ids()) == 2:
            chainable.append(entity)
            continue
        item = entity.as_offset_item(sketch)
        if item is not None:
            items.append(item)

    if chainable:
        components = _partition_components(sketch, chainable)
        if components is None:
            return None
        for component in components:
            if len(component) == 1:
                item = component[0].as_offset_item(sketch)
                if item is not None:
                    items.append(item)
                    continue
            edges, closed = _walk_component(sketch, component)
            vertices = _chain_vertices(registry, edges, closed)
            if len(vertices) >= 2:
                items.append(
                    PolygonOutline(
                        vertices,
                        closed,
                        source_ids=[entity.id for entity, _ in edges],
                    )
                )

    return items or None


def _partition_components(
    sketch: Sketch, entities: list[Entity]
) -> list[list[Entity]] | None:
    """
    Groups endpoint-bearing entities into connected components,
    treating coincident endpoints as the same node. Returns None if
    any junction connects more than two segments.
    """
    node_of: dict[int, int] = {}
    edge_nodes: list[tuple[int, int]] = []
    node_edges: dict[int, list[int]] = defaultdict(list)
    for edge_idx, entity in enumerate(entities):
        p1, p2 = entity.get_endpoint_ids()[:2]
        n1 = _node_for(sketch, node_of, p1)
        n2 = _node_for(sketch, node_of, p2)
        edge_nodes.append((n1, n2))
        node_edges[n1].append(edge_idx)
        node_edges[n2].append(edge_idx)

    if any(len(edges) > 2 for edges in node_edges.values()):
        logger.warning(
            "Offset selection branches at a junction; "
            "offset each contour separately."
        )
        return None

    visited = [False] * len(entities)
    components: list[list[Entity]] = []
    for start in range(len(entities)):
        if visited[start]:
            continue
        stack = [start]
        component: list[Entity] = []
        while stack:
            edge_idx = stack.pop()
            if visited[edge_idx]:
                continue
            visited[edge_idx] = True
            component.append(entities[edge_idx])
            stack.extend(
                other
                for node in edge_nodes[edge_idx]
                for other in node_edges[node]
                if not visited[other]
            )
        components.append(component)
    return components


def _walk_component(
    sketch: Sketch, entities: list[Entity]
) -> tuple[list[tuple[Entity, bool]], bool]:
    """
    Orders one connected component's entities into an oriented chain,
    starting at a degree-1 endpoint (open chains) or at an arbitrary
    edge (closed loops). Returns (ordered (entity, forward) edges,
    is_closed).
    """
    node_of: dict[int, int] = {}
    nodes = [
        (
            _node_for(sketch, node_of, entity.get_endpoint_ids()[0]),
            _node_for(sketch, node_of, entity.get_endpoint_ids()[1]),
        )
        for entity in entities
    ]
    node_edges: dict[int, list[int]] = defaultdict(list)
    for edge_idx, (n1, n2) in enumerate(nodes):
        node_edges[n1].append(edge_idx)
        node_edges[n2].append(edge_idx)

    start_node = next(
        (n for n, edges in node_edges.items() if len(edges) == 1),
        nodes[0][0],
    )
    closed = len(node_edges[start_node]) == 2

    used = [False] * len(entities)
    ordered: list[tuple[Entity, bool]] = []
    current_node = start_node
    while True:
        edge_idx = next(
            (i for i in node_edges[current_node] if not used[i]), None
        )
        if edge_idx is None:
            break
        n1, n2 = nodes[edge_idx]
        fwd = n1 == current_node
        ordered.append((entities[edge_idx], fwd))
        used[edge_idx] = True
        current_node = n2 if fwd else n1
    return ordered, closed


def _node_for(sketch: Sketch, node_of: dict[int, int], pid: int) -> int:
    """Maps a point ID to its coincident-group representative."""
    if pid not in node_of:
        node_of[pid] = min(sketch.get_coincident_points(pid))
    return node_of[pid]


def _chain_vertices(
    registry,
    edges: list[tuple[Entity, bool]],
    closed: bool,
) -> list[tuple[float, float]]:
    """Samples an ordered chain into one deduplicated vertex list."""
    vertices: list[tuple[float, float]] = []
    for entity, fwd in edges:
        for x, y in _entity_samples(registry, entity, fwd):
            if vertices:
                last = vertices[-1]
                if (
                    abs(last[0] - x) < _JUNCTION_EPS
                    and abs(last[1] - y) < _JUNCTION_EPS
                ):
                    continue
            vertices.append((x, y))
    if closed and len(vertices) > 1:
        first = vertices[0]
        last = vertices[-1]
        if (
            abs(first[0] - last[0]) < _JUNCTION_EPS
            and abs(first[1] - last[1]) < _JUNCTION_EPS
        ):
            vertices.pop()
    return vertices


def _entity_samples(
    registry, entity: Entity, fwd: bool
) -> list[tuple[float, float]]:
    """Samples one entity along its traversal direction."""
    points = entity.to_polyline(registry, _SAMPLE_TOLERANCE)
    if not fwd:
        points = list(reversed(points))
    return points

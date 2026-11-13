"""Preprocessing that turns a sketch selection into offsettable items.

The selection is partitioned into connected components (shared or
coincident endpoints). Each component becomes exactly one offsettable
item: an entity that offsets on its own (circles, arcs, ellipses,
polygons) passes through as itself and is updated in place; chains of
endpoint-bearing entities are sampled into a single
:class:`PolygonOutline`, whose offset replaces them.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .offset import PolygonOutline

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

    from .sketch import Sketch

logger = logging.getLogger(__name__)

_JUNCTION_EPS = 1e-6
_SAMPLE_TOLERANCE = 0.02

OffsetItem = Any


def build_offset_items(
    sketch: Sketch, entity_ids: list[int]
) -> list[OffsetItem] | None:
    """
    Partitions the selection into offsettable items, one per connected
    component.

    Entities that offset on their own (lone circles, arcs, ellipses,
    polygons) pass through as themselves; connected chains are
    sampled into outlines.

    Returns None when the selection cannot be offset: it contains no
    offsettable geometry, or a component branches at a junction with
    more than two connected segments.
    """
    registry = sketch.registry
    items: list[OffsetItem] = []
    chainable: list[Any] = []
    for eid in entity_ids:
        entity = registry.get_entity_object(eid)
        if entity is None:
            continue
        endpoints = registry.entity_endpoint_ids(eid)
        if endpoints and len(endpoints) == 2:
            chainable.append(entity)
            continue
        item = _as_offset_item(registry, entity)
        if item is not None:
            items.append(item)

    if chainable:
        components = partition_components(sketch, chainable)
        if components is None:
            return None
        for component in components:
            if len(component) == 1:
                item = _as_offset_item(registry, component[0])
                if item is not None:
                    items.append(item)
                    continue
            edges, closed = walk_component(sketch, component)
            vertices = chain_vertices(registry, edges, closed)
            if len(vertices) >= 2:
                items.append(
                    PolygonOutline(
                        vertices,
                        closed,
                        source_ids=[entity.id for entity, _ in edges],
                    )
                )

    return items or None


def _as_offset_item(registry, entity: Any) -> OffsetItem | None:
    """The offsettable item an entity forms when it is offset on its
    own, or None when it cannot stand alone (chain members fall back
    to outline sampling; non-offsettable types are skipped)."""
    kind = entity.type_name()
    if kind in ("circle", "arc", "ellipse"):
        return entity
    if kind == "polygon":
        from .entity_ops import polygon_world_vertices

        if entity.closed:
            return entity
        return PolygonOutline(
            polygon_world_vertices(registry, entity),
            False,
            source_ids=[entity.id],
        )
    return None


def partition_components(
    sketch: Sketch, entities: list[Any]
) -> list[list[Any]] | None:
    """
    Groups endpoint-bearing entities into connected components,
    treating coincident endpoints as the same node. Returns None if
    any junction connects more than two segments.
    """
    node_of: dict[int, int] = {}
    edge_nodes: list[tuple[int, int]] = []
    node_edges: dict[int, list[int]] = defaultdict(list)
    registry = sketch.registry
    for edge_idx, entity in enumerate(entities):
        p1, p2 = registry.entity_endpoint_ids(entity.id)[:2]
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
    components: list[list[Any]] = []
    for start in range(len(entities)):
        if visited[start]:
            continue
        stack = [start]
        component: list[Any] = []
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


def walk_component(
    sketch: Sketch, entities: list[Any]
) -> tuple[list[tuple[Any, bool]], bool]:
    """
    Orders one connected component's entities into an oriented chain,
    starting at a degree-1 endpoint (open chains) or at an arbitrary
    edge (closed loops). Returns (ordered (entity, forward) edges,
    is_closed).
    """
    node_of: dict[int, int] = {}
    registry = sketch.registry
    nodes = [
        (
            _node_for(sketch, node_of, registry.entity_endpoint_ids(e.id)[0]),
            _node_for(sketch, node_of, registry.entity_endpoint_ids(e.id)[1]),
        )
        for e in entities
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
    ordered: list[tuple[Any, bool]] = []
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


def chain_vertices(
    registry: EntityRegistry,
    edges: list[tuple[Any, bool]],
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
    registry: EntityRegistry, entity: Any, fwd: bool
) -> list[tuple[float, float]]:
    """Samples one entity along its traversal direction."""
    points = registry.to_polyline(entity.id, _SAMPLE_TOLERANCE)
    if not fwd:
        points = list(reversed(points))
    return points

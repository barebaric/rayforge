"""Boolean operations over a sketch selection.

The selection is partitioned into closed candidate regions: standalone
closed shapes (circles, ellipses, closed polygons) and closed chains of
endpoint-bearing entities. Each region is sampled into rings in world
coordinates and passed to raygeo's Clipper2-backed boolean functions,
which return a flat ring list using the winding convention (outer
contours CCW, hole contours CW). Results are regrouped into solids —
an outer ring plus its holes — one :class:`BooleanSolid` each, ready
to be baked into a single multi-ring PolygonEntity.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from raygeo.geo.shape.polygon import (
    get_polygon_signed_area,
    get_polygons_group_difference,
    get_polygons_group_intersection,
    get_polygons_union,
)

from .contour import chain_vertices, partition_components, walk_component
from .entities.polygon import (
    PolygonEntity,
    group_rings_into_solids,
    rings_to_geometry,
)

if TYPE_CHECKING:
    from .sketch import Sketch

logger = logging.getLogger(__name__)

_SAMPLE_TOLERANCE = 0.02
_MIN_RING_AREA = 1e-9


class BooleanOp(Enum):
    UNION = "union"
    DIFFERENCE = "difference"
    INTERSECTION = "intersection"
    EXCLUDE = "exclude"


@dataclass
class BooleanRegion:
    """
    One closed candidate shape: the outer ring plus optional hole
    rings (world coordinates, winding normalized), the contributing
    source entity IDs and the stacking index (registry order) used to
    order difference operands.
    """

    rings: list[list[tuple[float, float]]]
    entity_ids: list[int]
    z: int


@dataclass
class BooleanSolid:
    """One result piece: outer ring (CCW) plus hole rings (CW)."""

    outer: list[tuple[float, float]]
    holes: list[list[tuple[float, float]]] = field(default_factory=list)


def build_boolean_regions(
    sketch: Sketch, entity_ids: list[int]
) -> list[BooleanRegion] | None:
    """
    Partitions the selection into closed candidate regions.

    Standalone closed shapes pass through polymorphically; closed
    chains of endpoint-bearing entities are sampled into a single
    outer ring. Construction geometry and array copies are skipped.

    Returns None when fewer than two closed regions can be built
    (open chains, branching junctions, or an insufficient selection).
    """
    registry = sketch.registry
    z_of = {e.id: i for i, e in enumerate(registry.entities)}
    regions: list[BooleanRegion] = []
    chainable: list = []

    for eid in entity_ids:
        entity = registry.get_entity(eid)
        if entity is None or entity.construction or entity.array_copy:
            continue
        if entity.type == "text_box":
            logger.warning(
                "Text boxes cannot participate in boolean operations."
            )
            continue
        if len(entity.get_endpoint_ids()) == 2:
            chainable.append(entity)
            continue
        rings = _standalone_rings(registry, entity)
        if not rings:
            logger.warning(
                "Skipping %s: it does not bound a closed region.",
                entity.type,
            )
            continue
        regions.append(
            BooleanRegion(rings, [entity.id], z_of.get(entity.id, 0))
        )

    if chainable:
        components = partition_components(sketch, chainable)
        if components is None:
            logger.warning(
                "Selection branches at a junction; boolean operations "
                "need simple closed contours."
            )
            return None
        for component in components:
            edges, closed = walk_component(sketch, component)
            if not closed:
                logger.warning(
                    "Selection contains an open contour; boolean "
                    "operations need closed regions."
                )
                return None
            vertices = chain_vertices(registry, edges, closed)
            if len(vertices) < 3:
                logger.warning("Selection contains a degenerate contour.")
                return None
            member_ids = [entity.id for entity, _ in edges]
            regions.append(
                BooleanRegion(
                    [vertices],
                    member_ids,
                    max(z_of.get(eid, 0) for eid in member_ids),
                )
            )

    if len(regions) < 2:
        logger.warning("Boolean operations need at least two closed shapes.")
        return None
    return regions


def _standalone_rings(registry, entity) -> list | None:
    """Samples a standalone closed entity into normalized rings, or
    returns None when it does not bound a closed region."""
    if isinstance(entity, PolygonEntity):
        if not entity.closed:
            return None
        return _ensure_winding(entity.get_world_rings(registry))
    if not entity.is_closed_loop():
        return None
    ring = entity.to_polyline(registry, _SAMPLE_TOLERANCE)
    if len(ring) < 3:
        return None
    return _ensure_winding([ring])


def _ensure_winding(rings: list) -> list:
    """Normalizes region rings via raygeo's topology: outer CCW,
    holes CW, regardless of the input winding. The round-trip through
    Geometry only re-orients the rings; the vertex data is preserved
    at the sample tolerance."""
    return [
        list(ring)
        for ring in rings_to_geometry(rings)
        .normalize_winding_orders()
        .to_polygons(_SAMPLE_TOLERANCE)
    ]


def apply_boolean(
    op: BooleanOp, regions: Sequence[BooleanRegion]
) -> list[BooleanSolid] | None:
    """
    Runs the boolean operation over the regions and regroups the
    result rings into solids. Returns None when the operation
    produces no geometry (e.g. no overlap).
    """
    if op is BooleanOp.UNION:
        rings = [ring for region in regions for ring in region.rings]
        result = get_polygons_union(rings)
    elif op is BooleanOp.DIFFERENCE:
        ordered = sorted(regions, key=lambda region: region.z)
        result = get_polygons_group_difference(
            ordered[0].rings,
            [ring for region in ordered[1:] for ring in region.rings],
        )
    elif op is BooleanOp.INTERSECTION:
        ordered = sorted(regions, key=lambda region: region.z)
        result = ordered[0].rings
        for region in ordered[1:]:
            result = get_polygons_group_intersection(result, region.rings)
            if not result:
                return None
    else:
        result = list(regions[0].rings)
        for region in regions[1:]:
            result = _symmetric_difference(result, region.rings)
            if not result:
                return None
    return _rings_to_solids(result)


def _symmetric_difference(rings_a: list, rings_b: list) -> list:
    """XOR of two flat ring groups as (A\\B) ∪ (B\\A). Computing it as
    union(A, B) − intersection(A, B) instead would let Clipper emit a
    degenerate outer ring with a boundary-touching hole wherever the
    difference only touches the source boundary."""
    diff_ab = get_polygons_group_difference(rings_a, rings_b)
    diff_ba = get_polygons_group_difference(rings_b, rings_a)
    return get_polygons_union([*diff_ab, *diff_ba])


def _rings_to_solids(rings) -> list[BooleanSolid] | None:
    """Cleans the boolean result rings and groups them into solids."""
    cleaned = [
        list(ring)
        for ring in rings_to_geometry(rings).to_polygons(_SAMPLE_TOLERANCE)
        if abs(get_polygon_signed_area(ring)) > _MIN_RING_AREA
    ]
    solids = group_rings_into_solids(cleaned)
    if not solids:
        return None
    return [BooleanSolid(outer, holes) for outer, holes in solids]

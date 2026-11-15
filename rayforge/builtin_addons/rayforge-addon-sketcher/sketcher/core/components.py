"""Connected components of the sketcher constraint graph.

The constraint system decouples exactly along these components:
parameters are constant during a solve, so no constraint couples
points across components. Interactive drags can therefore restrict
the solve to the component containing the dragged geometry.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from raygeo.sketch.types import EntityID

from . import constraint_info

if TYPE_CHECKING:
    from raygeo.sketch.constraints import Constraint
    from raygeo.sketch.registry import EntityRegistry


def get_referenced_points(
    registry: EntityRegistry, constraint: Constraint
) -> set[EntityID]:
    """
    Returns all point IDs a constraint couples: the points it
    references directly plus the points of the entities it
    references. Entity-level constraints (e.g. tangency between a
    line and a circle) couple their entities' points through their
    gradients.
    """
    pids = set(constraint.referenced_points())
    for eid in constraint_info.referenced_entity_ids(registry, constraint):
        entity_pids = registry.entity_point_ids(eid)
        if entity_pids:
            pids.update(entity_pids)
    return pids


def compute_constraint_components(
    registry: EntityRegistry,
    constraints: Sequence[Constraint],
) -> list[set[EntityID]]:
    """
    Partitions all points referenced by constraints into connected
    components of the constraint graph. Two points share a component
    when some chain of constraints couples them. Points not referenced
    by any constraint belong to no component.
    """
    parent: dict[EntityID, EntityID] = {}

    def find(pid: EntityID) -> EntityID:
        root = pid
        while parent[root] != root:
            root = parent[root]
        while parent[pid] != root:
            parent[pid], pid = root, parent[pid]
        return root

    for constraint in constraints:
        pids = list(get_referenced_points(registry, constraint))
        if not pids:
            continue
        first = pids[0]
        parent.setdefault(first, first)
        for pid in pids[1:]:
            parent.setdefault(pid, pid)
            parent[find(pid)] = find(first)

    components: dict[EntityID, set[EntityID]] = {}
    for pid in parent:
        components.setdefault(find(pid), set()).add(pid)
    return list(components.values())

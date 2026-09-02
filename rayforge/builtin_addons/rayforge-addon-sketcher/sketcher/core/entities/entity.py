import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from raygeo.geo import Geometry
from raygeo.geo.types import Polygon, Rect

from ..types import EntityID

if TYPE_CHECKING:
    from ..commands.mirror import MirrorAxis
    from ..constraints import Constraint
    from ..contour import OffsetItem
    from ..entity_group import PlacementTransform
    from ..registry import EntityRegistry
    from ..sketch import Sketch
    from .point import Point


def quantize(value: float) -> float:
    return round(value, 6)


_MIN_OFFSET_RADIUS = 1e-6


@dataclass
class OffsetPlan:
    """
    Pure description of the changes one offset operation makes.

    ``point_moves`` repositions existing defining points (in-place
    update, e.g. a circle's radius point). ``points`` and ``entities``
    are new geometry to add; ``removed_entity_ids`` are source
    entities to remove (replacement). The offset command turns a plan
    into snapshot-backed point mutations plus add/remove commands.
    """

    point_moves: dict[EntityID, tuple[float, float]] = field(
        default_factory=dict
    )
    points: list["Point"] = field(default_factory=list)
    entities: list["Entity"] = field(default_factory=list)
    removed_entity_ids: list[EntityID] = field(default_factory=list)


class Entity:
    """Base class for geometric primitives."""

    def __init__(self, id: EntityID, construction: bool = False):
        self.id: EntityID = id
        self.construction = construction
        self.invisible = False
        # True if this entity was created as a copy of an array
        # instance. Rendered with a distinct dashed style.
        self.array_copy = False
        self.type = "entity"
        # Constrained state is calculated by solver
        self.constrained = False

    def get_state(self) -> dict[str, Any] | None:
        """
        Returns a dictionary of solver-relevant discrete state (e.g. winding),
        or None if the entity has no mutable discrete state.
        Used for Undo/Redo snapshots.
        """
        # Construction state affects solver topology/rendering, so we capture
        # it. Subclasses should call super().get_state() or merge dicts if
        # they have more state.
        return {"construction": self.construction}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restores state from a snapshot."""
        if "construction" in state:
            self.construction = state["construction"]

    def update_constrained_status(
        self, registry: "EntityRegistry", constraints: Sequence["Constraint"]
    ) -> None:
        """
        Updates self.constrained based on the status of defining points
        and relevant constraints.
        """
        self.constrained = False

    def get_point_ids(self) -> list[EntityID]:
        """Returns IDs of all control points used by this entity."""
        return []

    def geometry_signature(self, registry: "EntityRegistry") -> tuple:
        """
        Returns a hashable signature capturing all geometry that
        affects this entity's shape: quantized defining-point
        positions, extended by subclasses with entity-internal state
        such as bezier control-point offsets or arc chirality.

        Coordinates are quantized (6 decimals): the solver leaves
        residual noise of roughly its own convergence tolerance
        (~1e-9 and coarser) on unheld points after every solve, and a
        finer comparison would misread that noise as an edit,
        triggering a re-apply on every solve. Used by array sync
        change detection.
        """
        return tuple(
            (quantize(p.x), quantize(p.y))
            for p in (registry.get_point(pid) for pid in self.get_point_ids())
            if p is not None
        )

    def get_endpoint_ids(self) -> list[EntityID]:
        """
        Returns IDs of the two endpoints for path/loop traversal.
        Returns empty list for single-point entities (Circle).
        Index 0 is the start, index 1 is the end.
        """
        return []

    def get_ignorable_unconstrained_points(self) -> list[EntityID]:
        """
        Returns IDs of points that can remain unconstrained if this entity
        is constrained (e.g. radius handles).
        """
        return []

    def hit_test(
        self,
        mx: float,
        my: float,
        threshold: float,
        registry: "EntityRegistry",
    ) -> bool:
        """
        Returns True if the point (mx, my) is within threshold distance of
        this entity in model coordinates.
        """
        return False

    def get_junction_point_ids(self) -> list[EntityID]:
        """
        Returns point IDs that should be counted for junction detection.
        These are typically the endpoints of geometric entities.
        """
        return []

    def get_helper_ids(self) -> list[EntityID]:
        """
        Returns auxiliary/child IDs belonging to this compound entity
        (e.g. ellipse helper lines, text box construction lines).
        """
        return []

    def get_rigidly_connected_points(
        self, point_id: EntityID
    ) -> list[EntityID]:
        """
        Returns point IDs that should move together with the given point
        as a rigid body during dragging. Used for entities where certain
        points should maintain their relative positions (e.g., ellipse center
        should drag all points together).
        """
        return []

    def get_drag_anchor_points(self, point_id: EntityID) -> list[EntityID]:
        """
        Returns point IDs that should be pinned at their current position
        while the given point is dragged (e.g., an ellipse center should
        stay put when one of its radius points is dragged).
        """
        return []

    def mirror(self, axis: "MirrorAxis") -> None:
        """
        Updates entity-specific non-point state for a mirror transform.

        Point positions are mirrored centrally by the command (since points
        are shared resources in the registry). This method handles only
        entity-internal state such as bezier control-point deltas or arc
        chirality. The default is a no-op: entities whose geometry is fully
        defined by their control points need no special handling.
        """

    def transform_offsets(self, placement: "PlacementTransform") -> None:
        """
        Updates entity-internal state for a placement transform, in
        the same spirit as ``mirror``: defining points are moved by
        the caller (see ``EntityGroup.apply_placement``); this method
        transforms only state relative to those points, such as
        bezier control-point offsets. The default is a no-op for
        entities fully defined by their defining points.
        """

    def rewrite_offsets_from(
        self, template: "Entity", placement: "PlacementTransform"
    ) -> None:
        """
        Re-derives this (copy) entity's internal state from a
        template entity of the same kind: the template's state
        transformed by the placement. Mirrors
        ``EntityGroup.rewrite_copy_from``, which owns the defining
        points; the template is the source of truth, so the copy's own
        state is never read. The default is a no-op.
        """

    def as_offset_item(self, sketch: "Sketch") -> "OffsetItem | None":
        """
        Returns the offsettable item this entity forms when it is
        offset on its own, or None when it cannot stand alone
        (chain members fall back to outline sampling; non-offsettable
        types are skipped). Only called for entities that make up
        their own connected component.
        """
        return None

    def plan_offset(
        self,
        registry: "EntityRegistry",
        offset: float,
        allocate_id: Callable[[], EntityID],
    ) -> OffsetPlan | None:
        """
        Returns a pure description of the changes offsetting this
        entity by the given amount would make, or None when the entity
        has no exact analytic offset or the offset collapses.

        A positive offset grows the enclosed area; a negative one
        shrinks it. Nothing is applied to the sketch; the caller
        applies the plan and relies on the command snapshot for undo.
        """
        return None

    def preview_polylines(
        self,
        registry: "EntityRegistry",
        offset: float,
    ) -> list[list[tuple[float, float]]]:
        """
        Returns coarse polylines of this entity's offset for live
        preview, without touching the sketch. Default: no preview.
        """
        return []

    def is_contained_by(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        """
        Returns True if the entity is fully strictly contained within the rect.
        Used for Window Selection.
        """
        return False

    def intersects_rect(
        self,
        rect: Rect,
        registry: "EntityRegistry",
    ) -> bool:
        """
        Returns True if the entity intersects the rect or is contained by it.
        Used for Crossing Selection.
        """
        return False

    def to_geometry(self, registry: "EntityRegistry") -> Geometry:
        """Converts the entity to a Geometry object."""
        return Geometry()

    def create_fill_geometry(
        self, registry: "EntityRegistry"
    ) -> Geometry | None:
        """
        Creates a fill geometry for single-entity loops.
        Returns None if the entity does not support fill geometry.
        """
        return None

    def append_to_geometry(
        self,
        geo: Geometry,
        registry: "EntityRegistry",
        forward: bool,
    ) -> None:
        """
        Appends this entity to an existing geometry object.
        Used for multi-segment loops.
        """

    def to_polygon_vertices(
        self,
        registry: "EntityRegistry",
        forward: bool,
    ) -> Polygon:
        """
        Converts this entity to a list of polygon vertices for hit testing.
        Curves should be sampled/linearized appropriately.
        """
        return []

    def to_polyline(
        self,
        registry: "EntityRegistry",
        tolerance: float = 0.1,
    ) -> list[tuple[float, float]]:
        """
        Samples this entity into a polyline in model coordinates,
        e.g. for preview rendering. The entity's geometry is flattened
        by raygeo with the given deviation tolerance. Entities without
        a stroked shape yield an empty polyline.
        """
        polygons = self.to_geometry(registry).to_polygons(tolerance)
        return polygons[0] if polygons else []

    def to_dict(self) -> dict[str, Any]:
        """Base serialization method for entities."""
        data = {
            "id": self.id,
            "type": self.type,
            "construction": self.construction,
        }
        if self.invisible:
            data["invisible"] = True
        if self.array_copy:
            data["array_copy"] = True
        return data

    def restore_base_flags(self, data: dict[str, Any]) -> None:
        """Restores base flags serialized by to_dict() but not accepted
        as constructor arguments."""
        self.invisible = data.get("invisible", False)
        self.array_copy = data.get("array_copy", False)

    def characteristic_length_pairs(
        self,
    ) -> list[tuple[EntityID, EntityID]]:
        """Return point-index pairs defining the entity's characteristic
        length(s).

        Used by equal-length constraints. Subclasses should override to
        return the relevant point pairs (e.g. Line returns its endpoints,
        Circle returns center+radius).
        """
        return []

    def characteristic_length(self, registry: "EntityRegistry") -> float:
        """Return the length/radius value used by equal-length constraints.

        The default computes the length from the first pair returned by
        ``characteristic_length_pairs``. Subclasses with custom metrics
        (e.g. Ellipse averages its two radii) should override.
        """
        pairs = self.characteristic_length_pairs()
        if not pairs:
            return 0.0
        pa = registry.get_point(pairs[0][0])
        pb = registry.get_point(pairs[0][1])
        if pa and pb:
            return math.hypot(pb.x - pa.x, pb.y - pa.y)
        return 0.0

    def signed_distance_to(
        self, point: "Point", registry: "EntityRegistry"
    ) -> float:
        """Return the signed distance from ``point`` to this entity.

        Positive means the point is outside the entity's locus.
        Raises ``NotImplementedError`` for entities that cannot compute
        signed distance.
        """
        raise NotImplementedError

    def tangent_at(
        self, registry: "EntityRegistry", point_id: EntityID
    ) -> tuple[float, float]:
        """Return the tangent vector at a specific point on this entity.

        The vector points away from ``point_id`` along the entity's curve.
        Raises ``NotImplementedError`` for entities with no directional
        tangent (e.g. Circle).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"Entity(id={self.id}, type={self.type})"

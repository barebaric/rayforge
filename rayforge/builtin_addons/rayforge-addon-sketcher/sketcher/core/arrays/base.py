from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar

from ..entities import Circle
from ..entity_group import EntityGroup

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..entities import Entity, Point
    from ..registry import EntityRegistry

logger = logging.getLogger(__name__)


class PlacementKind(Enum):
    ROTATION = auto()
    TRANSLATION = auto()
    CURVE_ALIGNED = auto()


@dataclass(frozen=True)
class InstancePlacement:
    """
    Describes how one array instance is derived from the template.

    CURVE_ALIGNED applies a rigid transform that maps the template from
    its drawn frame onto a target frame along a guide curve: the
    template's center is translated to ``target_center`` and rotated by
    ``angle`` so the member follows the curve's tangent. This keeps
    each copy congruent to the template (a similarity-free rigid
    motion), the same invariant the other placement kinds rely on.
    """

    kind: PlacementKind
    angle: float = 0.0
    center: tuple[float, float] = (0.0, 0.0)
    delta: tuple[float, float] = (0.0, 0.0)
    target_center: tuple[float, float] = (0.0, 0.0)

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        if self.kind == PlacementKind.ROTATION:
            ca = math.cos(self.angle)
            sa = math.sin(self.angle)
            dx = x - self.center[0]
            dy = y - self.center[1]
            return (
                self.center[0] + ca * dx - sa * dy,
                self.center[1] + sa * dx + ca * dy,
            )
        if self.kind == PlacementKind.CURVE_ALIGNED:
            ca = math.cos(self.angle)
            sa = math.sin(self.angle)
            dx = x - self.center[0]
            dy = y - self.center[1]
            return (
                self.target_center[0] + ca * dx - sa * dy,
                self.target_center[1] + sa * dx + ca * dy,
            )
        return (x + self.delta[0], y + self.delta[1])

    def transform_offset(self, dx: float, dy: float) -> tuple[float, float]:
        """Transforms a relative offset (e.g. a bezier control point)."""
        if self.kind in (
            PlacementKind.ROTATION,
            PlacementKind.CURVE_ALIGNED,
        ):
            ca = math.cos(self.angle)
            sa = math.sin(self.angle)
            return (ca * dx - sa * dy, sa * dx + ca * dy)
        return (dx, dy)


class ArrayStrategy(ABC):
    """
    Base class for sketch array strategies.

    A strategy is an ephemeral, stateless math object: it carries the
    user-facing parameters of one array (as typed constructor
    attributes) and computes where the template and its copies sit on
    the guide, plus the optional "master" construction geometry that
    carries the array's definition (e.g. the guide circle of a
    circular array). Adding a new array type (e.g. linear) means
    adding a strategy subclass; commands and tools stay generic.

    Every array, regardless of strategy, follows the same process:

    1. identify a guide entity (a constructed circle for circular
       arrays, the first selected entity for curve arrays);
    2. extract the template: the selected entities minus the guide,
       with external constraints erased;
    3. identify the template's center point;
    4. place the template at position 0 on the guide
       (``template_placement``) and bake its position;
    5. place the copies (``member_placements``), derived from the
       baked template.

    On edit, the template is re-placed at position 0 and the copies
    are re-derived.
    """

    #: Number of members: the template plus ``count - 1`` copies.
    #: Shared by every strategy; strategies whose layout is
    #: spacing-driven may derive it at placement time.
    count: int = 6

    #: Whether copies follow rotation placements where applicable.
    #: Only meaningful for strategies with angular slots; kept here
    #: so every strategy carries a uniform parameter surface.
    rotate_copies: bool = True

    #: Whether the array is defined relative to an explicit center point.
    needs_center_point: ClassVar[bool] = False

    #: Whether the strategy records where it placed the template onto
    #: the guide (the ``template_anchor``), so guide edits re-anchor
    #: the template by the rigid motion between the stored and the new
    #: anchor.
    uses_template_anchor: ClassVar[bool] = False

    def existing_master_id(self) -> int | None:
        """Returns the entity the strategy reuses as the array's
        master when no master geometry was created (e.g. the
        pre-drawn guide path of a curve-along array), or None."""
        return None

    @abstractmethod
    def member_placements(
        self,
        template_center: tuple[float, float],
        registry: Any | None = None,
    ) -> list[InstancePlacement]:
        """
        Returns the placements deriving the copies (slots 1..N-1)
        from the baked template member (slot 0).

        Args:
            template_center: Center of the template geometry.
            registry: Entity registry. Required by strategies whose
                placement depends on existing geometry (e.g. the guide
                path of a "array along curve" array); ignored by
                strategies whose placement is closed-form.
        """

    def template_placement(
        self,
        template_center: tuple[float, float],
        registry: Any | None = None,
    ) -> InstancePlacement:
        """
        Returns the placement putting a freshly drawn template onto
        position 0 of the guide: for curve arrays it is placed on the
        path start, rotated to the tangent; for circular arrays it is
        translated radially onto the guide circle at the angle it was
        drawn at. Together with ``member_placements`` this is the
        single math every consumer (create, edit, sync, preview) uses
        to derive array geometry.
        """
        return InstancePlacement(
            kind=PlacementKind.TRANSLATION, delta=(0.0, 0.0)
        )

    def create_master_geometry(
        self,
        center_pid: int | None,
        radius_pt_pid: int | None,
    ) -> tuple[list[Any], list[Entity], list[Constraint]]:
        """
        Returns the master construction geometry (points, entities,
        constraints) that visualizes and carries the array definition.
        Instances are equal static copies; only the master is special.
        """
        return [], [], []

    # ------------------------------------------------------------------
    # Edit support: state transitions driven by EditArrayCommand live
    # on the Array itself (see Array.snapshot/restore/commit); the
    # strategy keeps only the master-frame behaviour below.
    # ------------------------------------------------------------------

    def capture_master_frame(
        self, registry: EntityRegistry, array_def: Array
    ) -> tuple[tuple[float, float], float] | None:
        """
        Returns the master geometry's reference frame as
        ``(center, radius)``, or None if the strategy has no scalable
        master (e.g. a curve-along array whose guide is a path, not a
        circle). Used by EditArrayCommand to similarity-transform the
        existing geometry into the edited frame before regenerating.
        """
        return

    def apply_frame(
        self,
        registry: EntityRegistry,
        array_def: Array,
        old_frame: tuple[tuple[float, float], float],
        constraints: list,
        frame_state: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Moves the whole existing array into the edited frame so the
        solver starts fully consistent. ``constraints`` is the sketch's
        constraint list (needed to update dimension values in place).
        Returns an opaque state object to be passed back to
        ``apply_frame`` on redo, or None if no frame transform applies.
        The default is a no-op (curve-along arrays have no scalable
        master).
        """
        return


def resolve_template_center(
    registry: EntityRegistry,
    template_entity_ids: list[int],
    template_points: list[Point],
) -> tuple[float, float]:
    """Returns the logical center of the template entities.

    When the template contains exactly one Circle or Ellipse (plus any
    attached helper geometry) that entity's explicit center point is
    the logical center — the defining points of an ellipse only span
    a quarter of its area, so a bbox over them is wrong. For anything
    else the bbox center of all defining points is used.

    Compatibility wrapper; the logic lives on ``EntityGroup.center``.
    """
    return EntityGroup(registry, template_entity_ids).center()


class Array(ABC):
    """
    Persistent definition of a sketch array ("master" object).

    Every array, regardless of strategy, consists of a guide entity
    (guide circle or guide path) and ``count`` members: member 0 is
    the template — the extracted copy of the selected entities, placed
    onto the guide by the
    strategy — and members 1..N-1 are copies derived from it by the
    strategy's placements. Groups keep their identity across
    deletions: removing part of a member leaves a smaller group, never
    orphaned fragments that regenerate as broken copies.

    ``self.members`` holds ``(slot, [entity_id, ...])`` pairs; slot 0
    is the template member, slots 1..N-1 correspond to the
    placements.

    Mode-specific state lives on the concrete subclasses
    (``CircularArray``, ``CurveAlongArray``), each serialized next to
    its strategy. ``from_dict`` dispatches on the stored mode.

    ``template_anchor`` (curve-along only) records where the strategy
    last placed the template onto the guide (``((x, y), angle)``).
    When the guide moves, the template is re-anchored by the rigid
    motion from the stored to the new anchor, so user edits of the
    template survive re-distribution.
    """

    #: The serialized mode discriminator of the concrete subclass.
    MODE: ClassVar[str]

    #: The strategy class implementing this array kind.
    STRATEGY: ClassVar[type[ArrayStrategy]]

    #: Concrete Array subclasses by serialized mode string, registered
    #: automatically so ``from_dict`` can dispatch without importing
    #: the mode modules (which import this one).
    _MODE_REGISTRY: ClassVar[dict[str, type[Array]]] = {}

    #: Concrete Array subclasses by strategy class, registered
    #: automatically so ``from_strategy`` can dispatch type-safely.
    _STRATEGY_REGISTRY: ClassVar[dict[type[ArrayStrategy], type[Array]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "MODE" in cls.__dict__:
            cls._MODE_REGISTRY[cls.MODE] = cls
        if "STRATEGY" in cls.__dict__:
            cls._STRATEGY_REGISTRY[cls.STRATEGY] = cls

    def __init__(
        self,
        uid: str,
        guide_circle_id: int,
        members: list[tuple[int, list[int]]] | None = None,
        count: int = 6,
        rotate_copies: bool = True,
    ):
        self.uid = uid
        self.mode = self.MODE
        self.guide_circle_id = guide_circle_id
        self.members: list[tuple[int, list[int]]] = [
            (slot, list(eids)) for slot, eids in (members or [])
        ]
        self.count = count
        self.rotate_copies = rotate_copies
        # Transient caches for sync_arrays: store the guide's and the
        # template's geometry after each solve so changes can be
        # detected.
        self._cached_guide_sig: tuple | None = None
        self._cached_template_sig: tuple | None = None
        self._cached_guide_frame: tuple[tuple[float, float], float] | None = (
            None
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Array:
        """Deserializes the concrete subclass registered for the
        array's mode."""
        mode = data.get("mode", "circular")
        subclass = cls._MODE_REGISTRY.get(mode)
        if subclass is None:
            raise ValueError(f"Unsupported sketch array mode: {mode}")
        return subclass._from_dict(data)

    @classmethod
    def from_strategy(
        cls,
        strategy: ArrayStrategy,
        uid: str,
        guide_circle_id: int,
        members: list[tuple[int, list[int]]],
        count: int,
        template_anchor: tuple[tuple[float, float], float] | None = None,
    ) -> Array:
        """Creates the concrete array registered for the given
        strategy class, carrying the strategy's parameters."""
        subclass = cls._STRATEGY_REGISTRY.get(type(strategy))
        if subclass is None:
            raise TypeError(
                f"Unsupported array strategy: {type(strategy).__name__}"
            )
        return subclass._from_strategy(
            strategy=strategy,
            uid=uid,
            guide_circle_id=guide_circle_id,
            members=members,
            count=count,
            template_anchor=template_anchor,
        )

    @abstractmethod
    def make_strategy(self, registry: EntityRegistry) -> ArrayStrategy:
        """Builds this array's strategy from its stored parameters.
        Strategies whose placement reads live geometry (e.g. the guide
        circle's center and radius) take the values from the
        registry; the others from the stored fields."""

    @classmethod
    def _from_strategy(
        cls,
        strategy: ArrayStrategy,
        uid: str,
        guide_circle_id: int,
        members: list[tuple[int, list[int]]],
        count: int,
        template_anchor: tuple[tuple[float, float], float] | None = None,
    ) -> Array:
        """Constructs the mode-specific array from a strategy.
        Implemented by each concrete subclass; dispatched to by
        ``from_strategy``."""
        raise NotImplementedError

    @staticmethod
    def _parse_members(data: dict[str, Any]) -> list[tuple[int, list[int]]]:
        """Parses the member list."""
        return [
            (int(slot), [int(eid) for eid in eids])
            for slot, eids in data.get("members", [])
        ]

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Array:
        """Deserializes the mode-specific fields. Implemented by each
        concrete subclass; dispatched to by ``from_dict``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Edit support: state transitions driven by EditArrayCommand.
    # The command stays generic; each concrete Array owns which fields
    # are editable and how they map to a strategy.
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """
        Returns a serializable snapshot of the array's editable state,
        for undo/redo. Mode-specific fields are added by subclasses.
        """
        return {
            "members": [(slot, list(eids)) for slot, eids in self.members],
            "count": self.count,
        }

    def restore(self, state: dict[str, Any]) -> None:
        """
        Restores the array's state from a snapshot produced by
        ``snapshot``. Called on undo/redo.
        """
        self.members = [(slot, list(eids)) for slot, eids in state["members"]]
        self.count = state["count"]

    def commit(self, strategy: ArrayStrategy) -> None:
        """
        Writes the strategy's parameters onto the array. Called after
        a successful edit. Mode-specific fields are handled by
        subclasses.
        """
        self.count = strategy.count

    def params_changed(self, strategy: ArrayStrategy) -> bool:
        """
        Returns True if the strategy's parameters differ from this
        array's state in a way that requires a full re-distribution
        of all members. Mode-specific fields are compared by
        subclasses.
        """
        return strategy.count != self.count

    # ------------------------------------------------------------------
    # Sync support: change detection driven by Sketch.sync_arrays.
    # The sketch owns the orchestration and the mutation (re-apply via
    # EditArrayCommand); the array only reads the registry, reports
    # decisions and maintains its own caches.
    # ------------------------------------------------------------------

    def guide_signature(self, registry: EntityRegistry) -> tuple:
        """
        Quantized signature of the guide geometry that array members
        depend on: the full guide shape. For circular arrays this
        includes the radius — the radius drives the member placement,
        so a radius edit redistributes the members onto the new
        orbit. A missing guide yields the empty signature.
        """
        sig = registry.geometry_signature(self.guide_entity_id)
        return sig if sig is not None else ()

    def template_signature(self, registry: EntityRegistry) -> tuple:
        """Quantized geometry signature of the template member (slot
        0), one entry per surviving template entity."""
        living = self.living_members(registry)
        if not living:
            return ()
        signatures: list[tuple] = []
        for eid in living[0][1]:
            sig = registry.geometry_signature(eid)
            if sig is not None:
                signatures.append(sig)
        return tuple(signatures)

    def signatures_changed(
        self, guide_sig: tuple, template_sig: tuple
    ) -> bool:
        """True if the stored sync caches disagree with the given
        signatures, i.e. the guide or the template was edited since
        the last sync. Missing caches (fresh arrays, clones) never
        count as a change."""
        guide_changed = (
            self._cached_guide_sig is not None
            and self._cached_guide_sig != guide_sig
        )
        template_changed = (
            self._cached_template_sig is not None
            and self._cached_template_sig != template_sig
        )
        if guide_changed:
            logger.info(
                "ArraySync[%s]: guide signature changed %r -> %r",
                self.uid[:8],
                self._cached_guide_sig,
                guide_sig,
            )
        if template_changed:
            logger.info(
                "ArraySync[%s]: template signature changed %r -> %r",
                self.uid[:8],
                self._cached_template_sig,
                template_sig,
            )
        return guide_changed or template_changed

    def update_caches(self, guide_sig: tuple, template_sig: tuple) -> None:
        """Stores the current guide/template signatures after a
        sync cycle."""
        self._cached_guide_sig = guide_sig
        self._cached_template_sig = template_sig

    def refresh_caches(
        self, registry: EntityRegistry, strategy: ArrayStrategy
    ) -> None:
        """Recomputes all sync caches from the current geometry so the
        next solve doesn't trigger a spurious re-apply. Called after
        creation and after a committed edit."""
        self._cached_guide_frame = strategy.capture_master_frame(
            registry, self
        )
        self._cached_guide_sig = self.guide_signature(registry)
        self._cached_template_sig = self.template_signature(registry)

    def prune(self, registry: EntityRegistry) -> bool:
        """
        Drops deleted entities from member groups. Returns False if
        the master geometry is gone and the array should be removed.
        Groups themselves are never dissolved by deletion; deleting
        them does not dissolve the array as long as the master
        geometry still exists.
        """
        if registry.get_entity(self.guide_circle_id) is None:
            return False
        self.members = self.living_members(registry)
        return True

    def is_guide_radius_point(
        self, registry: EntityRegistry, pid: int
    ) -> bool:
        """
        True when the point is the radius point of this array's
        construction circle. Its position is governed by the radius
        dimension (the array's size definition), so drags must not
        move it. Arrays whose guide is not a circle have no radius
        point.
        """
        circle = registry.get_entity(self.guide_circle_id)
        return isinstance(circle, Circle) and circle.radius_pt_idx == pid

    def reanchor_template(
        self,
        strategy: ArrayStrategy,
        registry: EntityRegistry,
        template_eids: list[int],
    ) -> None:
        """
        Re-positions the template member onto position 0 of the guide
        after a guide edit. The template's position is guide-owned
        (like the circular array, whose members are re-projected onto
        the orbit): its center is placed absolutely onto the guide's
        position-0 point and rotated by the tangent change. Template
        *shape* edits survive; position drags do not. No-op by
        default (strategies without a template placement).
        """

    @property
    def guide_entity_id(self) -> int:
        """
        The master entity of the array: the guide circle for circular
        arrays, or the guide path for curve-along arrays. ``guide_circle_id``
        carries the value for both for historical reasons.
        """
        return self.guide_circle_id

    def living_members(self, registry: Any) -> list[tuple[int, list[int]]]:
        """
        Returns (slot, [entity_id, ...]) pairs for members with at least
        one surviving entity, pruned to surviving entities only.
        """
        living: list[tuple[int, list[int]]] = []
        for slot, eids in self.members:
            alive = [
                eid for eid in eids if registry.get_entity(eid) is not None
            ]
            if alive:
                living.append((slot, alive))
        return sorted(living)

    def living_entity_ids(self, registry: Any) -> list[int]:
        """Flat list of all surviving member entity IDs."""
        return [
            eid
            for _slot, eids in self.living_members(registry)
            for eid in eids
        ]

    def occupied_slots(self, registry: Any) -> set[int]:
        """Returns the slot numbers of surviving members."""
        return {slot for slot, _eids in self.living_members(registry)}

    def to_dict(self) -> dict[str, Any]:
        """Serializes the strategy-neutral fields. Mode-specific keys
        are added by the concrete subclass."""
        return {
            "uid": self.uid,
            "mode": self.mode,
            "guide_circle_id": self.guide_circle_id,
            "members": [[slot, list(eids)] for slot, eids in self.members],
            "count": self.count,
            "rotate_copies": self.rotate_copies,
        }


def find_array_for_entity(arrays: list[Array], entity_id: int) -> Array | None:
    """Returns the array whose master entity is the given one."""
    for arr in arrays:
        if arr.guide_entity_id == entity_id:
            return arr
    return None

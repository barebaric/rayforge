import copy
import json
import logging
import math
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from gettext import gettext as _
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from blinker import Signal
from raygeo.geo import Arc as GeoArc
from raygeo.geo import Bezier as GeoBezier
from raygeo.geo import Geometry
from raygeo.geo import Line as GeoLine
from raygeo.geo import Move as GeoMove
from raygeo.geo.shape.polygon import is_point_inside_polygon
from raygeo.sketch.constraints.coincident import CoincidentConstraint
from raygeo.sketch.constraints.distance import DistanceConstraint
from raygeo.sketch.constraints.drag import DragConstraint
from raygeo.sketch.constraints.equal_distance import (
    EqualDistanceConstraint,
)
from raygeo.sketch.constraints.horizontal import HorizontalConstraint
from raygeo.sketch.constraints.point_on_line import PointOnLineConstraint
from raygeo.sketch.constraints.radius import RadiusConstraint
from raygeo.sketch.constraints.diameter import DiameterConstraint
from raygeo.sketch.constraints.symmetry import SymmetryConstraint
from raygeo.sketch.constraints.symmetry import SymmetryTarget
from raygeo.sketch.constraints.vertical import VerticalConstraint
from raygeo.sketch.entities.text_box import TextBox
from raygeo.sketch.registry import EntityRegistry

from rayforge.core.asset import IAsset
from rayforge.core.color import ColorRGBA
from rayforge.core.expression import ExpressionMap
from rayforge.core.geometry_provider import IGeometryProvider
from rayforge.core.varset import VarSet
from rayforge.image.geo_renderer import render_geometry_to_png
from rayforge.image.structures import FillRenderData, FillStyle

from . import constraint_info
from .arrays import Array
from .components import get_referenced_points
from .constraint_ops import rebuild as _rebuild_constraint
from .entity_flags import EntityFlags, WaypointType
from .legacy import migrate_sketch_data
from .params import ParameterContext
from .solver import Solver
from .template_functions import get_template_functions
from .types import EntityID

DEFAULT_FILL_COLOR: ColorRGBA = (0.85, 0.85, 0.85, 0.7)
FORMAT_VERSION = 1
_DEFAULT_VARSET_TITLE = _("Sketch Parameters")
_DEFAULT_VARSET_DESCRIPTION = _(
    "Parameters that control this sketch's geometry"
)

logger = logging.getLogger(__name__)


class Fill:
    """Represents a filled area bounded by sketch entities."""

    def __init__(
        self,
        uid: str,
        boundary: list[tuple[EntityID, bool]],
        style: FillStyle = FillStyle.SOLID,
        color: ColorRGBA = DEFAULT_FILL_COLOR,
        gradient_stops: list[tuple[float, ColorRGBA]] | None = None,
        gradient_angle: float = 0.0,
    ):
        self.uid = uid
        self.boundary: list[tuple[EntityID, bool]] = boundary
        self.style = style
        self.color = color
        self.gradient_stops = gradient_stops or []
        self.gradient_angle = gradient_angle

    def to_dict(self) -> dict[str, Any]:
        data = {
            "uid": self.uid,
            "boundary": [list(item) for item in self.boundary],
            "style": self.style.value,
            "color": list(self.color),
        }
        if self.gradient_stops:
            data["gradient_stops"] = [
                [pos, list(c)] for pos, c in self.gradient_stops
            ]
        if self.gradient_angle != 0.0:
            data["gradient_angle"] = self.gradient_angle
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fill":
        boundary = [tuple(item) for item in data["boundary"]]
        style = FillStyle(data.get("style", "solid"))
        color = tuple(data.get("color", list(DEFAULT_FILL_COLOR)))
        gradient_stops = None
        if "gradient_stops" in data:
            gradient_stops = [
                (pos, tuple(c)) for pos, c in data["gradient_stops"]
            ]
        gradient_angle = data.get("gradient_angle", 0.0)
        return cls(
            uid=data.get("uid", str(uuid.uuid4())),
            boundary=boundary,
            style=style,
            color=color,
            gradient_stops=gradient_stops,
            gradient_angle=gradient_angle,
        )


class Sketch(IAsset, IGeometryProvider):
    """
    A parametric sketcher that allows defining geometry via constraints
    and expressions.
    """

    is_addable: ClassVar[bool] = True
    asset_type_name: ClassVar[str] = "sketch"
    display_icon_name: ClassVar[str] = "sketch-edit-symbolic"
    is_reorderable: ClassVar[bool] = False
    is_draggable_to_canvas: ClassVar[bool] = True
    type_display_name: ClassVar[str] = _("Sketch")
    can_edit: ClassVar[bool] = True
    add_action: ClassVar[str | None] = "add-sketch"
    activate_action: ClassVar[str | None] = "activate-sketch"
    edit_item_action: ClassVar[str | None] = "edit-sketch-item"

    # Reentrancy guard for solve(). Class-level default so instances
    # created without __init__ (e.g. _clone_for_geometry via __new__)
    # always observe False.
    _solving: bool = False

    def __init__(self, name: str = "New Sketch") -> None:
        self._uid: str = str(uuid.uuid4())
        self._name = name
        self.params = ParameterContext()
        self.registry = EntityRegistry()
        self._constraints: list[Any] = []
        self.flags = EntityFlags()
        self.flags._registry = self.registry
        self.fills: list[Fill] = []
        self.arrays: list[Array] = []
        self.input_parameters = VarSet(
            title=_DEFAULT_VARSET_TITLE,
            description=_DEFAULT_VARSET_DESCRIPTION,
        )
        self._updated = Signal()
        self._hidden: bool = False
        self._last_solve_values: dict[str, Any] = {}
        self._resolved_text_cache: dict[EntityID, tuple[str, str | None]] = {}
        self._solved_ctx: dict[str, Any] | None = None

        # Cache for coincident-point groups. Keys are point IDs,
        # values are frozensets of all points in the same group.
        self._coincident_cache: dict[EntityID, frozenset[EntityID]] = {}
        self._coincident_dirty: bool = False

        # Initialize the Origin Point (Fixed Anchor)
        self.origin_id: EntityID = self.registry.add_point(
            0.0, 0.0, fixed=True
        )

    @property
    def constraints(self) -> list[Any]:
        """The sketch's constraints, in registration order. The list
        is a snapshot; add and remove constraints through the sketch's
        ``add_constraint``/``remove_constraint`` methods."""
        return list(self._constraints)

    def add_constraint(self, constraint: Any) -> Any:
        """Registers a constraint object and returns it."""
        self.registry.add_constraint(constraint)
        self._constraints.append(constraint)
        self._coincident_dirty = True
        return constraint

    def add_constraints(self, constraints: Iterable[Any]) -> list[Any]:
        """Registers several constraint objects and returns them."""
        added = []
        for constraint in constraints:
            added.append(self.add_constraint(constraint))
        return added

    def add_constraint_from_dict(self, data: dict[str, Any]) -> Any:
        """Registers a constraint from its reference dictionary
        (entity references resolved by the registry) and returns the
        stored constraint object."""
        index = self.registry.add_constraint_json(data)
        constraint = self.registry.get_constraint_object(index)
        self._constraints.append(constraint)
        self._coincident_dirty = True
        return constraint

    def remove_constraint_at(self, index: int) -> Any:
        """Removes the constraint at the given position and returns
        it."""
        constraint = self._constraints.pop(index)
        self.registry.remove_constraint(index)
        self._coincident_dirty = True
        return constraint

    def remove_constraint_object(self, constraint: Any) -> bool:
        """Removes the given constraint (matched by identity). Returns
        whether it was found."""
        for index, candidate in enumerate(self._constraints):
            if candidate is constraint:
                self.remove_constraint_at(index)
                return True
        return False

    def remove_constraints(self, constraints: Sequence[Any]) -> list[Any]:
        """Removes the given constraints (matched by identity) and
        returns the removed subset, in registration order."""
        targets = set(id(c) for c in constraints)
        removed = [c for c in self._constraints if id(c) in targets]
        for constraint in removed:
            self.remove_constraint_object(constraint)
        return removed

    def replace_constraint_at(self, index: int, constraint: Any) -> None:
        """Replaces the constraint at the given position."""
        self._constraints[index] = constraint
        self.registry.replace_constraint(index, constraint)

    def notify_update(self):
        """Public method to signal that the sketch has been modified."""
        if self._solving:
            # Re-entrant update from sync_arrays' array re-apply. The
            # solve() caller owns the redraw; re-emitting here would
            # fan out into the UI pipeline mid-solve and re-trigger
            # solves (solve/sync feedback loop with the array sync,
            # pipeline churn on every frame).
            return
        self._coincident_dirty = True
        self._updated.send(self)

    def capture_undo_state(self) -> dict[str, Any]:
        """Captures full sketch state for undo: points, entity states,
        app-level flags and array definitions.  The returned dict is
        opaque — callers store and pass it back via
        ``apply_undo_state``."""
        points = {
            point.id: (point.x, point.y)
            for point in self.registry.points()
        }
        entities: dict[int, Any] = {}
        for eid in self.registry.entity_ids():
            state = self.registry.get_state(eid)
            if state is not None:
                entities[eid] = state
        arrays = [(a.uid, a.snapshot()) for a in self.arrays]
        return {
            "points": points,
            "entities": entities,
            "arrays": arrays,
            "flags": self.flags.snapshot(),
        }

    def apply_undo_state(self, state: dict[str, Any]) -> None:
        """Restores sketch state from a dict captured by
        ``capture_undo_state``."""
        points = state["points"]
        entities = state["entities"]
        arrays = state["arrays"]

        for pid, (x, y) in points.items():
            self.registry.set_point_position(pid, x, y)

        for eid, estate in entities.items():
            self.registry.set_state(eid, estate)

        self.flags.restore(state.get("flags", {}))

        by_uid = {a.uid: a for a in self.arrays}
        for uid, astate in arrays:
            a = by_uid.get(uid)
            if a is not None:
                a.restore(astate)

    def _validate_and_cleanup_fills(self):
        """
        Removes any Fill objects whose boundary entities no longer form a
        valid, closed loop (e.g., if an entity was deleted).
        """
        valid_fills = []
        # Find all currently valid loops to check against
        current_loops = self._find_all_closed_loops()
        # For efficient lookup, convert lists to sets of tuples
        current_loop_sets = {frozenset(loop) for loop in current_loops}

        for fill in self.fills:
            fill_boundary_set = frozenset(fill.boundary)
            if fill_boundary_set in current_loop_sets:
                valid_fills.append(fill)

        self.fills = valid_fills

    @property
    def uid(self) -> str:
        """The unique identifier of the asset instance."""
        return self._uid

    @uid.setter
    def uid(self, value: str) -> None:
        """Set the unique identifier. Used for deserialization."""
        self._uid = value

    @property
    def updated(self) -> "Signal":
        """Signal emitted when the sketch changes."""
        return self._updated

    @property
    def name(self) -> str:
        """The user-facing name of the asset."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Sets the asset name and sends an update signal if changed."""
        if self._name != value:
            self._name = value
            self._updated.send(self)

    @property
    def provider_type_name(self) -> str:
        """The type name for geometry provider identification."""
        return "sketch"

    @property
    def renderer(self):
        """The renderer to use for rendering this sketch's geometry."""
        # Local import: sketcher.image.importer imports this package
        # (sketcher.core) at runtime, so a module-level import would
        # be circular.
        from ..image.renderer import SKETCH_RENDERER

        return SKETCH_RENDERER

    def get_reference_point(self) -> tuple[float, float]:
        """
        The sketch origin, in sketch coordinates. The origin is a fixed
        solver anchor, so geometry constrained to it stays registered
        when parameters change.
        """
        return self.registry.get_point(self.origin_id) or (0.0, 0.0)

    def get_geometry(
        self,
        params: dict[str, Any] | None = None,
        *,
        resolved_text_cache: dict | None = None,
    ) -> tuple[Geometry, list[FillRenderData]]:
        """
        Generate geometry with optional parameter overrides.

        Creates a clone, solves it with the given parameters, and returns
        the stroke and fill geometries.

        When *resolved_text_cache* is supplied the clone is seeded with
        those values so that volatile template expressions (e.g.
        ``uuid4()``) produce the same value across calls.  The dict is
        updated in-place with any newly resolved values so callers can
        persist the cache.

        Args:
            params: Optional dictionary of parameter values to override.
            resolved_text_cache: Optional mutable dict (entity_id → text)
                that carries resolved text across calls.

        Returns:
            A tuple of (stroke_geometry, fill_render_data).
        """
        clone = self._clone_for_geometry()
        if resolved_text_cache is not None:
            clone._resolved_text_cache = dict(resolved_text_cache)
        clone.solve(variable_overrides=params)
        if resolved_text_cache is not None:
            for k, v in resolved_text_cache.items():
                if k not in clone._resolved_text_cache:
                    clone._resolved_text_cache[k] = v
        geo = clone.to_geometry()
        fills = clone.get_fill_render_data()
        if resolved_text_cache is not None:
            resolved_text_cache.update(clone._resolved_text_cache)
        return geo, fills

    def _clone_for_geometry(self) -> "Sketch":
        """
        Lightweight clone for geometry generation.

        Copies the mutable state a solve modifies (point positions,
        parameter context) while sharing structural data (entities,
        constraints, fills) with the original.  Much cheaper than a
        full to_dict / from_dict round-trip of the whole Sketch.

        The registry itself is cloned (fresh point objects, the solver
        mutates positions); the entity and constraint definitions come
        along through its serialized form.  Entity/constraint lists
        and array definitions are copied because the clone's solve may
        re-apply arrays (removing/adding entities, dropping
        constraints, rewriting Array members).
        """
        clone = Sketch.__new__(Sketch)
        clone._uid = self._uid
        clone._name = self._name
        clone._hidden = self._hidden
        clone.origin_id = self.origin_id
        # Fresh signal: the clone must not trigger the original's
        # subscribers (solve/repaint) mid-clone-solve.
        clone._updated = Signal()

        registry_dict = self.registry.to_dict()
        clone.registry = EntityRegistry.from_dict(registry_dict)
        clone._constraints = list(clone.registry.constraints())
        clone.flags = copy.deepcopy(self.flags)

        # ParameterContext: copy expressions so evaluate_all on the clone
        # does not disturb the original's cache.
        clone.params = ParameterContext.from_dict(self.params.to_dict())

        # Copy containers and definitions that a solve may mutate
        # (removals/extensions during an array re-apply).
        clone.arrays = copy.deepcopy(self.arrays)
        # Share references to data not mutated by solve.
        clone.fills = self.fills
        clone.input_parameters = self.input_parameters

        # Fresh mutable state for the solve cycle.
        clone._last_solve_values = {}
        clone._resolved_text_cache = {}
        clone._solved_ctx = None
        # Build coincident cache so geometry generation is fast.
        clone._coincident_cache = {}
        clone._coincident_dirty = False
        clone._build_coincident_cache()
        return clone

    @property
    def hidden(self) -> bool:
        """Indicates if this asset should be hidden from the UI."""
        return self._hidden

    @hidden.setter
    def hidden(self, value: bool):
        """Sets the hidden state and sends an update signal if changed."""
        if self._hidden != value:
            self._hidden = value
            self._updated.send(self)

    def set_hidden(self, value: bool):
        """Setter method for use with undo commands."""
        self.hidden = value

    def get_thumbnail(self, size: int) -> bytes | None:
        """Returns a PNG thumbnail of the sketch geometry."""
        try:
            return render_geometry_to_png(self.to_geometry(), size)
        except Exception:
            logger.exception("Failed to generate sketch thumbnail")
            return None

    @property
    def is_empty(self) -> bool:
        """Returns True if the sketch has no drawable entities."""
        # We check entities rather than points, because an empty sketch
        # always contains at least one point (the origin).
        return self.registry.entity_count() == 0

    @property
    def is_fully_constrained(self) -> bool:
        """
        Returns True if every point and every entity in the sketch
        is fully constrained.

        Exception: Points that serve solely as internal handles for fully
        constrained entities (e.g., Circle radius point) are ignored if they
        are not constrained, provided they are not used by any other entity.
        """
        if self.registry.point_count() == 0:
            return True

        entities = self.registry.entities()

        # 1. All entities must be constrained
        if not all(entity.constrained for entity in entities):
            return False

        # 2. Calculate point usage counts to ensure exclusive ownership
        usage_count: dict[EntityID, int] = {}
        for entity in entities:
            for pid in entity.get_point_ids():
                usage_count[pid] = usage_count.get(pid, 0) + 1

        # 3. Collect allowed exemptions polymorphically
        allowed_unconstrained_ids = set()
        for entity in entities:
            candidates = self.registry.ignorable_unconstrained_points(
                entity.id
            )
            for pid in candidates:
                # Only allow exemption if the point is used exclusively
                # by this entity (usage count == 1)
                if usage_count.get(pid, 0) == 1:
                    allowed_unconstrained_ids.add(pid)

        # 4. Check all points
        for point in self.registry.points():
            if (
                not point.constrained
                and point.id not in allowed_unconstrained_ids
            ):
                # Unconstrained points must be in the exempt list
                return False
        return True

    @property
    def conflicting_constraints(self) -> list[Any]:
        """
        Returns a list of constraints that are currently marked as CONFLICTING.
        """
        return [
            c
            for c in self._constraints
            if constraint_info.status_of(c)
            is constraint_info.ConstraintStatus.CONFLICTING
        ]

    @property
    def has_conflicts(self) -> bool:
        """Returns True if any constraint has CONFLICTING status."""
        conflicting = constraint_info.ConstraintStatus.CONFLICTING
        return any(
            constraint_info.status_of(c) is conflicting
            for c in self._constraints
        )

    def to_dict(self, include_input_values: bool = False) -> dict[str, Any]:
        """Serializes the Sketch to a dictionary."""
        registry_dict = self.registry.to_dict()
        registry_dict.pop("constraints", None)
        self.flags.apply_to_registry_dict(registry_dict)
        return {
            "version": FORMAT_VERSION,
            "uid": self.uid,
            "type": self.asset_type_name,
            "name": self.name,
            "input_parameters": self.input_parameters.to_dict(
                include_value=include_input_values, include_metadata=False
            ),
            "params": self.params.to_dict(),
            "registry": registry_dict,
            "constraints": [c.to_dict() for c in self._constraints],
            "fills": [f.to_dict() for f in self.fills],
            "arrays": [p.to_dict() for p in self.arrays],
            "origin_id": self.origin_id,
            "hidden": self._hidden,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sketch":
        """Deserializes a dictionary into a Sketch instance."""
        file_version = data.get("version")
        if file_version is None:
            file_version = 1
        if file_version != FORMAT_VERSION:
            logger.warning(
                "Sketch file version %s differs from current version %s; "
                "loading may produce incorrect results",
                file_version,
                FORMAT_VERSION,
            )

        required_keys = ["params", "registry", "constraints", "origin_id"]
        if not all(key in data for key in required_keys):
            raise KeyError(
                "Sketch data is missing one of the required keys: "
                f"{required_keys}."
            )

        new_sketch = cls()
        new_sketch._uid = data.get("uid", str(uuid.uuid4()))
        new_sketch.name = data.get("name", "")

        # Handle backward compatibility for input_parameters
        if "input_parameters" in data:
            new_sketch.input_parameters = VarSet.from_dict(
                data["input_parameters"]
            )
            # Re-apply the default title and description, as they are not
            # serialized in the file.
            new_sketch.input_parameters.title = _DEFAULT_VARSET_TITLE
            new_sketch.input_parameters.description = (
                _DEFAULT_VARSET_DESCRIPTION
            )

        new_sketch.params = ParameterContext.from_dict(data["params"])
        migrated = migrate_sketch_data(data)
        new_sketch.registry = EntityRegistry.from_dict(migrated["registry"])
        new_sketch.origin_id = data["origin_id"]
        new_sketch._constraints = list(new_sketch.registry.constraints())
        for index, constraint in enumerate(new_sketch._constraints):
            stored = new_sketch.registry.constraint_dict(index)
            expression = stored.get("expression")
            if expression and hasattr(constraint, "set_expression"):
                constraint.expression = expression
            if hasattr(constraint, "set_user_visible"):
                constraint.set_user_visible(
                    stored.get("user_visible", True)
                )

        new_sketch.flags.absorb(new_sketch.registry)

        new_sketch.fills = []
        for f_data in data.get("fills", []):
            new_sketch.fills.append(Fill.from_dict(f_data))

        new_sketch.arrays = [
            Array.from_dict(p_data) for p_data in data.get("arrays", [])
        ]

        new_sketch._hidden = data.get("hidden", False)
        # Build coincident cache from loaded constraints.
        new_sketch._build_coincident_cache()
        return new_sketch

    def prune_arrays(self) -> None:
        """
        Removes arrays whose master geometry is gone. Groups themselves
        are never dissolved by deletion; deleting them does not
        dissolve the array as long as the master geometry still exists.
        """
        self.arrays = [
            array for array in self.arrays if array.prune(self.registry)
        ]

    def sync_arrays(self) -> None:
        """
        Re-applies every array whose guide or template has changed
        since the last sync.

        Array copies are static baked geometry: they don't carry
        solver constraints. When the user edits the guide (the guide
        path of a curve array, the guide circle of a circular array)
        or the template member, this method detects the change (by
        comparing cached signatures of the guide's and template's
        geometry — including Bezier control-point offsets) and re-runs
        ``EditArrayCommand`` to re-derive the copies. For circular
        arrays a guide edit similarity-transforms the template into
        the new guide frame first, so the template stays on the guide.
        Called after each solve.
        """
        for array in self.arrays:
            guide_sig = array.guide_signature(self.registry)
            if not guide_sig:
                continue
            template_sig = array.template_signature(self.registry)
            if array.signatures_changed(guide_sig, template_sig):
                self._reapply_array(array)
                template_sig = array.template_signature(self.registry)
            array.update_caches(guide_sig, template_sig)

    def _reapply_array(self, array_def: Array) -> None:
        """Re-distributes all copies of an array from its current
        guide and template geometry."""
        # Local import: the commands package __init__ pulls in modules
        # that import this module at runtime (e.g. fill), so a
        # module-level import would be circular.
        from .commands.edit_array import EditArrayCommand

        cmd = EditArrayCommand(
            self,
            array_def,
            array_def.make_strategy(self.registry),
            force_full_regen=True,
            capture_snapshot=False,
            old_frame=array_def._cached_guide_frame,
        )
        cmd.execute()

    def is_array_guide_radius_point(self, pid: int) -> bool:
        """
        True when the point is the radius point of an array's
        construction circle. Its position is governed by the radius
        dimension (the array's size definition), so drags must not
        move it.
        """
        return any(
            array.is_guide_radius_point(self.registry, pid)
            for array in self.arrays
        )

    def get_derived_point_ids(self) -> set[int]:
        """
        Returns the point ids whose position is owned by a master
        object rather than by the user: the member entities of all
        arrays (templates and their derived copies), including the
        standalone points each member carries (e.g. a rectangle's
        symmetry center). Hit-testing deprioritizes them in favor of
        coinciding user geometry.
        """
        pids: set[int] = set()
        for array in self.arrays:
            for _slot, entity_ids in array.members:
                for eid in entity_ids:
                    entity = self.registry.get_entity_object(eid)
                    if entity is not None:
                        pids.update(entity.get_point_ids())
            for standalone in array.standalone_pids.values():
                pids.update(standalone)
        return pids

    def find_standalone_point_ids(
        self, entity_ids: set[int] | list[int]
    ) -> list[int]:
        """
        Returns the standalone points that belong to the shape formed
        by the given entities: points referenced by no entity but tied
        to the group by constraints (e.g. a rectangle's symmetry
        center, held between two corners).

        A constraint internal to the group pulls in a point outside
        the group's entity points when that point is its only outside
        reference and at least two group points anchor it — a single
        shared point does not make a point part of the shape. Discovery
        iterates until stable so chains of standalone points converge.
        """
        group = set(entity_ids)
        group_pids: set[int] = set()
        for eid in group:
            entity = self.registry.get_entity_object(eid)
            if entity is not None:
                group_pids.update(entity.get_point_ids())
        extra: list[int] = []
        changed = True
        while changed:
            changed = False
            for constr in self._constraints:
                eids = constraint_info.referenced_entity_ids(
                    self.registry, constr
                )
                if not (eids <= group):
                    continue
                pids = constraint_info.referenced_point_ids(constr)
                missing = pids - group_pids
                if len(missing) == 1 and len(pids & group_pids) > 1:
                    group_pids.update(missing)
                    group.update(missing)
                    extra.extend(missing)
                    changed = True
        return extra

    def get_internal_constraints(
        self, entity_ids: set[int] | list[int]
    ) -> list[Any]:
        """
        Returns the constraints that hold a group of entities together
        as one shape: every point and entity they reference belongs to
        the group (e.g. the coincident edge/corner-arc endpoints or the
        line/arc tangencies of a rounded rectangle).

        Constraints referencing anything outside the group (the
        origin, other geometry) are global and excluded — and so are
        world-anchored orientation constraints (horizontal/vertical):
        they pin the shape to the world axes and would fight any
        array rotation. The exception is a horizontal+vertical pair
        on the same two points: the pair is a coincidence and holds
        the shape together, so it stays internal.
        """
        wanted = set(entity_ids)
        group_pids: set[int] = set()
        for eid in wanted:
            entity = self.registry.get_entity_object(eid)
            if entity is not None:
                group_pids.update(entity.get_point_ids())
            elif self.registry.get_point(eid) is not None:
                # Standalone Point (e.g. rectangle center) – include
                # its ID so its constraints count as internal.
                group_pids.add(eid)

        internal: list[Any] = []
        paired_axes = self._find_axis_coincidence_pairs()
        for constr in self._constraints:
            # Constraints pinned to the sketch origin are world-anchored.
            pids = constraint_info.referenced_point_ids(constr)
            if pids & {self.origin_id}:
                continue
            if constraint_info.is_world_anchored(
                constr
            ) and id(constr) not in paired_axes:
                continue
            eids = constraint_info.referenced_entity_ids(
                self.registry, constr
            )
            if not (pids & group_pids or eids & wanted):
                continue
            if pids <= group_pids and eids <= wanted:
                internal.append(constr)
        return internal

    def _find_axis_coincidence_pairs(self) -> set[int]:
        """Returns the IDs of horizontal/vertical constraints that
        have a perpendicular partner on the same two points.

        Such a pair is a coincidence in disguise: together the two
        constraints pin one point onto the other instead of anchoring
        the shape to the world axes, so they survive rotation and must
        not be stripped from array templates — dropping them would
        open closed shapes whose only closure is the pair.
        """
        verticals: dict[frozenset[EntityID], VerticalConstraint] = {}
        for constr in self._constraints:
            if isinstance(constr, VerticalConstraint):
                verticals[frozenset((constr.p1, constr.p2))] = constr
        paired: set[int] = set()
        for constr in self._constraints:
            if isinstance(constr, HorizontalConstraint):
                partner = verticals.get(frozenset((constr.p1, constr.p2)))
                if partner is not None:
                    paired.add(id(constr))
                    paired.add(id(partner))
        return paired

    @classmethod
    def from_file(cls, file_path: str | Path) -> "Sketch":
        """Deserializes a sketch from a JSON file (.rfs)."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_geometry(cls, geometry: Geometry) -> "Sketch":
        """
        Creates a Sketch from a Geometry object.

        The geometry can contain lines, arcs, and bezier curves.

        Args:
            geometry: The Geometry object to convert.

        Returns:
            A new Sketch instance with entities created from the geometry.
        """
        sketch = cls()

        if geometry.data is None or len(geometry.data) == 0:
            return sketch

        point_map: dict[tuple[float, float], EntityID] = {}

        def get_or_add_point(x: float, y: float) -> EntityID:
            key = (round(x, 6), round(y, 6))
            if key not in point_map:
                point_map[key] = sketch.add_point(x, y)
            return point_map[key]

        current_x, current_y = 0.0, 0.0
        current_pid: EntityID | None = None

        for cmd in geometry.iter_typed_commands():
            end_x, end_y = cmd.end[0], cmd.end[1]

            if isinstance(cmd, GeoMove):
                current_x, current_y = end_x, end_y
                current_pid = get_or_add_point(end_x, end_y)
            elif isinstance(cmd, GeoLine):
                if current_pid is None:
                    current_pid = get_or_add_point(current_x, current_y)
                end_pid = get_or_add_point(end_x, end_y)
                sketch.add_line(current_pid, end_pid)
                current_pid = end_pid
                current_x, current_y = end_x, end_y
            elif isinstance(cmd, GeoArc):
                if current_pid is None:
                    current_pid = get_or_add_point(current_x, current_y)
                end_pid = get_or_add_point(end_x, end_y)

                i_offset, j_offset, _ = cmd.center_offset
                clockwise = cmd.clockwise

                center_x = current_x + i_offset
                center_y = current_y + j_offset
                center_pid = get_or_add_point(center_x, center_y)

                sketch.add_arc(
                    current_pid, end_pid, center_pid, clockwise=clockwise
                )
                current_pid = end_pid
                current_x, current_y = end_x, end_y
            elif isinstance(cmd, GeoBezier):
                if current_pid is None:
                    current_pid = get_or_add_point(current_x, current_y)

                cp1_x, cp1_y, _ = cmd.control1
                cp2_x, cp2_y, _ = cmd.control2

                sketch.flags.set_waypoint_type(
                    current_pid, WaypointType.SMOOTH
                )

                end_pid = get_or_add_point(end_x, end_y)
                sketch.flags.set_waypoint_type(
                    end_pid, WaypointType.SMOOTH
                )

                start_pt = sketch.registry.get_point(current_pid)
                end_pt = sketch.registry.get_point(end_pid)
                cp1_offset = (cp1_x - start_pt[0], cp1_y - start_pt[1])
                cp2_offset = (cp2_x - end_pt[0], cp2_y - end_pt[1])
                sketch.add_bezier(
                    current_pid, end_pid, cp1=cp1_offset, cp2=cp2_offset
                )
                current_pid = end_pid
                current_x, current_y = end_x, end_y

        return sketch

    def set_param(self, name: str, value: str | float) -> None:
        """Define a parameter like 'width'=100 or 'height'='width/2'."""
        self.params.set(name, value)

    def add_point(self, x: float, y: float, fixed: bool = False) -> EntityID:
        """Adds a point. Returns its ID."""
        return self.registry.add_point(x, y, fixed)

    def add_line(
        self, p1: EntityID, p2: EntityID, construction: bool = False
    ) -> EntityID:
        """Adds a line segment between two point IDs."""
        return self.registry.add_line(p1, p2, construction)

    def add_arc(
        self,
        start: EntityID,
        end: EntityID,
        center: EntityID,
        clockwise: bool = False,
        construction: bool = False,
    ) -> EntityID:
        """Adds an arc defined by start, end, and center point IDs."""
        return self.registry.add_arc(
            start, end, center, clockwise, construction
        )

    def add_bezier(
        self,
        start: EntityID,
        end: EntityID,
        construction: bool = False,
        cp1: tuple[float, float] | None = None,
        cp2: tuple[float, float] | None = None,
    ) -> EntityID:
        """Adds a cubic bezier curve defined by start and end point IDs.

        Control points cp1 and cp2 are relative offsets from start and end
        points respectively.
        """
        return self.registry.add_bezier(start, end, construction, cp1, cp2)

    def add_circle(
        self, center: EntityID, radius_pt: EntityID, construction: bool = False
    ) -> EntityID:
        """Adds a circle defined by a center and a point on its radius."""
        return self.registry.add_circle(center, radius_pt, construction)

    def get_point_position(self, pid: EntityID) -> tuple[float, float] | None:
        """The point's coordinates, or None when it does not exist."""
        return self.registry.get_point(pid)

    def set_point_position(
        self, pid: EntityID, x: float, y: float
    ) -> None:
        """Moves a point to the given coordinates."""
        self.registry.set_point_position(pid, x, y)

    def get_entity(self, eid: EntityID) -> Any | None:
        """The entity object for an id, or None."""
        return self.registry.get_entity_object(eid)

    def remove_entities(self, entities_to_remove: Iterable[Any]) -> None:
        """
        Removes entities from the sketch and automatically cleans up any
        dependent fills.
        """
        ids_to_remove = [
            e.id if hasattr(e, "id") else e
            for e in entities_to_remove
        ]
        if not ids_to_remove:
            return
        self.registry.remove_entities(ids_to_remove)
        self.flags.drop_ids(set(ids_to_remove), set())
        self._validate_and_cleanup_fills()

    def remove_entities_by_id(self, entity_ids: Sequence[EntityID]) -> None:
        """
        Removes entities by their ids and cleans up dependent fills.
        """
        if not entity_ids:
            return
        self.registry.remove_entities(list(entity_ids))
        self.flags.drop_ids(set(entity_ids), set())
        self._validate_and_cleanup_fills()

    def remove_point_if_unused(self, pid: EntityID | None) -> bool:
        """
        Removes a point from the registry if it's not part of any entity.

        Args:
            pid: The point ID to remove. If None, returns False.

        Returns:
            True if the point was removed, False otherwise.
        """
        if pid is None:
            return False
        if not self.registry.is_point_used(pid):
            self.registry.remove_points([pid])
            self.flags.drop_ids(set(), {pid})
            return True
        return False

    def restore_structure(
        self,
        point_dicts: list[dict[str, Any]],
        entity_dicts: list[dict[str, Any]],
    ) -> None:
        """Restores points and entities (by their reference
        dictionaries, ids preserved) after a structural undo removed
        them. The registry is rebuilt from its serialized form; the
        sketch's constraint objects stay stable and are re-registered
        at their existing indices."""
        data = self.registry.to_dict()
        data.pop("constraints", None)
        points = data.setdefault("points", [])
        entities = data.setdefault("entities", [])
        known_points = {p.get("id") for p in points}
        known_entities = {e.get("id") for e in entities}
        for point_dict in point_dicts:
            if point_dict.get("id") not in known_points:
                points.append(point_dict)
        for entity_dict in entity_dicts:
            if entity_dict.get("id") not in known_entities:
                entities.append(entity_dict)
        self.registry = EntityRegistry.from_dict(data)
        self.flags._registry = self.registry
        for constraint in self._constraints:
            self.registry.add_constraint(constraint)

    def remap_entity_point_refs(
        self,
        pid_map: dict[int, int],
        entity_ids: set[int] | None = None,
    ) -> None:
        """Rewrites the point id references of entities through the
        given map, keeping entity ids stable. With ``entity_ids`` only
        those entities are rewritten.

        Raygeo entity objects are immutable, so the registry is
        rebuilt from its serialized form with the mapped ids; the
        sketch's constraint objects are re-registered at their stable
        indices. Used when a group's shared points are re-owned (array
        extraction, junction unsticking)."""
        if not pid_map:
            return
        registry_dict = self.registry.to_dict()
        registry_dict.pop("constraints", None)
        id_keys = (
            "p1_idx",
            "p2_idx",
            "start_idx",
            "end_idx",
            "center_idx",
            "radius_pt_idx",
            "radius_x_pt_idx",
            "radius_y_pt_idx",
            "origin_id",
            "width_id",
            "height_id",
            "handle_idx",
        )
        for entity_dict in registry_dict.get("entities", []):
            if entity_ids is not None and (
                entity_dict.get("id") not in entity_ids
            ):
                continue
            for key in id_keys:
                value = entity_dict.get(key)
                if value in pid_map:
                    entity_dict[key] = pid_map[value]
        self.registry = EntityRegistry.from_dict(registry_dict)
        self.flags._registry = self.registry
        for constraint in self._constraints:
            self.registry.add_constraint(constraint)

    def rebuild_constraints(
        self, constraints: Sequence[Any], pid_map: dict[int, int]
    ) -> None:
        """Replaces the given constraints with rebuilt objects whose
        point references are remapped through ``pid_map``."""
        for index, constraint in enumerate(self._constraints):
            if any(constraint is c for c in constraints):
                self.replace_constraint_at(
                    index, _rebuild_constraint(constraint, pid_map)
                )

    def copy_entity_flags(self, source_eid: int, target_eid: int) -> None:
        """Copies the app-level flags of one entity onto another
        (e.g. a text box's edited content onto its array copy)."""
        if self.flags.is_invisible(source_eid):
            self.flags.set_invisible(target_eid, True)
        if self.flags.is_array_copy(source_eid):
            self.flags.set_array_copy(target_eid, True)
        color = self.flags.fill_color(source_eid)
        if color is not None:
            self.flags.set_fill_color(target_eid, color)
        text = self.flags.text_properties(source_eid)
        for key, value in text.items():
            self.flags.set_text_property(target_eid, key, value)

    def _build_adjacency_list(self) -> dict[EntityID, list[dict[str, Any]]]:
        """
        Builds a map of point_id -> list of outgoing edges.
        Each edge dict contains: {'to': point_id, 'id': entity_id, 'fwd': bool}

        Coincident points are treated as the same node in the graph, so edges
        are added for all points in a coincident group.
        """
        adj = defaultdict(list)

        # Build a mapping from each point to its coincident group
        point_to_group: dict[EntityID, set[EntityID]] = {}
        for point in self.registry.points():
            if point.id not in point_to_group:
                coincident_group = self.get_coincident_points(point.id)
                for pid in coincident_group:
                    point_to_group[pid] = coincident_group

        for eid in self.registry.entity_ids():
            # Only edge entities participate in graph traversal; closed
            # loops (circles, ellipses) are handled separately.
            if not self.registry.is_edge_entity(eid):
                continue
            p_ids = self.registry.entity_endpoint_ids(eid)
            p1_id, p2_id = p_ids[0], p_ids[1]

            # Get the coincident groups for both endpoints
            group1 = point_to_group.get(p1_id, {p1_id})
            group2 = point_to_group.get(p2_id, {p2_id})

            # Add edges from all points in group1 to all points in group2
            for src in group1:
                for dst in group2:
                    if src != dst:
                        adj[src].append({"to": dst, "id": eid, "fwd": True})

            # Add edges from all points in group2 to all points in group1
            for src in group2:
                for dst in group1:
                    if src != dst:
                        adj[src].append({"to": dst, "id": eid, "fwd": False})
        return adj

    def _sort_edges_by_angle(
        self, adj: dict[EntityID, list[dict[str, Any]]]
    ) -> dict[EntityID, list[dict[str, Any]]]:
        """
        Sorts the outgoing edges at each node by angle (CCW).
        """
        sorted_adj = {}
        for p_id, edges in adj.items():
            edges_with_angle = []
            for edge in edges:
                tangent_vec = self.registry.tangent_at(edge["id"], p_id)
                if tangent_vec is None:
                    continue
                angle = math.atan2(tangent_vec[1], tangent_vec[0])
                edges_with_angle.append({"angle": angle, **edge})

            # Sort by angle [-pi, pi]
            edges_with_angle.sort(key=lambda x: x["angle"])
            sorted_adj[p_id] = edges_with_angle
        return sorted_adj

    def _get_next_edge_ccw(
        self,
        current_p_id: EntityID,
        incoming_entity_id: EntityID,
        incoming_fwd: bool,
        sorted_adj: dict[EntityID, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        """
        Given an incoming edge to a node, picks the next edge in CCW order
        (left-most turn) to traverse faces.
        """
        outgoing_edges = sorted_adj.get(current_p_id, [])
        if not outgoing_edges:
            return None

        # If we arrived via `incoming_entity_id` traveling `incoming_fwd`,
        # then looking back from the current node, that edge is the reverse.
        rev_fwd = not incoming_fwd

        try:
            # Find the edge entry in the current node's list that corresponds
            # to where we came from.
            idx = next(
                i
                for i, e in enumerate(outgoing_edges)
                if e["id"] == incoming_entity_id and e["fwd"] == rev_fwd
            )
            # Pick the previous edge in the sorted list (CCW rotation)
            next_idx = (idx - 1) % len(outgoing_edges)
            return outgoing_edges[next_idx]
        except StopIteration:
            return None

    def _calculate_loop_signed_area(
        self, loop: list[tuple[EntityID, bool]]
    ) -> float:
        """Calculates signed area of the loop using Shoelace formula."""
        if not loop:
            return 0.0

        # Special case for single-entity closed loops (circle, ellipse)
        if len(loop) == 1:
            eid = loop[0][0]
            if self.registry.is_closed_loop(eid):
                return (
                    self.registry.enclosed_signed_area(eid) or 0.0
                )

        points = []
        first_ent = self.registry.entity_point_ids(loop[0][0])
        if not first_ent:
            return 0.0
        first_fwd = loop[0][1]
        p_ids = first_ent
        curr_p_id = p_ids[0] if first_fwd else p_ids[1]

        for eid, fwd in loop:
            pt = self.registry.get_point(curr_p_id)
            if pt is None:
                return 0.0
            points.append(pt)
            p_ids = self.registry.entity_point_ids(eid)
            if not p_ids or len(p_ids) != 2:
                return 0.0
            curr_p_id = p_ids[1] if curr_p_id == p_ids[0] else p_ids[0]

        area = 0.0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            area += p1[0] * p2[1] - p2[0] * p1[1]
        area *= 0.5

        # Add contributions from Arcs (area between chord and arc)
        for eid, fwd in loop:
            if self.registry.entity_type(eid) != "arc":
                continue
            entity = self.registry.get_entity_object(eid)
            if entity is None:
                continue
            start = self.registry.get_point(entity.start_idx)
            end = self.registry.get_point(entity.end_idx)
            center = self.registry.get_point(entity.center_idx)
            if not (start and end and center):
                continue

            # Vectors from center
            r_vec_start = (start[0] - center[0], start[1] - center[1])
            r_vec_end = (end[0] - center[0], end[1] - center[1])
            radius_sq = r_vec_start[0] ** 2 + r_vec_start[1] ** 2

            # Calculate sweep angle of the arc definition
            ang_start = math.atan2(r_vec_start[1], r_vec_start[0])
            ang_end = math.atan2(r_vec_end[1], r_vec_end[0])

            if entity.clockwise:
                # CW: Start -> End decreases angle
                diff = ang_start - ang_end
            else:
                # CCW: Start -> End increases angle
                diff = ang_end - ang_start

            # Normalize to [0, 2pi)
            while diff < 0:
                diff += 2 * math.pi
            while diff >= 2 * math.pi:
                diff -= 2 * math.pi

            # Area of segment = 0.5 * r^2 * (theta - sin(theta))
            # This area is always positive.
            seg_area = 0.5 * radius_sq * (diff - math.sin(diff))

            # Determine sign contribution to the loop area (assumed
            # CCW positive).
            # If Arc is CCW and we traverse Fwd: Left turn. Add.
            # If Arc is CW and we traverse Fwd: Right turn. Subtract.
            # If Arc is CCW and we traverse Rev: Right turn. Subtract.
            # If Arc is CW and we traverse Rev: Left turn. Add.

            is_ccw_arc = not entity.clockwise
            is_left_turn = is_ccw_arc == fwd

            if is_left_turn:
                area += seg_area
            else:
                area -= seg_area

        return area

    def _find_all_closed_loops(self) -> list[list[tuple[EntityID, bool]]]:
        """
        Finds all closed loops (faces) in the sketch graph.
        """
        adj = self._build_adjacency_list()
        sorted_adj = self._sort_edges_by_angle(adj)

        loops = []
        visited_half_edges: set[tuple[EntityID, EntityID, bool]] = set()

        for p_start, edges in sorted_adj.items():
            for start_edge in edges:
                half_edge_key = (p_start, start_edge["id"], start_edge["fwd"])
                if half_edge_key in visited_half_edges:
                    continue

                loop: list[tuple[EntityID, bool]] = []
                loop_half_edges: list[tuple[EntityID, EntityID, bool]] = []
                curr_p = p_start
                curr_edge = start_edge

                for __ in range(self.registry.entity_count() + 1):
                    current_half_edge = (
                        curr_p,
                        curr_edge["id"],
                        curr_edge["fwd"],
                    )
                    if current_half_edge in visited_half_edges:
                        loop = []
                        break

                    loop.append((curr_edge["id"], curr_edge["fwd"]))
                    loop_half_edges.append(current_half_edge)

                    next_p = curr_edge["to"]

                    next_edge_info = self._get_next_edge_ccw(
                        next_p, curr_edge["id"], curr_edge["fwd"], sorted_adj
                    )

                    if not next_edge_info:
                        loop = []
                        break

                    next_key = (
                        next_p,
                        next_edge_info["id"],
                        next_edge_info["fwd"],
                    )
                    if next_key == half_edge_key:
                        break  # Loop closed

                    curr_p = next_p
                    curr_edge = next_edge_info
                else:
                    loop = []  # Loop did not close

                if loop and self._calculate_loop_signed_area(loop) > 1e-6:
                    loops.append(loop)
                    # Mark all half-edges from the valid loop as visited
                    visited_half_edges.update(loop_half_edges)

        # Add closed single entities (circles, ellipses) as loops
        for eid in self.registry.entity_ids():
            if self.registry.is_closed_loop(eid):
                loops.append([(eid, True)])

        return loops

    def _loop_to_polygon(
        self, loop: list[tuple[EntityID, bool]]
    ) -> list[tuple[float, float]]:
        """
        Converts a loop of entities into a list of 2D polygon vertices,
        sampling beziers and linearizing arcs.
        """
        polygon: list[tuple[float, float]] = []

        for eid, fwd in loop:
            if self.registry.get_entity_object(eid) is None:
                return []
            vertices = self.registry.to_polygon_vertices(eid, fwd)
            if not vertices:
                return []
            polygon.extend(vertices)

        return polygon

    def get_loop_at_point(
        self, mx: float, my: float
    ) -> list[tuple[EntityID, bool]] | None:
        """
        Finds the smallest closed loop containing the given point.
        Returns None if no loop contains the point.
        """
        all_loops = self._find_all_closed_loops()
        hit_loops = []

        for loop in all_loops:
            is_hit = False

            if len(loop) == 1:
                eid = loop[0][0]
                if self.registry.is_closed_loop(eid) and (
                    self.registry.contains_point(eid, mx, my)
                ):
                    is_hit = True
            else:
                polygon = self._loop_to_polygon(loop)
                if polygon and is_point_inside_polygon((mx, my), polygon):
                    is_hit = True

            if is_hit:
                area = abs(self._calculate_loop_signed_area(loop))
                hit_loops.append((area, loop))

        if not hit_loops:
            return None

        hit_loops.sort(key=lambda x: x[0])
        return hit_loops[0][1]

    # --- Constraint Shortcuts ---

    def get_coincident_points(self, start_pid: EntityID) -> set[EntityID]:
        """
        Finds all points transitively connected to start_pid via
        CoincidentConstraints. Returns a set including the starting
        point itself.

        Uses a precomputed cache that is rebuilt when constraints are
        modified, giving O(1) lookups after the first call.
        """
        if self._coincident_dirty or not self._coincident_cache:
            self._build_coincident_cache()
        return set(self._coincident_cache.get(start_pid, {start_pid}))

    def _build_coincident_cache(self) -> None:
        """Build the coincident-point group cache from current constraints."""
        adjacency: dict[EntityID, set[EntityID]] = defaultdict(set)
        for constr in self._constraints:
            if isinstance(constr, CoincidentConstraint):
                adjacency[constr.p1].add(constr.p2)
                adjacency[constr.p2].add(constr.p1)

        # Build connected components so each call is O(1).
        self._coincident_cache = {}
        for pid in adjacency:
            if pid in self._coincident_cache:
                continue
            group: set[EntityID] = set()
            stack = [pid]
            while stack:
                current = stack.pop()
                if current in group:
                    continue
                group.add(current)
                for neighbor in adjacency.get(current, ()):
                    if neighbor not in group:
                        stack.append(neighbor)
            frozen = frozenset(group)
            for p in group:
                self._coincident_cache[p] = frozen
        self._coincident_dirty = False

    def constrain_distance(
        self, p1: EntityID, p2: EntityID, dist: str | float
    ) -> DistanceConstraint:
        if isinstance(dist, str):
            constr = DistanceConstraint(p1, p2, 0.0)
            constr.expression = dist
        else:
            constr = DistanceConstraint(p1, p2, float(dist))
        self.add_constraint(constr)
        return constr

    def constrain_equal_distance(
        self, p1: EntityID, p2: EntityID, p3: EntityID, p4: EntityID
    ) -> None:
        """Enforces dist(p1, p2) == dist(p3, p4)."""
        self.add_constraint(EqualDistanceConstraint(p1, p2, p3, p4))

    def constrain_horizontal(self, p1: EntityID, p2: EntityID) -> None:
        self.add_constraint(HorizontalConstraint(p1, p2))

    def constrain_vertical(self, p1: EntityID, p2: EntityID) -> None:
        self.add_constraint(VerticalConstraint(p1, p2))

    def constrain_coincident(self, p1: EntityID, p2: EntityID) -> None:
        self.add_constraint(CoincidentConstraint(p1, p2))

    def constrain_point_on_line(
        self, point_id: EntityID, shape_id: EntityID
    ) -> None:
        self.add_constraint_from_dict(
            {
                "type": "PointOnLineConstraint",
                "point_id": point_id,
                "shape_id": shape_id,
            }
        )

    def constrain_radius(
        self, entity_id: EntityID, radius: str | float
    ) -> RadiusConstraint:
        data = {
            "type": "RadiusConstraint",
            "entity_id": entity_id,
            "value": float(radius) if not isinstance(radius, str) else 0.0,
        }
        if isinstance(radius, str):
            data["expression"] = radius
        constr = self.add_constraint_from_dict(data)
        return constr

    def constrain_diameter(
        self, circle_id: EntityID, diameter: str | float
    ) -> DiameterConstraint:
        data = {
            "type": "DiameterConstraint",
            "circle_id": circle_id,
            "value": (
                float(diameter) if not isinstance(diameter, str) else 0.0
            ),
        }
        if isinstance(diameter, str):
            data["expression"] = diameter
        constr = self.add_constraint_from_dict(data)
        return constr

    def constrain_perpendicular(self, l1: EntityID, l2: EntityID) -> None:
        self.add_constraint_from_dict(
            {"type": "PerpendicularConstraint", "e1_id": l1, "e2_id": l2}
        )

    def constrain_tangent(self, line: EntityID, shape: EntityID) -> None:
        self.add_constraint_from_dict(
            {
                "type": "TangentConstraint",
                "line_id": line,
                "shape_id": shape,
            }
        )

    def constrain_equal_length(self, entity_ids: list[EntityID]) -> None:
        """Enforces equal length/radius between two or more entities."""
        if len(entity_ids) < 2:
            return
        self.add_constraint_from_dict(
            {
                "type": "EqualLengthConstraint",
                "entity_ids": list(entity_ids),
            }
        )

    def constrain_symmetry(
        self, point_ids: list[EntityID], entity_ids: list[EntityID]
    ) -> None:
        """
        Enforces symmetry.
        - If 3 points: The first in point_ids is treated as the center.
        - If 2 points + 1 Line: The line is the axis.
        """
        if len(point_ids) == 3 and not entity_ids:
            # 3 Points: First is Center, other two are symmetric
            center = point_ids[0]
            p1 = point_ids[1]
            p2 = point_ids[2]
            self.add_constraint(
                SymmetryConstraint(
                    p1, p2, SymmetryTarget.Point(center=center)
                )
            )

        elif len(point_ids) == 2 and len(entity_ids) == 1:
            # 2 Points + 1 Line: Line is Axis
            p1 = point_ids[0]
            p2 = point_ids[1]
            axis = entity_ids[0]
            self.add_constraint_from_dict(
                {
                    "type": "SymmetryConstraint",
                    "p1": p1,
                    "p2": p2,
                    "axis": axis,
                }
            )

    # --- Manipulation & Processing ---

    def move_point(self, pid: EntityID, x: float, y: float) -> bool:
        """
        Attempts to move a point to a new location and resolve constraints.
        Returns True if the point was moved, False if it is locked/constrained.
        """
        if self.registry.get_point(pid) is None:
            return False

        if self.registry.point_is_fixed(pid):
            return False

        # Backend Logic: If the solver has determined this point has 0 degrees
        # of freedom (fully constrained), we reject kinematic movement.
        if self.registry.point_constrained(pid):
            return False

        # Perturbation Strategy: Update initial guess, then solve.
        self.registry.set_point_position(pid, x, y)

        return self.solve()

    def solve(
        self,
        extra_constraints: Sequence[Any] | None = None,
        update_constraint_status: bool = True,
        variable_overrides: dict[str, Any] | None = None,
        excluded_constraints: set[int] | None = None,
        point_scope: set[EntityID] | None = None,
    ) -> bool:
        """
        Resolves all constraints.

        Args:
            extra_constraints: A list of temporary constraints to add for this
                solve, e.g., for dragging.
            update_constraint_status: If True, re-calculates the degrees of
                freedom for all points and entities after a successful solve.
            variable_overrides: A dictionary of parameter values to use for
                this solve only, without permanently changing the sketch's
                parameters. e.g., `{'width': 150.0}`.
            excluded_constraints: Indices (into the sketch's constraints) of
                constraints to skip for this solve, e.g. an array's radius
                dimension while a member is being dragged.
            point_scope: If given, only these points are optimized and
                only constraints referencing them are applied; points
                outside keep their positions. The scope must be a union
                of connected components of the constraint graph (see
                compute_constraint_components) plus the dragged points.
                Constraint status is never updated for scoped solves.
                Falls back to a global solve if the scope references
                points that no longer exist.

        Returns:
            True if the solver converged successfully.
        """
        # Guard against recursive solves triggered by notify_update()
        # during sync_arrays → EditArrayCommand.execute().  The
        # outer solve already converged; the registry edits applied by
        # the array re-apply are valid and will be picked up on the
        # next user-initiated solve.
        if self._solving:
            return True
        self._solving = True
        solve_started = time.perf_counter()
        params_done = prep_done = solver_done = arrays_done = solve_started
        success = False
        solver: Solver | None = None
        all_constraints: list[Any] = []
        update_status = False
        solve_completed = False
        scope = point_scope
        try:
            # A scope referencing deleted points (e.g. after an array
            # re-apply removed geometry mid-drag) cannot be honored;
            # solve globally instead.
            if scope is not None:
                existing = {p.id for p in self.registry.points()}
                if not scope.issubset(existing):
                    scope = None

            # Step 1: Create a disposable ParameterContext clone for this
            # solve.
            solve_params = ParameterContext.from_dict(self.params.to_dict())

            # Step 2: Build the seed dictionary, starting with defaults from
            # the VarSet, then applying instance-specific overrides.
            initial_values = {}
            if self.input_parameters:
                initial_values.update(self.input_parameters.get_values())
            if variable_overrides:
                initial_values.update(variable_overrides)

            self._last_solve_values = dict(initial_values)
            self._resolved_text_cache = {}

            # Step 3: Evaluate all expressions from scratch using the temporary
            # context, seeded with the combined values.
            solve_params.evaluate_all(initial_values=initial_values)
            ctx = solve_params.get_all_values()

            # Cache the solved context (with template functions) for
            # reuse by _resolve_text_content, avoiding redundant
            # ParameterContext rebuilds per text box.
            self._solved_ctx = ctx.copy()
            self._solved_ctx.update(get_template_functions())
            params_done = time.perf_counter()

            # Step 4: Update constraints with the final, resolved values.
            excluded = excluded_constraints or set()
            kept_indices = [
                i for i in range(len(self._constraints)) if i not in excluded
            ]
            all_constraints = [self._constraints[i] for i in kept_indices]
            if scope is not None:
                # For a valid scope, every constraint touching a scope
                # point lies fully inside it, so filtering by overlap
                # keeps exactly the constraints that can move scope
                # points.
                all_constraints = [
                    c
                    for c in all_constraints
                    if get_referenced_points(self.registry, c) & scope
                ]
            if extra_constraints:
                extra = list(extra_constraints)
                if scope is not None:
                    extra = [
                        c
                        for c in extra
                        if get_referenced_points(self.registry, c) & scope
                    ]
                all_constraints = all_constraints + extra
            for c in all_constraints:
                if constraint_info.has_expression(c):
                    constraint_info.update_from_context(c, ctx)

            # --- Solver Stabilization ---
            # Add weak, temporary constraints to every non-fixed point
            # the solve references, pulling it towards its current
            # location. This acts as an "inertia" term, encouraging the
            # solver to find the solution closest to the current state
            # and preventing large, unexpected geometric jumps. Points
            # no constraint references cannot move anyway; pinning them
            # would only grow the solver's state vector.
            touched: set[EntityID] = set()
            for c in all_constraints:
                touched |= get_referenced_points(self.registry, c)
            stabilizer_constraints = []
            hold_weight = 1e-4
            for point in self.registry.points():
                if (
                    point.fixed
                    or point.id not in touched
                    or (scope is not None and point.id not in scope)
                ):
                    continue
                stabilizer_constraints.append(
                    DragConstraint(
                        point.id,
                        point.x,
                        point.y,
                        weight=hold_weight,
                    )
                )

            prep_done = time.perf_counter()

            # Step 5: Run the solver with the disposable, correctly
            # evaluated context.
            solver = Solver(
                self.registry,
                all_constraints,
                auxiliary_constraints=stabilizer_constraints,
                point_filter=scope,
            )
            update_status = update_constraint_status and scope is None
            success = solver.solve(update_dof=update_status)
            solve_completed = True
            solver_done = time.perf_counter()

        except (np.linalg.LinAlgError, ValueError):
            logger.exception("Sketch solve failed")
            success = False

        # Re-apply arrays whose guide or template has moved. This is
        # done after solving so the solver's final point positions are
        # used, which keeps the members following drags live. Keep
        # _solving=True so that notify_update() inside the command
        # does not trigger a recursive solve.
        try:
            self.sync_arrays()
        finally:
            self._solving = False
        arrays_done = time.perf_counter()

        # Step 6: Update constraint conflict status. This must happen
        # AFTER the array sync: re-anchoring an array template can
        # violate constraints the solver had just satisfied (the
        # template's placement belongs to the array), so residuals are
        # evaluated against the final geometry. Solver indices are
        # mapped back to sketch constraint positions, since
        # sync_arrays may have changed the constraint list.
        if solve_completed and update_status and solver is not None:
            conflicting = solver.get_conflicting_constraints()
            mapped = {
                kept_indices[i]
                for i in conflicting
                if 0 <= i < len(kept_indices)
            }
            self._apply_conflict_status(mapped)

        logger.debug(
            "Sketch.solve: points=%d constraints=%d scope=%s "
            "params=%.1fms prep=%.1fms solver=%.1fms arrays=%.1fms "
            "total=%.1fms ok=%s",
            len(self.registry.points()),
            len(all_constraints),
            len(scope) if scope is not None else None,
            (params_done - solve_started) * 1000.0,
            (prep_done - params_done) * 1000.0,
            (solver_done - prep_done) * 1000.0,
            (arrays_done - solver_done) * 1000.0,
            (time.perf_counter() - solve_started) * 1000.0,
            success,
        )
        return success

    def _apply_conflict_status(self, conflicting_indices: set[int]) -> None:
        """
        Updates the conflict status of all constraints based on solver
        results. Indices refer to positions in the sketch's constraint
        list. Constraints with significant residual error are marked as
        CONFLICTING.
        """
        status = constraint_info.ConstraintStatus
        for idx, constraint in enumerate(self._constraints):
            current = constraint_info.status_of(constraint)
            if idx in conflicting_indices:
                if current is not status.ERROR:
                    constraint_info.set_status(
                        constraint, status.CONFLICTING
                    )
            elif current is status.CONFLICTING:
                constraint_info.set_status(constraint, status.VALID)

    def _resolve_text_content(self, entity: TextBox) -> str | None:
        """
        Resolves template expressions in a text box's content using
        the sketch's current parameter values. Returns None if the
        content has no templates or resolution fails.

        Results are cached per entity so that volatile expressions
        (e.g. uuid4()) produce the same value across multiple calls
        within a single solve cycle. Each entry records the source
        content it was resolved from; a cached entry whose source no
        longer matches the entity's current content is re-resolved,
        otherwise reverting an edit would keep rendering stale text.
        """
        source_content = self.flags.text_content(
            entity.id, entity.content
        )
        cached = self._resolved_text_cache.get(entity.id)
        if isinstance(cached, (tuple, list)):
            cached_source, resolved = cached
            if cached_source == source_content:
                return resolved

        if not source_content:
            self._resolved_text_cache[entity.id] = ("", None)
            return None

        try:
            # Use the cached context from solve() when available,
            # avoiding a fresh ParameterContext per text box.
            if self._solved_ctx is not None:
                ctx = self._solved_ctx
            else:
                solve_params = ParameterContext.from_dict(
                    self.params.to_dict()
                )
                initial_values = dict(self._last_solve_values)
                if not initial_values and self.input_parameters:
                    initial_values.update(self.input_parameters.get_values())
                solve_params.evaluate_all(initial_values=initial_values)
                ctx = solve_params.get_all_values()
                ctx.update(get_template_functions())

            expr_map = ExpressionMap(ctx)
            resolved = expr_map.format(source_content)
            logger.debug(
                f"_resolve_text_content: content="
                f"{source_content!r} -> {resolved!r} "
                f"values={list(ctx.keys())[:5]}"
            )
            self._resolved_text_cache[entity.id] = (
                source_content,
                resolved,
            )
            return resolved
        except (KeyError, IndexError, ValueError) as e:
            logger.debug(
                f"Template resolution failed for '{source_content}': {e}"
            )
            self._resolved_text_cache[entity.id] = (
                source_content,
                None,
            )
            return None

    def resolved_text_box(self, entity: TextBox) -> TextBox:
        """A text box carrying the resolved template content and the
        app-level font properties, sharing the entity's frame points.

        The raygeo entity object is an immutable snapshot, so template
        substitution renders through a stand-in box with the same
        frame points."""
        props = self.flags.text_properties(entity.id)
        content = props.get("content", entity.content)
        resolved = self._resolve_text_content(entity)
        return TextBox(
            entity.origin_id,
            entity.width_id,
            entity.height_id,
            content=resolved if resolved is not None else content,
            font_family=props.get("font_family", entity.font_family),
            font_size=props.get("font_size", entity.font_size),
            bold=props.get("bold", entity.bold),
            italic=props.get("italic", entity.italic),
        )

    def to_geometry(self) -> Geometry:
        """
        Converts the solved sketch into a Geometry object.
        Links separate entities into continuous paths where possible.
        """
        geo = Geometry()
        registry = self.registry

        # 1. Identify chainable vs standalone
        chainable: list[tuple[EntityID, str]] = []
        standalone: list[tuple[EntityID, str]] = []

        for eid in registry.entity_ids():
            entity_type = registry.entity_type(eid)
            entity = registry.get_entity_object(eid)
            if entity is None or entity.construction:
                continue
            if self.flags.is_invisible(eid):
                continue
            if entity_type in ("line", "arc", "bezier"):
                chainable.append((eid, entity_type))
            else:
                standalone.append((eid, entity_type))

        # 2. Add standalone geometry (Circles, Polygons, Text)
        for eid, entity_type in standalone:
            entity = registry.get_entity_object(eid)
            if entity_type == "text_box":
                resolved_box = self.resolved_text_box(entity)
                geo.extend(resolved_box.to_geometry(registry))
            else:
                geo.extend(registry.to_geometry(eid))

        if not chainable:
            return geo

        # 3. Build Connectivity Graph for Lines/Arcs
        # Use simple Union-Find to group coincident points
        parent = {p.id: p.id for p in registry.points()}

        def find(i):
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        # Apply Coincident Constraints
        for c in self._constraints:
            if (
                isinstance(c, CoincidentConstraint)
                and c.p1 in parent
                and c.p2 in parent
            ):
                # Points exist (sanity check)
                union(c.p1, c.p2)

        adj = defaultdict(list)
        for eid, _type in chainable:
            p_ids = registry.entity_endpoint_ids(eid)
            if not p_ids or len(p_ids) != 2:
                continue
            u, v = p_ids[0], p_ids[1]

            root_u = find(u)
            root_v = find(v)
            adj[root_u].append((eid, root_v))
            adj[root_v].append((eid, root_u))

        # 4. Traverse Graph to build continuous paths
        visited = set()

        # Helper to get start/end group IDs
        def get_endpoints(ent_id):
            p_ids = registry.entity_endpoint_ids(ent_id)
            if not p_ids or len(p_ids) != 2:
                return -1, -1
            return find(p_ids[0]), find(p_ids[1])

        for start_eid, _type in chainable:
            if start_eid in visited:
                continue

            # Start a new chain
            visited.add(start_eid)

            # Seed direction
            u, v = get_endpoints(start_eid)

            # Grow Right (from v)
            right_list = []
            curr = v
            while True:
                found = None
                for cand, neighbor in adj[curr]:
                    if cand not in visited:
                        found = (cand, neighbor)
                        break
                if found:
                    cand, next_node = found
                    visited.add(cand)
                    # Direction check: if c_u == curr, then Forward (u->v)
                    c_u, _ = get_endpoints(cand)
                    is_fwd = c_u == curr
                    right_list.append((cand, is_fwd))
                    curr = next_node
                else:
                    break

            # Grow Left (from u)
            left_list = []
            curr = u
            while True:
                found = None
                for cand, neighbor in adj[curr]:
                    if cand not in visited:
                        found = (cand, neighbor)
                        break
                if found:
                    cand, next_node = found
                    visited.add(cand)
                    # We are growing backwards from u.
                    # cand connects next_node <-> curr(u).
                    # We want flow: next_node -> curr.
                    # If cand is Forward (c_u -> c_v), then c_u must be
                    # next_node.
                    c_u, _ = get_endpoints(cand)
                    is_fwd = c_u == next_node
                    left_list.append((cand, is_fwd))
                    curr = next_node
                else:
                    break

            # Assemble: Reversed(Left) -> Seed -> Right
            final_chain = (
                list(reversed(left_list)) + [(start_eid, True)] + right_list
            )

            # Generate Geometry
            first_eid, first_fwd = final_chain[0]
            p_ids = registry.entity_endpoint_ids(first_eid)
            s_id = p_ids[0] if first_fwd else p_ids[1]

            start_pt = registry.get_point(s_id)
            geo.move_to(start_pt[0], start_pt[1])

            for ent_id, fwd in final_chain:
                registry.append_to_geometry(geo, ent_id, fwd)

        return geo

    def get_fill_render_data(
        self, exclude_ids: set[EntityID] | None = None
    ) -> list[FillRenderData]:
        """
        Generates FillRenderData objects for all defined fills.

        Each FillRenderData contains the geometry and styling information
        needed to render a fill region.

        Args:
            exclude_ids: Optional set of entity IDs to exclude from fill
                generation (e.g., text boxes being edited).
        """
        if exclude_ids is None:
            exclude_ids = set()

        render_data = []
        for fill in self.fills:
            if not fill.boundary:
                continue

            geo = self._create_fill_geometry(fill, exclude_ids)
            if geo is not None:
                render_data.append(
                    FillRenderData(
                        geometry=geo,
                        style=fill.style,
                        color=fill.color,
                        gradient_stops=fill.gradient_stops,
                        gradient_angle=fill.gradient_angle,
                    )
                )

        for eid in self.registry.entity_ids():
            if eid in exclude_ids:
                continue
            entity = self.registry.get_entity_object(eid)
            if entity is None or entity.construction:
                continue
            if entity.type_name() != "text_box":
                continue
            resolved_box = self.resolved_text_box(entity)
            text_geo = resolved_box.to_geometry(self.registry)
            if text_geo:
                color = self.flags.fill_color(eid) or DEFAULT_FILL_COLOR
                render_data.append(
                    FillRenderData(
                        geometry=text_geo,
                        style=FillStyle.SOLID,
                        color=color,
                    )
                )

        return render_data

    def _create_fill_geometry(
        self, fill: "Fill", exclude_ids: set[EntityID]
    ) -> Geometry | None:
        """Create geometry for a single fill."""
        if len(fill.boundary) == 1:
            eid, _ = fill.boundary[0]
            if eid in exclude_ids:
                return None
            if self.registry.get_entity_object(eid) is not None:
                return self.registry.to_geometry(eid)
            return None

        try:
            first_eid, first_fwd = fill.boundary[0]
            if first_eid in exclude_ids:
                return None
            p_ids = self.registry.entity_point_ids(first_eid)
            if not p_ids:
                return None
            start_pid = p_ids[0] if first_fwd else p_ids[1]
            start_pt = self.registry.get_point(start_pid)
            if start_pt is None:
                return None

            geo = Geometry()
            geo.move_to(start_pt[0], start_pt[1])

            for eid, fwd in fill.boundary:
                if eid in exclude_ids:
                    return None
                if self.registry.get_entity_object(eid) is None:
                    return None
                self.registry.append_to_geometry(geo, eid, fwd)

            return geo

        except (IndexError, AttributeError):
            return None

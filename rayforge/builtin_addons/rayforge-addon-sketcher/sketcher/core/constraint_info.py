"""Constraint metadata and helpers over the raygeo constraint classes.

The raygeo constraint objects carry pure solver geometry; the
user-facing vocabulary (titles, subtitles, editing hints, dragging,
applicability) lives here, dispatched on the constraint's concrete
class.
"""

from __future__ import annotations

import math
from enum import IntEnum
from gettext import gettext as _
from locale import format_string
from typing import TYPE_CHECKING, Any

from raygeo.sketch.constraints.angle import AngleConstraint
from raygeo.sketch.constraints.aspect_ratio import AspectRatioConstraint
from raygeo.sketch.constraints.coincident import CoincidentConstraint
from raygeo.sketch.constraints.collinear import CollinearConstraint
from raygeo.sketch.constraints.diameter import DiameterConstraint
from raygeo.sketch.constraints.distance import DistanceConstraint
from raygeo.sketch.constraints.drag import DragConstraint
from raygeo.sketch.constraints.equal_distance import (
    EqualDistanceConstraint,
)
from raygeo.sketch.constraints.equal_length import EqualLengthConstraint
from raygeo.sketch.constraints.horizontal import HorizontalConstraint
from raygeo.sketch.constraints.parallelogram import ParallelogramConstraint
from raygeo.sketch.constraints.perpendicular import PerpendicularConstraint
from raygeo.sketch.constraints.point_on_curve import (
    PointOnCurveConstraint,
)
from raygeo.sketch.constraints.point_on_line import PointOnLineConstraint
from raygeo.sketch.constraints.radius import RadiusConstraint
from raygeo.sketch.constraints.rotational import RotationalConstraint
from raygeo.sketch.constraints.symmetry import SymmetryConstraint
from raygeo.sketch.constraints.tangent import TangentConstraint
from raygeo.sketch.constraints.vertical import VerticalConstraint

from rayforge.core.expression import safe_evaluate

from .types import EntityID

if TYPE_CHECKING:
    from raygeo.sketch.constraints import Constraint
    from raygeo.sketch.constraints.perpendicular import PerpendicularTarget
    from raygeo.sketch.constraints.symmetry import SymmetryTarget
    from raygeo.sketch.entities import Entity
    from raygeo.sketch.registry import EntityRegistry

    from .selection import SketchSelection
    from .sketch import Sketch


class ConstraintStatus(IntEnum):
    """Validation status of a constraint, as shown in the UI."""

    VALID = 1
    EXPRESSION_BASED = 2
    ERROR = 3
    CONFLICTING = 4


def status_of(constraint: Constraint) -> ConstraintStatus:
    """Reads the constraint's status; unpersisted values fall back to
    VALID."""
    try:
        return ConstraintStatus(constraint.status)
    except ValueError:
        return ConstraintStatus.VALID


def set_status(constraint: Constraint, status: ConstraintStatus) -> None:
    """Writes the UI status ordinal onto the constraint."""
    constraint.status = status.value


def _format_value(value: float) -> str:
    return f"{float(value):.1f}"


def _format_coord(x: float, y: float) -> str:
    return format_string("%.1f/%.1f", (x, y), grouping=True)


_TYPE_KEYS: dict[type, str] = {
    HorizontalConstraint: "horiz",
    VerticalConstraint: "vert",
    DistanceConstraint: "dist",
    RadiusConstraint: "radius",
    DiameterConstraint: "diameter",
    PerpendicularConstraint: "perp",
    TangentConstraint: "tangent",
    EqualLengthConstraint: "equal",
    CoincidentConstraint: "coincident",
    PointOnLineConstraint: "point_on_line",
    PointOnCurveConstraint: "point_on_curve",
    SymmetryConstraint: "symmetry",
    AspectRatioConstraint: "aspect_ratio",
    AngleConstraint: "angle",
    RotationalConstraint: "rotational",
}

_TYPE_NAMES: dict[type, str] = {
    HorizontalConstraint: _("Horizontal"),
    VerticalConstraint: _("Vertical"),
    DistanceConstraint: _("Distance"),
    RadiusConstraint: _("Radius"),
    DiameterConstraint: _("Diameter"),
    PerpendicularConstraint: _("Perpendicular"),
    TangentConstraint: _("Tangent"),
    EqualLengthConstraint: _("Equal"),
    CoincidentConstraint: _("Coincident"),
    PointOnLineConstraint: _("Point on Line"),
    PointOnCurveConstraint: _("Point on Curve"),
    SymmetryConstraint: _("Symmetric"),
    AspectRatioConstraint: _("Aspect Ratio"),
    AngleConstraint: _("Angle"),
    RotationalConstraint: _("Rotational Array"),
    EqualDistanceConstraint: _("Equal Distance"),
    CollinearConstraint: _("Collinear"),
    ParallelogramConstraint: _("Parallelogram"),
}

_DICT_TYPE_NAMES: dict[type, str] = {
    AngleConstraint: "AngleConstraint",
    AspectRatioConstraint: "AspectRatioConstraint",
    CoincidentConstraint: "CoincidentConstraint",
    CollinearConstraint: "CollinearConstraint",
    DiameterConstraint: "DiameterConstraint",
    DistanceConstraint: "DistanceConstraint",
    DragConstraint: "DragConstraint",
    EqualDistanceConstraint: "EqualDistanceConstraint",
    EqualLengthConstraint: "EqualLengthConstraint",
    HorizontalConstraint: "HorizontalConstraint",
    ParallelogramConstraint: "ParallelogramConstraint",
    PerpendicularConstraint: "PerpendicularConstraint",
    PointOnCurveConstraint: "PointOnCurveConstraint",
    PointOnLineConstraint: "PointOnLineConstraint",
    RadiusConstraint: "RadiusConstraint",
    RotationalConstraint: "RotationalConstraint",
    SymmetryConstraint: "SymmetryConstraint",
    TangentConstraint: "TangentConstraint",
    VerticalConstraint: "VerticalConstraint",
}

_EDIT_SUBTITLES: dict[type, str] = {
    DistanceConstraint: _("Enter length or expression."),
    HorizontalConstraint: _("Enter length or expression."),
    VerticalConstraint: _("Enter length or expression."),
    RadiusConstraint: _("Enter radius or expression (e.g. 'width/2')."),
    DiameterConstraint: _("Enter diameter or expression."),
    AngleConstraint: _("Enter angle in degrees or expression."),
}


def type_key(constraint: Constraint) -> str | None:
    """The short key identifying the constraint type, if any."""
    return _TYPE_KEYS.get(type(constraint))


def dict_type_name(constraint: Constraint) -> str:
    """The constraint type string used in serialized dictionaries."""
    return _DICT_TYPE_NAMES.get(type(constraint), type(constraint).__name__)


def get_type_name(constraint: Constraint) -> str:
    return _TYPE_NAMES.get(type(constraint), type(constraint).__name__)


def is_world_anchored(constraint: Constraint) -> bool:
    """Whether the constraint pins geometry to the global axes."""
    return isinstance(constraint, (HorizontalConstraint, VerticalConstraint))


def is_mirror_compatible(constraint: Constraint) -> bool:
    """Whether the constraint survives mirroring of its geometry."""
    return not (
        isinstance(constraint, AngleConstraint) and constraint.expression
    )


def user_visible(constraint: Constraint) -> bool:
    """Whether the constraint is surfaced in lists and hit testing."""
    getter = getattr(constraint, "user_visible", None)
    if getter is None:
        return True
    if callable(getter):
        return bool(getter())
    return bool(getter)


def has_expression(constraint: Constraint) -> bool:
    """Whether the constraint carries an expression string."""
    expression = (
        constraint.expression if hasattr(constraint, "expression") else None
    )
    return bool(expression)


def get_expression(constraint: Constraint) -> str | None:
    """The constraint's expression string, if any."""
    return constraint.expression if hasattr(constraint, "expression") else None


def get_title(constraint: Constraint) -> str:
    """A human-readable title for the constraint."""
    name = get_type_name(constraint)
    if isinstance(
        constraint, (DistanceConstraint, RadiusConstraint, DiameterConstraint)
    ):
        return f"{name} {_format_value(constraint.value)}"
    if isinstance(constraint, AngleConstraint):
        return f"{name} {_format_value(constraint.value)}°"
    if isinstance(constraint, AspectRatioConstraint):
        return f"{name} {constraint.ratio:.2f}"
    if isinstance(constraint, RotationalConstraint):
        angle_deg = math.degrees(constraint.angle_rad)
        return f"{name} {angle_deg:.1f}°"
    return name


def get_subtitle(registry: EntityRegistry, constraint: Constraint) -> str:
    """A human-readable subtitle describing the constrained geometry."""

    def pt(pid: EntityID):
        return registry.get_point(pid)

    def coord(p) -> str:
        return _format_coord(p[0], p[1])

    if isinstance(
        constraint,
        (
            DistanceConstraint,
            AspectRatioConstraint,
            HorizontalConstraint,
            VerticalConstraint,
            SymmetryConstraint,
        ),
    ):
        p1, p2 = pt(constraint.p1), pt(constraint.p2)
        if p1 and p2:
            return _("From {} to {}").format(coord(p1), coord(p2))
        return ""
    if isinstance(constraint, CoincidentConstraint):
        p1 = pt(constraint.p1)
        if p1:
            return _("At {}").format(coord(p1))
        return ""
    if isinstance(constraint, CollinearConstraint):
        pts = [pt(pid) for pid in constraint.referenced_points()]
        if all(p is not None for p in pts):
            return _(", ").join(coord(p) for p in pts)
        return ""
    if isinstance(constraint, EqualDistanceConstraint):
        p1, p2 = pt(constraint.p1), pt(constraint.p2)
        p3, p4 = pt(constraint.p3), pt(constraint.p4)
        if p1 and p2 and p3 and p4:
            return _("{}-{} and {}-{}").format(
                coord(p1), coord(p2), coord(p3), coord(p4)
            )
        return ""
    if isinstance(constraint, DiameterConstraint):
        center = pt(constraint.center)
        if center:
            return _("Circle at {}").format(coord(center))
        return ""
    if isinstance(constraint, RadiusConstraint):
        center = pt(constraint.center)
        if center:
            entity = _radius_entity_of(registry, constraint)
            kind = type(entity).__name__ if entity else _("Radius")
            return _("{} at {}").format(kind, coord(center))
        return ""
    if isinstance(constraint, TangentConstraint):
        center = pt(constraint.center)
        if center:
            return _("Line to circle at {}").format(coord(center))
        return ""
    if isinstance(constraint, (PointOnLineConstraint, PointOnCurveConstraint)):
        position = pt(constraint.point)
        if position:
            return _("Point at {}").format(coord(position))
        return ""
    if isinstance(constraint, ParallelogramConstraint):
        p_origin = pt(constraint.p_origin)
        if p_origin:
            return _("Origin at {}").format(coord(p_origin))
        return ""
    if isinstance(constraint, EqualLengthConstraint):
        if len(constraint.pairs) >= 2:
            return _("{} entities").format(len(constraint.pairs))
        return ""
    if isinstance(constraint, PerpendicularConstraint):
        return _("Between two entities")
    if isinstance(constraint, AngleConstraint):
        return _("Between two lines")
    return ""


def get_edit_subtitle(constraint: Constraint) -> str:
    """The editing hint shown in the constraint edit dialog."""
    return _EDIT_SUBTITLES.get(
        type(constraint), _("Enter value or expression.")
    )


def get_draggable_point(constraint: Constraint) -> EntityID | None:
    """The point that can be dragged to manipulate the constraint."""
    if isinstance(constraint, CoincidentConstraint):
        return constraint.p1
    if isinstance(constraint, (PointOnLineConstraint, PointOnCurveConstraint)):
        return constraint.point
    return None


def _is_line(entity: Entity) -> bool:
    return entity.type_name() == "line"


def _is_radius_entity(entity: Entity) -> bool:
    return entity.type_name() in ("circle", "arc")


def _has_characteristic_length(entity: Entity) -> bool:
    return entity.type_name() in ("line", "arc", "bezier", "circle", "ellipse")


def can_apply(
    constraint_type: type,
    selection: SketchSelection,
    sketch: Sketch | None = None,
) -> bool:
    """Whether the constraint type can be created from the selection."""
    registry = sketch.registry if sketch is not None else None
    entities = selection.resolve_entities(registry)
    if constraint_type in (
        HorizontalConstraint,
        VerticalConstraint,
        DistanceConstraint,
        AngleConstraint,
    ):
        if len(selection.point_ids) == 2 and not selection.entity_ids:
            return True
        if selection.point_ids:
            return False
        if entities is None:
            return False
        return _applies_to_entities(constraint_type, entities)
    if constraint_type is CoincidentConstraint:
        return len(selection.point_ids) == 2 and not selection.entity_ids
    if constraint_type in (PointOnLineConstraint, PointOnCurveConstraint):
        if len(selection.point_ids) != 1 or len(selection.entity_ids) != 1:
            return False
        if entities is None:
            return False
        if not _applies_to_entities(constraint_type, entities):
            return False
        entity = entities[0]
        if constraint_type is PointOnLineConstraint:
            excluded = registry.junction_point_ids(entity.id)
        else:
            excluded = registry.entity_endpoint_ids(entity.id) or []
        return selection.point_ids[0] not in excluded
    if constraint_type is SymmetryConstraint:
        if len(selection.point_ids) == 3 and not selection.entity_ids:
            return True
        if len(selection.point_ids) == 2 and len(selection.entity_ids) == 1:
            return entities is not None and _applies_to_entities(
                constraint_type, entities
            )
        return False
    if constraint_type in (
        RadiusConstraint,
        DiameterConstraint,
        PerpendicularConstraint,
        TangentConstraint,
        EqualLengthConstraint,
        EqualDistanceConstraint,
        AspectRatioConstraint,
    ):
        if selection.point_ids:
            return False
        if entities is None or not entities:
            return False
        return _applies_to_entities(constraint_type, entities)
    return False


def _applies_to_entities(constraint_type: type, entities: list) -> bool:
    if constraint_type is DistanceConstraint:
        return len(entities) == 1 and _is_line(entities[0])
    if constraint_type in (HorizontalConstraint, VerticalConstraint):
        return bool(entities) and all(_is_line(e) for e in entities)
    if constraint_type is AngleConstraint:
        return len(entities) == 2 and all(_is_line(e) for e in entities)
    if constraint_type is RadiusConstraint:
        return len(entities) == 1 and _is_radius_entity(entities[0])
    if constraint_type is DiameterConstraint:
        return len(entities) == 1 and (entities[0].type_name() == "circle")
    if constraint_type is AspectRatioConstraint:
        return len(entities) == 2 and all(_is_line(e) for e in entities)
    if constraint_type is PointOnLineConstraint:
        return len(entities) == 1 and (
            _is_line(entities[0]) or _is_radius_entity(entities[0])
        )
    if constraint_type is PointOnCurveConstraint:
        return len(entities) == 1 and (entities[0].type_name() == "bezier")
    if constraint_type is TangentConstraint:
        if len(entities) != 2:
            return False
        e1, e2 = entities
        if _is_line(e1):
            return _is_radius_entity(e2)
        return _is_line(e2) and _is_radius_entity(e1)
    if constraint_type is PerpendicularConstraint:
        return len(entities) == 2 and all(
            _is_line(e) or _is_radius_entity(e) for e in entities
        )
    if constraint_type is EqualLengthConstraint:
        return len(entities) >= 2 and all(
            _has_characteristic_length(e) for e in entities
        )

    if constraint_type is EqualDistanceConstraint:
        return len(entities) == 2 and all(_is_line(e) for e in entities)
    if constraint_type is SymmetryConstraint:
        return len(entities) == 1 and _is_line(entities[0])
    return False


def referenced_point_ids(constraint: Constraint) -> set[EntityID]:
    """The point ids the constraint references."""
    return set(constraint.referenced_points())


def referenced_entity_ids(
    registry: EntityRegistry, constraint: Constraint
) -> set[EntityID]:
    """The entity ids the constraint restricts, recovered through the
    registry by matching the constraint's resolved geometry."""
    eids: set[EntityID] = set()
    supports_targets = hasattr(constraint, "targets_segment")
    supports_radius = hasattr(constraint, "constrains_radius")
    for eid in registry.entity_ids():
        endpoints = registry.entity_endpoint_ids(eid)
        if (
            supports_targets
            and endpoints
            and constraint.targets_segment(
                endpoints[0], endpoints[1], eid, registry
            )
        ) or (
            supports_radius
            and registry.is_radius_entity(eid)
            and constraint.constrains_radius(registry, eid)
        ):
            eids.add(eid)
    if isinstance(constraint, PointOnLineConstraint):
        eids.update(_target_entity_ids(registry, constraint.target))
    elif isinstance(constraint, PointOnCurveConstraint):
        eids.update(
            _entities_with_points(registry, (constraint.start, constraint.end))
        )
    elif isinstance(constraint, SymmetryConstraint):
        eids.update(_target_entity_ids(registry, constraint.target))
    elif isinstance(constraint, AngleConstraint):
        eids.update(
            _entities_with_points(registry, (constraint.p1, constraint.p2))
        )
        eids.update(
            _entities_with_points(registry, (constraint.p3, constraint.p4))
        )
    elif isinstance(constraint, PerpendicularConstraint):
        eids.update(_perpendicular_entity_ids(registry, constraint.target))
    return eids


def _radius_entity_of(
    registry: EntityRegistry, constraint: Constraint
) -> Entity | None:
    for eid in registry.entity_ids():
        if registry.is_radius_entity(eid) and constraint.constrains_radius(
            registry, eid
        ):
            return registry.get_entity_object(eid)
    return None


def _target_entity_ids(
    registry: EntityRegistry, target: SymmetryTarget | PerpendicularTarget
) -> set[EntityID]:
    if hasattr(target, "l1"):
        return _entities_with_points(registry, (target.l1, target.l2))
    if hasattr(target, "radius_point"):
        return _entities_with_points(
            registry, (target.center, target.radius_point)
        )
    return set()


def _perpendicular_entity_ids(
    registry: EntityRegistry, target: SymmetryTarget | PerpendicularTarget
) -> set[EntityID]:
    if hasattr(target, "a1"):
        return _entities_with_points(
            registry, (target.a1, target.a2, target.b1, target.b2)
        )
    if hasattr(target, "l1"):
        return _entities_with_points(
            registry, (target.l1, target.l2, target.center)
        )
    return _entities_with_points(
        registry,
        (target.c1, target.r1_point, target.c2, target.r2_point),
    )


def _entities_with_points(
    registry: EntityRegistry, pids: tuple[EntityID, ...]
) -> set[EntityID]:
    """Entities whose defining points contain all of the given ids."""
    wanted = {pid for pid in pids if pid is not None}
    if not wanted:
        return set()
    matches: set[EntityID] = set()
    for eid in registry.entity_ids():
        entity_pids = set(registry.entity_point_ids(eid) or [])
        if wanted <= entity_pids:
            matches.add(eid)
    return matches


def depends_on_points(
    constraint: Constraint, point_ids: set[EntityID]
) -> bool:
    """Whether the constraint references any of the given points."""
    return not referenced_point_ids(constraint).isdisjoint(point_ids)


def depends_on_entities(
    registry: EntityRegistry, constraint: Constraint, entity_ids: set[EntityID]
) -> bool:
    """Whether the constraint restricts any of the given entities."""
    return not referenced_entity_ids(registry, constraint).isdisjoint(
        entity_ids
    )


def update_from_context(
    constraint: Constraint, context: dict[str, Any]
) -> None:
    """Re-evaluates the constraint's expression and updates its value
    and status."""
    expression = get_expression(constraint)
    if not expression:
        set_status(constraint, ConstraintStatus.VALID)
        return
    try:
        value = safe_evaluate(expression, context)
        constraint.set_value(value)
        set_status(constraint, ConstraintStatus.EXPRESSION_BASED)
    except (ValueError, SyntaxError, NameError, TypeError):
        set_status(constraint, ConstraintStatus.ERROR)

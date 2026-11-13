"""Migration from the legacy sketcher document shape to raygeo's.

The sketch file format is unchanged: entity and point dictionaries
were already written in raygeo's reference shape, and the constraint
dictionaries use the same type keys raygeo's ``add_constraint_json``
parses. This module only reconciles the few legacy constraint
dictionary quirks and merges the constraint list into the registry
dictionary so ``EntityRegistry.from_dict`` can load the whole
document model in one pass.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

_KNOWN_CONSTRAINT_TYPES = {
    "HorizontalConstraint",
    "VerticalConstraint",
    "CoincidentConstraint",
    "CollinearConstraint",
    "DistanceConstraint",
    "RadiusConstraint",
    "DiameterConstraint",
    "PerpendicularConstraint",
    "TangentConstraint",
    "EqualLengthConstraint",
    "EqualDistanceConstraint",
    "PointOnLineConstraint",
    "PointOnCurveConstraint",
    "SymmetryConstraint",
    "AspectRatioConstraint",
    "AngleConstraint",
    "RotationalConstraint",
    "ParallelogramConstraint",
}

logger = logging.getLogger(__name__)


def migrate_sketch_data(data: dict[str, Any]) -> dict[str, Any]:
    """Returns the sketch dictionary with its constraints merged into
    the registry dictionary, ready for ``EntityRegistry.from_dict``.

    The input is not modified. Unknown constraint types (including
    the empty dictionaries legacy files wrote for drag constraints)
    are dropped with a warning; the symmetry constraint's explicit
    ``None`` ``center``/``axis`` keys are removed because raygeo
    cannot resolve them.
    """
    migrated = copy.deepcopy(data)
    registry_data = dict(migrated.get("registry", {}))
    constraints: list[dict[str, Any]] = []
    for constraint_dict in migrated.get("constraints", []):
        migrated_constraint = _migrate_constraint(constraint_dict)
        if migrated_constraint is not None:
            constraints.append(migrated_constraint)
    registry_data["constraints"] = constraints
    migrated["registry"] = registry_data
    return migrated


def _migrate_constraint(
    constraint_dict: dict[str, Any],
) -> dict[str, Any] | None:
    """Sanitizes one legacy constraint dictionary for raygeo."""
    if not isinstance(constraint_dict, dict):
        return None
    type_name = constraint_dict.get("type")
    if type_name not in _KNOWN_CONSTRAINT_TYPES:
        logger.debug(
            "Skipping constraint of unknown type during migration: %r",
            type_name,
        )
        return None
    migrated = {
        key: value
        for key, value in constraint_dict.items()
        if value is not None or key not in ("center", "axis")
    }
    return migrated

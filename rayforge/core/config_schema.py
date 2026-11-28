"""Versioned schema migrations for the application configuration.

Configuration files did not carry a schema version historically
(``schema_version`` absent means version 0). :func:`migrate` chains
pure dict→dict transforms up to
:data:`CURRENT_CONFIG_SCHEMA_VERSION` and stamps the result, so
``Config.from_dict`` always reads current-format data. Every future
config change that renames a persisted value adds one migration
function here instead of a legacy branch in the config loader.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

CURRENT_CONFIG_SCHEMA_VERSION = 1


def _migrate_v0_to_v1(data: dict[str, Any]) -> dict[str, Any]:
    """Renames the ops color mode written before workflows carried
    their own colors.

    The mode used to be stored as ``"layer"``; the color now lives on
    the workflow, so the value is ``"workflow"``.
    """
    result = dict(data)
    if result.get("ops_color_mode") == "layer":
        result["ops_color_mode"] = "workflow"
    return result


_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    1: _migrate_v0_to_v1,
}


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Chains all migrations up to the current version and stamps it.

    Passes the dict through unchanged when it is already current. The
    input dict is not mutated.
    """
    version = data.get("schema_version", 0)
    if version > CURRENT_CONFIG_SCHEMA_VERSION:
        logger.warning(
            "Config schema version %s is newer than this app "
            "understands (%s); loading as-is",
            version,
            CURRENT_CONFIG_SCHEMA_VERSION,
        )
        return data
    result = data
    while version < CURRENT_CONFIG_SCHEMA_VERSION:
        version += 1
        migration = _MIGRATIONS.get(version)
        if migration is None:
            raise ValueError(f"No migration to config schema {version}")
        result = migration(result)
    result["schema_version"] = CURRENT_CONFIG_SCHEMA_VERSION
    return result

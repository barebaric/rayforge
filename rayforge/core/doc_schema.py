"""Versioned schema migrations for .ryp documents.

Documents did not carry a schema version historically (schema_version
absent means version 0, the layer-based format). :func:`migrate`
chains pure dict→dict transforms up to
:data:`CURRENT_DOC_SCHEMA_VERSION` and stamps the result, so
``Doc.from_dict`` always parses current-format data. Every future
document-model change adds one migration function here and bumps the
version; legacy branches never accumulate in model code.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

CURRENT_DOC_SCHEMA_VERSION = 1


def _migrate_v0_to_v1(data: dict[str, Any]) -> dict[str, Any]:
    """Layer-based (v0) → setup-based (v1).

    Each layer becomes a Setup: its job settings carry over, its
    visibility and color are dropped (workpieces carry visibility;
    the workflow inherits the color). Layer-level post processors
    move onto the workflow, whose uid replaces the layer's uid as the
    span marker key downstream. Document-level stock items move into
    the first setup.
    """
    children = data.get("children", [])
    new_children: list[dict[str, Any]] = []
    stock_dicts: list[dict[str, Any]] = []

    for child in children:
        if child.get("type") == "layer":
            new_children.append(_layer_to_setup(child))
        elif child.get("type") == "stockitem":
            stock_dicts.append(child)
        else:
            new_children.append(child)

    if stock_dicts and new_children:
        first_setup = next(
            (c for c in new_children if c.get("type") == "setup"), None
        )
        if first_setup is not None:
            first_setup["children"] = stock_dicts + first_setup.get(
                "children", []
            )
        else:
            new_children = stock_dicts + new_children

    result = dict(data)
    result["children"] = new_children
    result["active_setup_index"] = data.get("active_layer_index", 0)
    result.pop("active_layer_index", None)
    return result


def _layer_to_setup(layer: dict[str, Any]) -> dict[str, Any]:
    """Transforms one layer dict into a setup dict, uid-preserving."""
    setup = {
        k: v
        for k, v in layer.items()
        if k
        not in (
            "type",
            "visible",
            "color",
            "post_processors_dicts",
            "children",
        )
    }
    setup["type"] = "setup"

    color = layer.get("color")
    post_dicts = layer.get("post_processors_dicts")

    workflows = []
    migrated_workflows: list[dict[str, Any]] = []
    content = []
    for child in layer.get("children", []):
        if child.get("type") == "workflow":
            workflows.append(child)
        else:
            content.append(dict(child))

    for workflow in workflows:
        workflow = dict(workflow)
        if color is not None and "color" not in workflow:
            workflow["color"] = color
        if post_dicts and not workflow.get("post_processors_dicts"):
            workflow["post_processors_dicts"] = list(post_dicts)
        migrated_workflows.append(workflow)

    setup["children"] = content + migrated_workflows
    return setup


_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    1: _migrate_v0_to_v1,
}


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Chains all migrations up to the current version and stamps it.

    Passes the dict through unchanged when it is already current. The
    input dict is not mutated.
    """
    version = data.get("schema_version", 0)
    if version > CURRENT_DOC_SCHEMA_VERSION:
        logger.warning(
            "Document schema version %s is newer than this app "
            "understands (%s); loading as-is",
            version,
            CURRENT_DOC_SCHEMA_VERSION,
        )
        return data
    result = data
    while version < CURRENT_DOC_SCHEMA_VERSION:
        version += 1
        migration = _MIGRATIONS.get(version)
        if migration is None:
            raise ValueError(f"No migration to schema version {version}")
        result = migration(result)
    result["schema_version"] = CURRENT_DOC_SCHEMA_VERSION
    return result

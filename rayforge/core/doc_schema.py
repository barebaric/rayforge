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
    """Restructures v0 layers to the v1 shape.

    The document still holds layers, but a layer no longer carries a
    color: the workflow inherits it. Layer-level post processors move
    onto the workflow. Document-level stock items move into the first
    layer, so a layer owns the stock its parts are cut from.
    """
    children = data.get("children", [])
    new_children: list[dict[str, Any]] = []
    stock_dicts: list[dict[str, Any]] = []

    for child in children:
        if child.get("type") == "layer":
            new_children.append(_layer_to_v1(child))
        elif child.get("type") == "stockitem":
            stock_dicts.append(child)
        else:
            new_children.append(child)

    if stock_dicts and new_children:
        first_layer = next(
            (c for c in new_children if c.get("type") == "layer"), None
        )
        if first_layer is not None:
            first_layer["children"] = stock_dicts + first_layer.get(
                "children", []
            )
        else:
            new_children = stock_dicts + new_children

    result = dict(data)
    result["children"] = _migrate_import_modes(new_children)
    return result


def _migrate_import_modes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Renames the legacy new-layers import mode in segment specs."""
    for item in items:
        segment = item.get("source_segment")
        if isinstance(segment, dict):
            spec = segment.get("vectorization_spec")
            if (
                isinstance(spec, dict)
                and spec.get("layer_import_mode") == "new_layers"
            ):
                spec = dict(spec)
                spec["layer_import_mode"] = "new_workflows"
                segment = dict(segment)
                segment["vectorization_spec"] = spec
                item["source_segment"] = segment
        children = item.get("children")
        if isinstance(children, list):
            _migrate_import_modes(children)
    return items


def _layer_to_v1(source: dict[str, Any]) -> dict[str, Any]:
    """Transforms one v0 layer dict to the v1 shape, uid-preserving."""
    migrated = {
        k: v
        for k, v in source.items()
        if k
        not in (
            "type",
            "color",
            "post_processors_dicts",
            "children",
        )
    }
    migrated["type"] = "layer"

    color = source.get("color")
    post_dicts = source.get("post_processors_dicts")

    workflows: list[dict[str, Any]] = []
    migrated_workflows: list[dict[str, Any]] = []
    content = []
    for child in source.get("children", []):
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

    migrated["children"] = content + migrated_workflows
    return migrated


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

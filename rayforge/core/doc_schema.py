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

import copy
import logging
import uuid
from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from raygeo.geo import Matrix

logger = logging.getLogger(__name__)

CURRENT_DOC_SCHEMA_VERSION = 1

_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

# Fields of a schema-0 layer that describe the layer itself. Everything
# else on the layer dict is carried over to the workflow it becomes.
_LAYER_KEYS = {
    "uid",
    "type",
    "name",
    "matrix",
    "visible",
    "color",
    "post_processors_dicts",
    "wcs",
    "rotary_enabled",
    "rotary_diameter",
    "rotary_module_uid",
    "stock_material_uid",
    "children",
}

# Settings that describe the physical setup. Schema-0 layers sharing
# these values can live in one schema-1 layer.
_SETUP_KEYS = (
    "wcs",
    "rotary_enabled",
    "rotary_diameter",
    "rotary_module_uid",
    "stock_material_uid",
)


def _migrate_v0_to_v1(data: dict[str, Any]) -> dict[str, Any]:
    """Turns schema-0 layers into schema-1 workflows.

    A schema-0 layer was one operation pass over the document, which is
    exactly what a schema-1 workflow is. The layers' workpieces move
    into the new layer that owns the shared physical setup, and each
    old layer becomes a workflow restricted to its own workpieces. The
    layer's color and post processors move to that workflow.

    Layers with different setup settings (WCS, rotary, stock material)
    cannot share one schema-1 layer, so they are grouped by those
    settings; each group becomes one layer. Document-level stock moves
    into the first layer.
    """
    children = copy.deepcopy(data.get("children", []))
    old_layers = [c for c in children if c.get("type") == "layer"]
    stocks = [c for c in children if c.get("type") == "stockitem"]
    others = [
        c for c in children if c.get("type") not in ("layer", "stockitem")
    ]

    groups = _group_layers(old_layers)
    new_layers: list[dict[str, Any]] = []
    old_index_to_new: dict[int, int] = {}
    for new_index, group in enumerate(groups):
        new_layers.append(_group_to_v1(group, new_index))
        for old_index, _layer in group:
            old_index_to_new[old_index] = new_index

    if stocks:
        if new_layers:
            new_layers[0]["children"] = stocks + new_layers[0]["children"]
        else:
            new_layers.append(
                {
                    "uid": str(uuid.uuid4()),
                    "type": "layer",
                    "name": _("Layer 1"),
                    "matrix": copy.deepcopy(_IDENTITY),
                    "visible": True,
                    "children": stocks,
                }
            )

    result = dict(data)
    result["children"] = _migrate_import_modes(new_layers + others)

    if old_layers:
        old_active = data.get("active_layer_index", 0)
        if not isinstance(old_active, int) or not 0 <= old_active < len(
            old_layers
        ):
            old_active = 0
        result["active_layer_index"] = old_index_to_new.get(old_active, 0)
    return result


def _setup_key(layer: dict[str, Any]) -> tuple[Any, ...]:
    """The normalized physical setup of a schema-0 layer."""
    return (
        layer.get("wcs"),
        bool(layer.get("rotary_enabled", False)),
        float(layer.get("rotary_diameter", 25.0)),
        layer.get("rotary_module_uid"),
        layer.get("stock_material_uid"),
    )


def _group_layers(
    layers: list[dict[str, Any]],
) -> list[list[tuple[int, dict[str, Any]]]]:
    """Groups old layers by their physical setup, preserving order."""
    groups: list[list[tuple[int, dict[str, Any]]]] = []
    keys: list[tuple[Any, ...]] = []
    for index, layer in enumerate(layers):
        key = _setup_key(layer)
        try:
            position = keys.index(key)
        except ValueError:
            keys.append(key)
            groups.append([])
            position = len(groups) - 1
        groups[position].append((index, layer))
    return groups


def _group_to_v1(
    group: list[tuple[int, dict[str, Any]]], index: int
) -> dict[str, Any]:
    """Builds one schema-1 layer from a group of same-setup old layers."""
    first = group[0][1]
    new_layer: dict[str, Any] = {
        "uid": str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"rayforge-layer:{first.get('uid', index)}",
            )
        ),
        "type": "layer",
        "name": _("Layer {number}").format(number=index + 1),
        "matrix": copy.deepcopy(_IDENTITY),
        "visible": any(
            layer.get("visible", True) for _old_index, layer in group
        ),
        "children": [],
    }
    for key in _SETUP_KEYS:
        if key in first:
            new_layer[key] = copy.deepcopy(first[key])

    children: list[dict[str, Any]] = []
    for _old_index, old_layer in group:
        children.extend(_layer_children_to_v1(old_layer))
    new_layer["children"] = children
    return new_layer


def _layer_children_to_v1(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Converts one old layer's children into a workflow and content."""
    layer_matrix = source.get("matrix")
    content: list[dict[str, Any]] = []
    workflows: list[dict[str, Any]] = []
    for child in source.get("children", []):
        if child.get("type") == "workflow":
            workflows.append(child)
        else:
            content.append(child)

    workpiece_uids = _collect_workpiece_uids(content)
    if not workflows:
        workflows = [
            {
                "uid": source.get("uid"),
                "type": "workflow",
                "name": source.get("name", ""),
                "matrix": copy.deepcopy(_IDENTITY),
                "children": [],
            }
        ]

    result: list[dict[str, Any]] = []
    for index, workflow in enumerate(workflows):
        result.append(
            _workflow_to_v1(
                source,
                workflow,
                index,
                workpiece_uids,
                layer_matrix,
            )
        )
    for item in content:
        item = copy.deepcopy(item)
        item["matrix"] = _bake_matrix(layer_matrix, item.get("matrix"))
        result.append(item)
    return result


def _workflow_to_v1(
    layer: dict[str, Any],
    workflow: dict[str, Any],
    index: int,
    workpiece_uids: list[str],
    layer_matrix: list[list[float]] | None,
) -> dict[str, Any]:
    """Builds the workflow for one old layer."""
    migrated = copy.deepcopy(workflow)
    migrated["type"] = "workflow"
    migrated["matrix"] = _bake_matrix(layer_matrix, workflow.get("matrix"))
    migrated["workpiece_uids"] = list(workpiece_uids)
    if index == 0:
        # The old layer's uid identified the cut span in jobs, so the
        # workflow inherits it. Its name is the user-visible operation.
        migrated["uid"] = layer.get("uid", migrated.get("uid"))
        migrated["name"] = (
            layer.get("name") or workflow.get("name") or _("Workflow")
        )
        for key, value in layer.items():
            if key not in _LAYER_KEYS and key not in migrated:
                migrated[key] = copy.deepcopy(value)

    color = layer.get("color")
    if color is not None:
        migrated["color"] = color
    post_processors = layer.get("post_processors_dicts")
    if post_processors:
        migrated["post_processors_dicts"] = copy.deepcopy(post_processors)
    return migrated


def _collect_workpiece_uids(items: list[dict[str, Any]]) -> list[str]:
    """Collects the uids of all workpieces in a content subtree."""
    uids: list[str] = []
    for item in items:
        uid = item.get("uid")
        if item.get("type") == "workpiece" and uid:
            uids.append(uid)
        uids.extend(_collect_workpiece_uids(item.get("children", [])))
    return uids


def _bake_matrix(
    parent: list[list[float]] | None, child: list[list[float]] | None
) -> list[list[float]]:
    """Composes an old layer transform into a child's transform."""
    if parent is None:
        return (
            copy.deepcopy(child)
            if child is not None
            else copy.deepcopy(_IDENTITY)
        )
    parent_matrix = Matrix.from_list(parent)
    if parent_matrix == Matrix.identity():
        return (
            copy.deepcopy(child)
            if child is not None
            else copy.deepcopy(_IDENTITY)
        )
    child_matrix = (
        Matrix.from_list(child) if child is not None else Matrix.identity()
    )
    return (parent_matrix @ child_matrix).to_list()


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

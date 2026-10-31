"""
Serialize the document tree for the web UI.

Positions and sizes are decomposed via the DocItem ``pos`` / ``size`` /
``angle`` properties (WORLD space, mm); raw matrices are never sent.
"""

from typing import TYPE_CHECKING, Any

from rayforge.core.group import Group
from rayforge.core.item import DocItem
from rayforge.core.layer import Layer
from rayforge.core.stock import StockItem
from rayforge.core.workpiece import WorkPiece

if TYPE_CHECKING:
    from rayforge.core.doc import Doc
    from rayforge.core.step import Step
    from rayforge.doceditor.editor import DocEditor


def _round(value: float) -> float:
    return round(float(value), 4)


def _point(value) -> list[float]:
    return [_round(value[0]), _round(value[1])]


def _json_safe(value: Any) -> Any:
    """Convert a parameter value to a JSON-safe primitive."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


def find_item_by_uid(doc: "Doc", uid: str) -> DocItem | None:
    """Find any document item (layer, group, workpiece, stock) by uid."""
    for child in doc.children:
        if child.uid == uid:
            return child
    for layer in doc.layers:
        if layer.uid == uid:
            return layer
        workflow = layer.workflow
        if workflow is not None:
            if workflow.uid == uid:
                return workflow
            for step in workflow.steps:
                if step.uid == uid:
                    return step
        for child in layer.children:
            if child.uid == uid:
                return child
            if isinstance(child, Group):
                descendant = child.find_descendant_by_uid(uid)
                if descendant is not None:
                    return descendant
    return None


def serialize_layer(layer: Layer, order: int) -> dict[str, Any]:
    workflow = layer.workflow
    return {
        "uid": layer.uid,
        "type": "layer",
        "name": layer.name,
        "visible": layer.visible,
        "color": layer.color,
        "wcs": layer.wcs,
        "rotary_enabled": layer.rotary_enabled,
        "rotary_diameter_mm": layer.rotary_diameter,
        "rotary_module_uid": layer.rotary_module_uid,
        "order": order,
        "workflow_uid": workflow.uid if workflow else None,
        "step_uids": [step.uid for step in workflow.steps] if workflow else [],
        "content_uids": [
            child.uid
            for child in layer.children
            if isinstance(child, (WorkPiece, Group))
        ],
    }


def serialize_workpiece(
    workpiece: WorkPiece, layer: Layer | None = None
) -> dict[str, Any]:
    owner = layer or workpiece.layer
    return {
        "uid": workpiece.uid,
        "type": "workpiece",
        "name": workpiece.name,
        "layer_uid": owner.uid if owner else None,
        "pos_mm": _point(workpiece.pos),
        "size_mm": _point(workpiece.size),
        "angle_deg": _round(workpiece.angle),
        "geometry_revision": workpiece.geometry_revision,
        "transform_revision": workpiece.transform_revision,
    }


def serialize_group(group: Group) -> dict[str, Any]:
    return {
        "uid": group.uid,
        "type": "group",
        "name": group.name,
        "pos_mm": _point(group.pos),
        "size_mm": _point(group.size),
        "angle_deg": _round(group.angle),
        "children_uids": [child.uid for child in group.children],
    }


def serialize_stock_item(stock: StockItem) -> dict[str, Any]:
    return {
        "uid": stock.uid,
        "type": "stock",
        "name": stock.name,
        "visible": stock.visible,
        "color": stock.get_effective_color(),
        "pos_mm": _point(stock.pos),
        "size_mm": _point(stock.size),
        "angle_deg": _round(stock.angle),
        "geometry_revision": stock.geometry_revision,
        "transform_revision": stock.transform_revision,
    }


def serialize_step(step: "Step", order: int) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in step.recipe_keys():
        value = getattr(step, key, None)
        params[key] = _json_safe(step.recipe_value(key, value))
    return {
        "uid": step.uid,
        "type": "step",
        "step_type": type(step).__name__,
        "typelabel": step.TYPELABEL,
        "name": step.name,
        "params": params,
        "workpiece_uids": list(step.workpiece_uids),
        "order": order,
        "visible": step.visible,
        "selected_head_uid": step.selected_head_uid,
    }


def doc_tree(editor: "DocEditor") -> dict[str, Any]:
    """Full document tree as a JSON-safe dict."""
    doc = editor.doc
    workpieces: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for order, layer in enumerate(doc.layers):
        layers.append(serialize_layer(layer, order))
        workflow = layer.workflow
        if workflow is not None:
            for step_order, step in enumerate(workflow.steps):
                steps.append(
                    dict(serialize_step(step, step_order), layer_uid=layer.uid)
                )
        for child in layer.children:
            if isinstance(child, WorkPiece):
                workpieces.append(serialize_workpiece(child, layer))
            elif isinstance(child, Group):
                groups.append(serialize_group(child))
                for descendant in child.get_descendants(WorkPiece):
                    workpieces.append(serialize_workpiece(descendant, layer))
    stock_items = [serialize_stock_item(stock) for stock in doc.stock_items]
    return {
        "active_layer_uid": doc.active_layer.uid,
        "layers": layers,
        "groups": groups,
        "workpieces": workpieces,
        "stock_items": stock_items,
        "steps": steps,
    }


def item_revisions(doc: "Doc") -> dict[str, int]:
    """Combined revision counters of all workpieces, keyed by uid."""
    return {
        workpiece.uid: int(workpiece.geometry_revision)
        + int(workpiece.transform_revision)
        for workpiece in doc.all_workpieces
    }


def tree_digest(editor: "DocEditor") -> dict[str, Any]:
    """Compact tree summary used as the WebSocket connect snapshot."""
    doc = editor.doc
    return {
        "active_layer_uid": doc.active_layer.uid,
        "layer_count": len(doc.layers),
        "workpiece_count": len(doc.all_workpieces),
        "workpiece_uids": [wp.uid for wp in doc.all_workpieces],
        "revisions": item_revisions(doc),
    }

"""Layer endpoints: add, update, delete, activate, reorder."""

from gettext import gettext as _

from fastapi import APIRouter, Request
from pydantic import BaseModel

from rayforge.core.layer import Layer
from rayforge.core.undo.property_cmd import ChangePropertyCommand

from ..serializers import find_item_by_uid
from ..serializers.doc_tree import serialize_layer
from .deps import bad_request, get_bridge, item_not_found

router = APIRouter(tags=["layers"])


class LayerUpdateRequest(BaseModel):
    name: str | None = None
    visible: bool | None = None
    color: str | None = None
    wcs: str | None = None
    rotary_enabled: bool | None = None


class LayerReorderRequest(BaseModel):
    uids: list[str]


def _get_layer(editor, uid: str) -> Layer:
    item = find_item_by_uid(editor.doc, uid)
    if not isinstance(item, Layer):
        raise item_not_found()
    return item


def _add_layer(editor) -> dict:
    editor.layer.add_layer_and_set_active()
    layer = editor.doc.active_layer
    return serialize_layer(layer, editor.doc.layers.index(layer))


@router.post("/layers")
async def add_layer(request: Request) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_add_layer, bridge.editor)


def _update_layer(editor, uid: str, payload: LayerUpdateRequest) -> dict:
    layer = _get_layer(editor, uid)
    history = editor.history_manager
    with history.transaction(_("Update layer")) as t:
        if payload.name is not None and payload.name != layer.name:
            editor.layer.rename_layer(layer, payload.name)
        if payload.visible is not None and payload.visible != layer.visible:
            editor.layer.set_layer_visibility(layer, payload.visible)
        if payload.color is not None and payload.color != layer.color:
            t.execute(
                ChangePropertyCommand(
                    target=layer,
                    property_name="color",
                    new_value=payload.color,
                    setter_method_name="set_color",
                    name=_("Change layer color"),
                )
            )
        if payload.wcs is not None and payload.wcs != layer.wcs:
            t.execute(
                ChangePropertyCommand(
                    target=layer,
                    property_name="wcs",
                    new_value=payload.wcs,
                    setter_method_name="set_wcs",
                    name=_("Set layer WCS"),
                )
            )
        if (
            payload.rotary_enabled is not None
            and payload.rotary_enabled != layer.rotary_enabled
        ):
            t.execute(
                ChangePropertyCommand(
                    target=layer,
                    property_name="rotary_enabled",
                    new_value=payload.rotary_enabled,
                    setter_method_name="set_rotary_enabled",
                    name=_("Toggle layer rotary"),
                )
            )
    return serialize_layer(layer, editor.doc.layers.index(layer))


@router.put("/layers/{uid}")
async def update_layer(
    request: Request, uid: str, payload: LayerUpdateRequest
) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_update_layer, bridge.editor, uid, payload)


def _delete_layer(editor, uid: str) -> dict:
    layer = _get_layer(editor, uid)
    if len(editor.doc.layers) <= 1:
        raise bad_request(_("Cannot delete the last layer"))
    editor.layer.delete_layer(layer)
    return {"removed": uid}


@router.delete("/layers/{uid}")
async def delete_layer(request: Request, uid: str) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_delete_layer, bridge.editor, uid)


def _activate_layer(editor, uid: str) -> dict:
    layer = _get_layer(editor, uid)
    editor.layer.set_active_layer(layer)
    return {"active_layer_uid": editor.doc.active_layer.uid}


@router.post("/layers/{uid}/activate")
async def activate_layer(request: Request, uid: str) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_activate_layer, bridge.editor, uid)


def _reorder_layers(editor, payload: LayerReorderRequest) -> dict:
    by_uid = {layer.uid: layer for layer in editor.doc.layers}
    ordered: list[Layer] = []
    for uid in payload.uids:
        layer = by_uid.get(uid)
        if not isinstance(layer, Layer):
            raise item_not_found()
        ordered.append(layer)
    for layer in editor.doc.layers:
        if layer not in ordered:
            ordered.append(layer)
    editor.layer.reorder_layers(ordered)
    return {
        "layer_uids": [layer.uid for layer in editor.doc.layers],
    }


@router.post("/layers/reorder")
async def reorder_layers(
    request: Request, payload: LayerReorderRequest
) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_reorder_layers, bridge.editor, payload)


@router.get("/layers")
async def get_layers(request: Request) -> dict:
    bridge = get_bridge(request)

    def _layers(editor) -> dict:
        return {
            "active_layer_uid": editor.doc.active_layer.uid,
            "layers": [
                serialize_layer(layer, order)
                for order, layer in enumerate(editor.doc.layers)
            ],
        }

    return await bridge.run(_layers, bridge.editor)


@router.get("/layers/{uid}")
async def get_layer(request: Request, uid: str) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_serialize_layer_by_uid, bridge.editor, uid)


def _serialize_layer_by_uid(editor, uid: str) -> dict:
    layer = _get_layer(editor, uid)
    return serialize_layer(layer, editor.doc.layers.index(layer))

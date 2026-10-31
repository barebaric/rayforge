"""Item endpoints: geometry, texture, transforms, edit operations."""

import io
from gettext import gettext as _

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from rayforge.core.group import Group
from rayforge.core.stock import StockItem
from rayforge.core.undo.property_cmd import ChangePropertyCommand
from rayforge.core.workpiece import WorkPiece

from ..serializers import find_item_by_uid, stock_geometry, workpiece_geometry
from .deps import (
    bad_request,
    get_bridge,
    item_not_found,
    machine_unavailable,
)

router = APIRouter(prefix="/items", tags=["items"])

TEXTURE_MAX_PX = 512


class TransformRequest(BaseModel):
    pos_mm: list[float] | None = None
    size_mm: list[float] | None = None
    angle_deg: float | None = None


class UidsRequest(BaseModel):
    uids: list[str] = Field(min_length=1)


class FlipRequest(UidsRequest):
    axis: str


class AlignRequest(UidsRequest):
    mode: str


class DistributeRequest(UidsRequest):
    axis: str


def _get_item(editor, uid: str):
    item = find_item_by_uid(editor.doc, uid)
    if item is None:
        raise item_not_found()
    return item


def _get_content_items(editor, uids: list[str]) -> list:
    items = []
    for uid in uids:
        item = find_item_by_uid(editor.doc, uid)
        if isinstance(item, (WorkPiece, Group)):
            items.append(item)
    if not items:
        raise item_not_found()
    return items


def _texture_unavailable() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=_("No raster rendering available for this workpiece"),
    )


def _apply_item_transform(editor, uid, payload: TransformRequest) -> dict:
    item = _get_item(editor, uid)
    if not isinstance(item, (WorkPiece, Group)):
        raise bad_request(_("Only workpieces and groups can be transformed"))
    if payload.pos_mm is not None and len(payload.pos_mm) != 2:
        raise bad_request(_("pos_mm must contain two values"))
    if payload.size_mm is not None and len(payload.size_mm) != 2:
        raise bad_request(_("size_mm must contain two values"))

    history = editor.history_manager
    with history.transaction(_("Transform item")) as t:
        old_matrix = item.matrix.copy()
        if payload.pos_mm is not None:
            item.pos = (payload.pos_mm[0], payload.pos_mm[1])
        if payload.size_mm is not None:
            item.set_size(payload.size_mm[0], payload.size_mm[1])
        if payload.angle_deg is not None:
            item.angle = payload.angle_deg
        new_matrix = item.matrix.copy()
        if old_matrix.is_close(new_matrix):
            return {"uid": uid}
        t.execute(
            ChangePropertyCommand(
                target=item,
                property_name="matrix",
                new_value=new_matrix,
                old_value=old_matrix,
                name=_("Transform item"),
            )
        )
    return {"uid": uid}


@router.get("/{uid}/geometry")
async def get_item_geometry(
    request: Request, uid: str, tolerance: float = 0.0
) -> dict:
    bridge = get_bridge(request)

    def _serialize(editor) -> dict:
        item = _get_item(editor, uid)
        if isinstance(item, StockItem):
            return stock_geometry(item, max(tolerance, 0.0))
        if isinstance(item, WorkPiece):
            return workpiece_geometry(item, max(tolerance, 0.0))
        raise bad_request(_("Only workpieces and stock have vector geometry"))

    return await bridge.run(_serialize, bridge.editor)


@router.get("/{uid}/texture.png")
async def get_item_texture(request: Request, uid: str, rev: int = 0):
    bridge = get_bridge(request)

    def _render(editor):
        item = _get_item(editor, uid)
        if not isinstance(item, WorkPiece):
            return None
        width_mm, height_mm = item.size
        if width_mm <= 0 or height_mm <= 0:
            return None
        px_per_mm = TEXTURE_MAX_PX / max(width_mm, height_mm)
        width = max(1, int(width_mm * px_per_mm))
        height = max(1, int(height_mm * px_per_mm))
        surface = item.render_to_pixels(width, height)
        if surface is None:
            return None
        buffer = io.BytesIO()
        surface.write_to_png(buffer)
        revision = int(item.geometry_revision) + int(item.transform_revision)
        return buffer.getvalue(), revision

    rendered = await bridge.run(_render, bridge.editor)
    if rendered is None:
        raise _texture_unavailable()
    png_bytes, revision = rendered
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"ETag": f'"rev-{revision}"'},
    )


@router.put("/{uid}/transform")
async def put_item_transform(
    request: Request, uid: str, payload: TransformRequest
) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_apply_item_transform, bridge.editor, uid, payload)


@router.post("/delete")
async def delete_items(request: Request, payload: UidsRequest) -> dict:
    bridge = get_bridge(request)

    def _delete(editor) -> list[str]:
        items = _get_content_items(editor, payload.uids)
        editor.edit.remove_items(items)
        return [item.uid for item in items]

    removed = await bridge.run(_delete, bridge.editor)
    return {"removed": removed}


@router.post("/duplicate")
async def duplicate_items(request: Request, payload: UidsRequest) -> dict:
    bridge = get_bridge(request)

    def _duplicate(editor) -> list[str]:
        items = _get_content_items(editor, payload.uids)
        created = editor.edit.duplicate_items(items)
        return [item.uid for item in created]

    created = await bridge.run(_duplicate, bridge.editor)
    return {"created": created}


@router.post("/flip")
async def flip_items(request: Request, payload: FlipRequest) -> dict:
    if payload.axis not in ("horizontal", "vertical"):
        raise bad_request(_("axis must be 'horizontal' or 'vertical'"))
    bridge = get_bridge(request)

    def _flip(editor) -> list[str]:
        items = _get_content_items(editor, payload.uids)
        if payload.axis == "horizontal":
            editor.transform.flip_horizontal(items)
        else:
            editor.transform.flip_vertical(items)
        return [item.uid for item in items]

    flipped = await bridge.run(_flip, bridge.editor)
    return {"flipped": flipped}


_ALIGN_MODES = {
    "left": "align_left",
    "right": "align_right",
    "top": "align_top",
    "bottom": "align_bottom",
    "center_horizontal": "center_horizontally",
    "center_vertical": "center_vertically",
}

_SURFACE_DIM = {
    "align_right": 0,
    "align_top": 1,
    "center_horizontally": 0,
    "center_vertically": 1,
}


def _surface_size(editor) -> tuple[float, float]:
    machine = editor.context.machine
    if machine is None:
        raise machine_unavailable()
    width, height = machine.panel.extents
    return float(width), float(height)


@router.post("/align")
async def align_items(request: Request, payload: AlignRequest) -> dict:
    method_name = _ALIGN_MODES.get(payload.mode)
    if method_name is None:
        raise bad_request(
            _("Unknown align mode '{mode}'").format(mode=payload.mode)
        )
    bridge = get_bridge(request)

    def _align(editor) -> list[str]:
        items = _get_content_items(editor, payload.uids)
        is_centering = method_name.startswith("center")
        if len(items) < 2 and not is_centering:
            raise bad_request(_("Alignment needs at least two items"))
        method = getattr(editor.layout, method_name)
        dim = _SURFACE_DIM.get(method_name)
        if dim is None:
            method(items)
        else:
            method(items, _surface_size(editor)[dim])
        return [item.uid for item in items]

    aligned = await bridge.run(_align, bridge.editor)
    return {"aligned": aligned}


@router.post("/distribute")
async def distribute_items(
    request: Request, payload: DistributeRequest
) -> dict:
    if payload.axis not in ("horizontal", "vertical"):
        raise bad_request(_("axis must be 'horizontal' or 'vertical'"))
    bridge = get_bridge(request)

    def _distribute(editor) -> list[str]:
        items = _get_content_items(editor, payload.uids)
        if len(items) < 3:
            raise bad_request(_("Distribution needs at least three items"))
        if payload.axis == "horizontal":
            editor.layout.spread_horizontally(items)
        else:
            editor.layout.spread_vertically(items)
        return [item.uid for item in items]

    distributed = await bridge.run(_distribute, bridge.editor)
    return {"distributed": distributed}

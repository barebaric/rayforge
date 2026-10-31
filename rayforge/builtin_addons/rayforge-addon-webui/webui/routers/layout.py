"""Layout endpoints: list registered strategies and run one."""

import inspect
from gettext import gettext as _

from fastapi import APIRouter, Request
from pydantic import BaseModel

from rayforge.core.undo.property_cmd import ChangePropertyCommand
from rayforge.doceditor.layout.base import LayoutStrategy
from rayforge.doceditor.layout.registry import layout_registry

from .deps import bad_request, get_bridge, not_found

router = APIRouter(tags=["layout"])

STRATEGY_LABELS = {
    "pixel-perfect": _("Auto Layout"),
}


class LayoutRequest(BaseModel):
    strategy: str
    params: dict[str, float] | None = None


def _strategy_label(name: str) -> str:
    label = STRATEGY_LABELS.get(name)
    if label:
        return label
    return _("{name} layout").format(name=name)


def _list_layouts() -> list[dict]:
    return [
        {"id": name, "label": _strategy_label(name)}
        for name in layout_registry.list_names()
    ]


@router.get("/layouts")
async def get_layouts(request: Request) -> list[dict]:
    bridge = get_bridge(request)
    return await bridge.run(_list_layouts)


def _active_layer_items(editor) -> list:
    layer = editor.doc.active_layer
    if layer is None:
        return []
    return list(layer.get_content_items())


def _build_strategy(name: str, items: list, params: dict | None):
    strategy_class = layout_registry.get(name)
    if strategy_class is None:
        raise not_found(_("Layout strategy not found"))
    accepted = inspect.signature(strategy_class.__init__).parameters
    for key in params or {}:
        if key not in accepted:
            raise bad_request(
                _("Unknown layout parameter: {key}").format(key=key)
            )
    try:
        return strategy_class(items=items, **(params or {}))
    except ValueError as exc:
        raise bad_request(str(exc)) from exc


async def _calculate_deltas(editor, strategy: LayoutStrategy) -> dict:
    supports_async = (
        type(strategy).calculate_deltas_async
        is not LayoutStrategy.calculate_deltas_async
    )
    if not supports_async:
        return strategy.calculate_deltas()
    return await strategy.calculate_deltas_async(None, editor.task_manager)


def _apply_deltas(editor, deltas: dict, label: str) -> int:
    if not deltas:
        return 0
    with editor.history_manager.transaction(label) as t:
        for item, delta_matrix in deltas.items():
            old_matrix = item.matrix.copy()
            new_matrix = delta_matrix @ old_matrix
            cmd = ChangePropertyCommand(
                target=item,
                property_name="matrix",
                new_value=new_matrix,
                old_value=old_matrix,
            )
            t.execute(cmd)
    return len(deltas)


async def _run_layout(editor, name: str, params: dict | None) -> dict:
    items = _active_layer_items(editor)
    if not items:
        raise bad_request(_("Nothing to arrange on the active layer"))
    strategy = _build_strategy(name, items, params)
    label = _strategy_label(name)
    messages: list[str] = []

    def on_error_reported(sender, message: str) -> None:
        messages.append(message)

    strategy.error_reported.connect(on_error_reported)
    try:
        deltas = await _calculate_deltas(editor, strategy)
    finally:
        strategy.error_reported.disconnect(on_error_reported)
    moved = _apply_deltas(editor, deltas, label)
    for message in messages:
        editor.notification_requested.send(
            editor, message=message, persistent=False
        )
    return {"moved": moved}


@router.post("/layout")
async def run_layout(request: Request, body: LayoutRequest) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(
        _run_layout, bridge.editor, body.strategy, body.params
    )

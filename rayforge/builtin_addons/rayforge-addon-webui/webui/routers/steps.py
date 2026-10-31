"""Step endpoints: registry listing, add, update, delete, schema."""

from gettext import gettext as _

from fastapi import APIRouter, Request
from pydantic import BaseModel

from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.step_registry import step_registry
from rayforge.core.undo import ListItemCommand
from rayforge.core.undo.property_cmd import ChangePropertyCommand
from rayforge.core.workflow import Workflow

from ..serializers import find_item_by_uid
from ..serializers.doc_tree import serialize_step
from ..serializers.varset_schema import varset_groups_schema
from .deps import bad_request, get_bridge, item_not_found, not_found

router = APIRouter(tags=["steps"])


class StepAddRequest(BaseModel):
    step_type: str
    name: str | None = None
    params: dict = {}


class StepUpdateRequest(BaseModel):
    name: str | None = None
    params: dict = {}


def _serialize_step_type(step_cls: type[Step]) -> dict:
    return {
        "step_type": step_cls.__name__,
        "typelabel": step_cls.TYPELABEL,
        "icon": step_cls.ICON,
        "required_machine_caps": sorted(
            cap.name for cap in step_cls.REQUIRED_MACHINE_CAPS
        ),
        "hidden": step_cls.HIDDEN,
    }


def _available_step_types(context) -> list[dict]:
    capabilities = (
        context.machine.get_capabilities() if context.machine else None
    )
    available = []
    for name, step_cls in sorted(step_registry.all_steps().items()):
        if step_cls.HIDDEN:
            continue
        if capabilities is not None and not (
            step_cls.REQUIRED_MACHINE_CAPS.issubset(capabilities)
        ):
            continue
        entry = _serialize_step_type(step_cls)
        entry["schema"] = varset_groups_schema(step_cls.recipe_varset_groups())
        available.append(entry)
    return available


@router.get("/steps")
async def get_steps(request: Request) -> dict:
    bridge = get_bridge(request)

    def _steps(editor) -> dict:
        return {"steps": _available_step_types(editor.context)}

    return await bridge.run(_steps, bridge.editor)


def _get_workflow(editor, layer_uid: str) -> Workflow:
    layer = find_item_by_uid(editor.doc, layer_uid)
    if not isinstance(layer, Layer) or layer.workflow is None:
        raise item_not_found()
    return layer.workflow


def _get_step(editor, uid: str) -> Step:
    item = find_item_by_uid(editor.doc, uid)
    if not isinstance(item, Step):
        raise item_not_found()
    return item


def _step_updated_callback(step: Step):
    def _send_updated() -> None:
        step.updated.send(step)

    return _send_updated


def _apply_params(editor, step: Step, params: dict, label: str) -> None:
    history = editor.history_manager
    with history.transaction(label) as t:
        for key, value in params.items():
            setter_name = step.get_recipe_setter_name(key)
            if setter_name is None and not hasattr(step, key):
                raise bad_request(
                    _("Unknown step parameter '{key}'").format(key=key)
                )
            t.execute(
                ChangePropertyCommand(
                    target=step,
                    property_name=key,
                    new_value=value,
                    setter_method_name=setter_name,
                    on_change_callback=_step_updated_callback(step),
                    name=_("Set {key}").format(key=key),
                )
            )


def _add_step(editor, layer_uid: str, payload: StepAddRequest) -> dict:
    step_cls = step_registry.get(payload.step_type)
    if step_cls is None:
        raise not_found(
            _("Unknown step type '{step_type}'").format(
                step_type=payload.step_type
            )
            + ". "
            + _("No matching step type is registered.")
        )
    workflow = _get_workflow(editor, layer_uid)
    step = step_cls.create(editor.context, name=payload.name)
    history = editor.history_manager
    with history.transaction(_("Add step")) as t:
        t.execute(
            ListItemCommand(
                owner_obj=workflow,
                item=step,
                undo_command="remove_child",
                redo_command="add_child",
                name=_("Add step"),
            )
        )
    if payload.params:
        _apply_params(editor, step, payload.params, _("Configure step"))
    result = serialize_step(step, workflow.steps.index(step))
    result["layer_uid"] = layer_uid
    return result


@router.post("/layers/{uid}/steps")
async def add_step(
    request: Request, uid: str, payload: StepAddRequest
) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_add_step, bridge.editor, uid, payload)


def _update_step(editor, uid: str, payload: StepUpdateRequest) -> dict:
    step = _get_step(editor, uid)
    if payload.name is not None and payload.name != step.name:
        editor.step.rename_step(step, payload.name)
    if payload.params:
        _apply_params(editor, step, payload.params, _("Configure step"))
    workflow = step.workflow
    order = workflow.steps.index(step) if workflow else 0
    return serialize_step(step, order)


@router.put("/steps/{uid}")
async def update_step(
    request: Request, uid: str, payload: StepUpdateRequest
) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_update_step, bridge.editor, uid, payload)


def _delete_step(editor, uid: str) -> dict:
    step = _get_step(editor, uid)
    workflow = step.workflow
    if workflow is None:
        raise bad_request(_("Step has no parent workflow"))
    history = editor.history_manager
    with history.transaction(_("Remove step")) as t:
        t.execute(
            ListItemCommand(
                owner_obj=workflow,
                item=step,
                undo_command="add_child",
                redo_command="remove_child",
                name=_("Remove step"),
            )
        )
    return {"removed": uid}


@router.delete("/steps/{uid}")
async def delete_step(request: Request, uid: str) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_delete_step, bridge.editor, uid)


def _step_schema(editor, uid: str) -> dict:
    step = _get_step(editor, uid)
    return {
        "step_type": type(step).__name__,
        "groups": varset_groups_schema(step.recipe_varset_groups()),
    }


@router.get("/steps/{uid}/schema")
async def get_step_schema(request: Request, uid: str) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_step_schema, bridge.editor, uid)


@router.get("/layers/{uid}/steps")
async def get_layer_steps(request: Request, uid: str) -> dict:
    bridge = get_bridge(request)

    def _layer_steps(editor) -> dict:
        workflow = _get_workflow(editor, uid)
        return {
            "layer_uid": uid,
            "steps": [
                serialize_step(step, order)
                for order, step in enumerate(workflow.steps)
            ],
        }

    return await bridge.run(_layer_steps, bridge.editor)

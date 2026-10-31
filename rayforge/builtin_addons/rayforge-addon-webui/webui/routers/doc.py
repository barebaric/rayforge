"""Document-level endpoints: tree, history, undo/redo, new document."""

from fastapi import APIRouter, Request

from rayforge.core.doc import Doc

from ..serializers import doc_tree
from .deps import get_bridge, tree_with_revision

router = APIRouter(tags=["doc"])


def _history_payload(editor) -> dict:
    history = editor.history_manager
    return {
        "can_undo": history.can_undo(),
        "can_redo": history.can_redo(),
        "undo_labels": [command.name for command in history.undo_stack],
        "redo_labels": [command.name for command in history.redo_stack],
        "dirty": not editor.is_saved,
    }


@router.get("/doc/tree")
async def get_doc_tree(request: Request) -> dict:
    bridge = get_bridge(request)
    tree = await bridge.run(doc_tree, bridge.editor)
    return tree_with_revision(request, tree)


@router.get("/doc/history")
async def get_doc_history(request: Request) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_history_payload, bridge.editor)


@router.post("/doc/undo")
async def undo(request: Request) -> dict:
    bridge = get_bridge(request)

    def _undo(editor) -> None:
        editor.history_manager.undo()

    await bridge.run(_undo, bridge.editor)
    return await bridge.run(_history_payload, bridge.editor)


@router.post("/doc/redo")
async def redo(request: Request) -> dict:
    bridge = get_bridge(request)

    def _redo(editor) -> None:
        editor.history_manager.redo()

    await bridge.run(_redo, bridge.editor)
    return await bridge.run(_history_payload, bridge.editor)


def _create_new_document(editor) -> dict:
    new_doc = Doc()
    machine = editor.context.machine
    if machine:
        new_doc.active_layer.set_rotary_enabled(machine.rotary_enabled_default)
        default_rm = machine.get_default_rotary_module()
        if default_rm:
            new_doc.active_layer.set_rotary_diameter(
                default_rm.default_diameter
            )
            new_doc.active_layer.set_rotary_module_uid(default_rm.uid)
    editor.set_doc(new_doc)
    editor.set_file_path(None)
    editor.mark_as_saved()
    return doc_tree(editor)


@router.post("/doc/new")
async def new_document(request: Request) -> dict:
    bridge = get_bridge(request)
    tree = await bridge.run(_create_new_document, bridge.editor)
    return tree_with_revision(request, tree)

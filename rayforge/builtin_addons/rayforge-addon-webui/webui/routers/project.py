"""Import and project endpoints: file upload, .ryp open/save, gcode."""

import asyncio
import json
import tempfile
from gettext import gettext as _
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from rayforge.core.item import DocItem
from rayforge.core.vectorization_spec import VectorizationSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image.base_importer import ImporterFeature

from ..serializers import doc_tree
from .deps import bad_request, get_bridge, not_found, tree_with_revision

router = APIRouter(tags=["import"])


class ImportPositionRequest(BaseModel):
    position_mm: list[float] | None = None


def _collect_workpieces(items: list[DocItem]) -> list[WorkPiece]:
    found: list[WorkPiece] = []
    for item in items:
        if isinstance(item, WorkPiece):
            found.append(item)
        found.extend(item.get_descendants(WorkPiece))
    return found


def _manifest_dict(manifest) -> dict:
    return {
        "title": manifest.title,
        "natural_size_mm": (
            list(manifest.natural_size_mm)
            if manifest.natural_size_mm
            else None
        ),
        "layers": [
            {
                "id": layer.id,
                "name": layer.name,
                "default_active": layer.default_active,
                "feature_count": layer.feature_count,
            }
            for layer in manifest.layers
        ],
        "warnings": list(manifest.warnings),
        "errors": list(manifest.errors),
    }


def _analyze_import(editor, data: bytes, filename: str, mime: str) -> dict:
    path = Path(filename)
    importer_cls, features = editor.file.get_importer_info(path, mime)
    if importer_cls is None:
        return {
            "supported": False,
            "action": "unsupported",
            "importer": None,
            "features": [],
            "manifest": None,
        }
    manifest = editor.file.scan_import_file(data, path, mime)
    action = editor.file.analyze_import_target(path, mime)
    feature_names = sorted(
        feature.name
        for feature in ImporterFeature
        if feature in features and feature.name is not None
    )
    return {
        "supported": True,
        "action": action.name.lower(),
        "importer": importer_cls.__name__,
        "features": feature_names,
        "manifest": _manifest_dict(manifest),
    }


@router.post("/import/analyze")
async def analyze_import(request: Request, file: UploadFile) -> dict:
    bridge = get_bridge(request)
    data = await file.read()
    mime = file.content_type or "application/octet-stream"
    result = await bridge.run(
        _analyze_import, bridge.editor, data, file.filename, mime
    )
    return result


def _parse_spec(spec: dict | None):
    if spec is None:
        return None
    try:
        return VectorizationSpec.from_dict(spec)
    except (ValueError, KeyError) as e:
        raise bad_request(
            _("Invalid vectorization spec: {error}").format(error=str(e))
        ) from e


async def _import_flow(editor, data, filename, mime, spec, position):
    importer_cls, _features = editor.file.get_importer_info(
        Path(filename), mime
    )
    if importer_cls is None:
        raise bad_request(
            _("No importer for '{filename}'").format(filename=filename)
        )
    importer = importer_cls(data, source_file=Path(filename))
    import_result = await asyncio.to_thread(importer.get_doc_items, spec)
    payload = import_result.payload if import_result else None
    if payload is None or not payload.items:
        raise bad_request(
            _("Import produced no items from '{filename}'").format(
                filename=filename
            )
        )
    editor.file._finalize_import_on_main_thread(
        payload, Path(filename), position, spec
    )
    workpieces = _collect_workpieces(payload.items)
    layer = editor.doc.active_layer
    return {
        "created": [workpiece.uid for workpiece in workpieces],
        "layer_uid": layer.uid if workpieces else None,
    }


@router.post("/import")
async def import_file(
    request: Request,
    file: UploadFile,
    spec: str | None = None,
    x: float | None = None,
    y: float | None = None,
) -> dict:
    bridge = get_bridge(request)
    data = await file.read()
    mime = file.content_type or "application/octet-stream"
    spec_dict = json.loads(spec) if spec else None
    vectorization_spec = _parse_spec(spec_dict)
    position = (x, y) if x is not None and y is not None else None
    return await bridge.run(
        _import_flow,
        bridge.editor,
        data,
        file.filename,
        mime,
        vectorization_spec,
        position,
    )


def _open_project(editor, data: bytes, filename: str) -> dict:
    suffix = Path(filename).suffix or ".ryp"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        path = Path(handle.name)
    try:
        ok = editor.file.load_project_from_path(path)
    finally:
        path.unlink(missing_ok=True)
    if not ok:
        raise bad_request(
            _("Failed to load project '{filename}'").format(filename=filename)
        )
    return doc_tree(editor)


@router.post("/project/open")
async def open_project(request: Request, file: UploadFile) -> dict:
    bridge = get_bridge(request)
    data = await file.read()
    tree = await bridge.run(_open_project, bridge.editor, data, file.filename)
    return tree_with_revision(request, tree)


def _delete_path(path: Path) -> None:
    path.unlink(missing_ok=True)


def _save_project(editor) -> str:
    old_path = editor.file_path
    was_saved = editor.is_saved
    with tempfile.NamedTemporaryFile(suffix=".ryp", delete=False) as handle:
        path = Path(handle.name)
    if not editor.file.save_project_to_path(path):
        path.unlink(missing_ok=True)
        raise bad_request(_("Failed to save project"))
    editor.set_file_path(old_path)
    if was_saved:
        editor.mark_as_saved()
    else:
        editor.mark_as_unsaved()
    return str(path)


@router.get("/project/save")
async def save_project(request: Request) -> FileResponse:
    bridge = get_bridge(request)
    path = Path(await bridge.run(_save_project, bridge.editor))
    return FileResponse(
        path=path,
        media_type="application/octet-stream",
        filename="project.ryp",
        background=BackgroundTask(_delete_path, path),
    )


def _export_gcode(editor) -> str:
    handle = editor.pipeline.get_existing_job_handle()
    if handle is None:
        raise not_found(_("No generated gcode available"))
    with editor.context.artifact_store.checkout_handle(handle) as artifact:
        if artifact is None or artifact.machine_code is None:
            raise not_found(_("No generated gcode available"))
        return artifact.machine_code


@router.get("/project/gcode")
async def get_gcode(request: Request) -> PlainTextResponse:
    bridge = get_bridge(request)
    gcode = await bridge.run(_export_gcode, bridge.editor)
    return PlainTextResponse(
        content=gcode,
        media_type="text/plain",
        headers={"Content-Disposition": 'attachment; filename="gcode.g"'},
    )

"""Ops preview endpoint: world-space segments of the current job artifact."""

from gettext import gettext as _

from fastapi import APIRouter, Request

from ..serializers import ops_segments
from .deps import get_bridge, not_found

router = APIRouter(tags=["ops"])


def _no_ops() -> Exception:
    return not_found(_("No job operations available"))


@router.get("/ops")
async def get_ops(request: Request, tolerance: float = 0.0) -> dict:
    bridge = get_bridge(request)

    def _serialize(editor):
        handle = editor.pipeline.get_existing_job_handle()
        if handle is None:
            raise _no_ops()
        store = editor.context.artifact_store
        with store.checkout_handle(handle) as artifact:
            if artifact is None or artifact.ops is None:
                raise _no_ops()
            result = ops_segments(artifact.ops, max(tolerance, 0.0))
            result["generation_id"] = artifact.generation_id
        return result

    return await bridge.run(_serialize, bridge.editor)

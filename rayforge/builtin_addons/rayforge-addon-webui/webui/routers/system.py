"""System information endpoints."""

import sys

from fastapi import APIRouter, Request

from rayforge import __version__
from rayforge.core.hooks import PLUGIN_API_VERSION

from ..coordinates import viewport_contract_dict
from .deps import get_bridge, machine_unavailable

router = APIRouter(tags=["system"])


@router.get("/system/info")
async def get_system_info(request: Request) -> dict:
    context = request.app.state.context
    return {
        "name": "rayforge",
        "version": __version__,
        "api_version": PLUGIN_API_VERSION,
        "python_version": sys.version.split()[0],
        "machine_count": len(context.machine_mgr.machines),
    }


@router.get("/viewport")
async def get_viewport(request: Request) -> dict:
    bridge = get_bridge(request)

    def _viewport(editor):
        if editor.context.machine is None:
            raise machine_unavailable()
        return viewport_contract_dict(editor.context)

    return await bridge.run(_viewport, bridge.editor)

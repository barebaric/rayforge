"""FastAPI application factory for the rayforge web UI."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from rayforge import __version__
from rayforge.machine.cmd import MachineCmd

from .core_bridge import CoreBridge
from .routers import (
    doc,
    items,
    job,
    layers,
    layout,
    machine_control,
    machines,
    ops,
    project,
    settings,
    steps,
    system,
)
from .ws import EventHub, EventSource
from .ws import router as ws_router

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
API_PREFIX = "/api/v1"

ROUTERS = (
    system.router,
    doc.router,
    items.router,
    layers.router,
    steps.router,
    project.router,
    machines.router,
    machine_control.router,
    job.router,
    ops.router,
    layout.router,
    settings.router,
    ws_router,
)


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Rejects HTTP requests that lack the configured access token."""

    def __init__(self, app, token: str):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        header = request.headers.get("Authorization")
        if header != f"Bearer {self.token}":
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing access token"},
            )
        return await call_next(request)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    hub: EventHub = app.state.hub
    hub.set_loop(asyncio.get_running_loop())
    app.state.events.connect()
    logger.info("Web UI server started")
    yield
    app.state.events.disconnect()
    logger.info("Web UI server shutting down")


def create_app(editor, context, token: str | None = None) -> FastAPI:
    """
    Builds the web application for the given headless environment.

    Args:
        editor: The DocEditor owning the document.
        context: The RayforgeContext providing machines and services.
        token: When set, all requests must present this bearer token.
    """
    app = FastAPI(
        title="Rayforge",
        version=__version__ or "",
        lifespan=_lifespan,
    )
    bridge = CoreBridge(editor, context)
    app.state.editor = editor
    app.state.context = context
    app.state.token = token
    app.state.bridge = bridge
    app.state.machine_cmd = MachineCmd(editor)
    app.state.hub = EventHub()
    app.state.events = EventSource(
        editor, context, app.state.hub, app.state.machine_cmd
    )

    if token:
        app.add_middleware(TokenAuthMiddleware, token=token)

    for router in ROUTERS:
        app.include_router(router, prefix=API_PREFIX)

    if (STATIC_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True))

    return app

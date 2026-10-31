"""
WebSocket hub: fan-out of core events to connected web clients.

Blinker signal handlers run on the TaskManager loop (the headless
scheduler in :mod:`rayforge.serve` routes them there). The hub lives on
the uvicorn loop; publishers hand events over with
``asyncio.run_coroutine_threadsafe`` so the core loop never blocks on
network I/O.
"""

import asyncio
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .coordinates import viewport_contract_dict
from .serializers import machine_state
from .serializers.doc_tree import tree_digest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ws"])

MACHINE_EVENT_INTERVAL_S = 0.2


def _uid_of(origin) -> str | None:
    return getattr(origin, "uid", None)


class EventHub:
    """Holds connected WebSockets and broadcasts JSON events to them."""

    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the uvicorn loop at lifespan startup."""
        self._loop = loop

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def broadcast(self, event: dict) -> None:
        if not self._clients:
            return
        payload = json.dumps(event, default=str)
        dead = []
        for websocket in self._clients:
            try:
                await websocket.send_text(payload)
            except Exception:  # noqa: BLE001 - drop broken clients
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)

    async def send_to(self, websocket: WebSocket, event: dict) -> bool:
        try:
            await websocket.send_text(json.dumps(event, default=str))
            return True
        except Exception:  # noqa: BLE001 - drop broken clients
            self.disconnect(websocket)
            return False

    def publish(self, event: dict) -> None:
        """Publish from any thread; routes the broadcast correctly."""
        if self._loop is None or self._loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            asyncio.ensure_future(self.broadcast(event))
        else:
            asyncio.run_coroutine_threadsafe(self.broadcast(event), self._loop)


class RateLimiter:
    """Allows at most one event per interval (drop, not queue)."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def allow(self) -> bool:
        now = time.monotonic()
        if now - self._last >= self.min_interval:
            self._last = now
            return True
        return False


class EventSource:
    """
    Subscribes to core blinker signals and publishes hub events.

    All subscriptions are made once in the app lifespan. Document
    subscriptions are re-targeted whenever the editor swaps its Doc,
    and machine subscriptions whenever the active machine changes.
    """

    def __init__(self, editor, context, hub: EventHub, machine_cmd):
        self._editor = editor
        self._context = context
        self._hub = hub
        self._machine_cmd = machine_cmd
        self._machine_limiter = RateLimiter(MACHINE_EVENT_INTERVAL_S)
        self._machine: object | None = None
        self._tree_rev = 0
        self._doc_connections: list = []
        self._machine_connections: list = []
        self._editor_connections: list = []

    @property
    def tree_revision(self) -> int:
        """Monotonic document revision, bumped on every doc event."""
        return self._tree_rev

    def connect(self) -> None:
        self._connect_doc()
        self._connect_editor()
        self._connect_machine()

    def disconnect(self) -> None:
        self._disconnect_doc()
        self._disconnect_editor()
        self._disconnect_machine()

    def publish_doc_event(self, kind: str, uids=None, **extra) -> None:
        self._tree_rev += 1
        revisions = self._doc_revisions()
        revisions["tree"] = self._tree_rev
        event = {
            "type": "doc",
            "kind": kind,
            "uids": uids or [],
            "revisions": revisions,
        }
        event.update(extra)
        self._hub.publish(event)

    def publish_machine_event(self, force: bool = False) -> None:
        if not force and not self._machine_limiter.allow():
            return
        machine = self._context.machine
        if machine is None:
            return
        self._hub.publish({"type": "machine", "data": machine_state(machine)})

    def publish_viewport_event(self, *args, **kwargs) -> None:
        machine = self._context.machine
        if machine is None:
            return
        self._hub.publish(
            {"type": "viewport", "data": viewport_contract_dict(self._context)}
        )

    def publish_job_event(self, *args, **kwargs) -> None:
        machine = self._context.machine
        if machine is None:
            return
        current = machine.current_job
        data = (
            {"active": True, "job": _job_snapshot(current)}
            if current is not None
            else {"active": False, "job": None}
        )
        self._hub.publish({"type": "job", "data": data})

    def on_job_progress(self, metrics: dict) -> None:
        """Progress callback passed to MachineCmd.send_job/frame_job."""
        self.publish_job_event()

    def _doc_revisions(self) -> dict:
        from .serializers.doc_tree import item_revisions

        return item_revisions(self._editor.doc)

    def _doc_handler(self, kind: str):
        def handler(sender, **kwargs):
            origin = kwargs.get("origin", sender)
            self.publish_doc_event(kind, uids=[_uid_of(origin)])

        return handler

    def _processing_handler(self, sender, **kwargs) -> None:
        is_processing = bool(kwargs.get("is_processing", False))
        self.publish_doc_event("processing", is_processing=is_processing)

    def _settled_handler(self, sender, **kwargs) -> None:
        self.publish_doc_event("settled")

    def _notify_handler(self, sender, **kwargs) -> None:
        message = str(kwargs.get("message", ""))
        level = "warning" if kwargs.get("persistent") else "info"
        self._hub.publish(
            {"type": "notify", "level": level, "message": message}
        )

    def _history_handler(self, sender, **kwargs) -> None:
        self.publish_doc_event("history")

    def _document_changed_handler(self, sender, **kwargs) -> None:
        self._disconnect_doc()
        self._connect_doc()
        self.publish_doc_event("structure")

    def _connect_doc(self) -> None:
        doc = self._editor.doc
        subscriptions = [
            (doc.updated, self._doc_handler("updated")),
            (doc.transform_changed, self._doc_handler("transform")),
            (doc.descendant_added, self._doc_handler("structure")),
            (doc.descendant_removed, self._doc_handler("structure")),
            (doc.descendant_updated, self._doc_handler("updated")),
            (
                doc.descendant_transform_changed,
                self._doc_handler("transform"),
            ),
            (doc.descendant_renamed, self._doc_handler("structure")),
            (doc.renamed, self._doc_handler("structure")),
            (doc.active_layer_changed, self._doc_handler("active-layer")),
            (self._editor.history_manager.changed, self._history_handler),
        ]
        for signal, handler in subscriptions:
            signal.connect(handler)
            self._doc_connections.append((signal, handler))

    def _disconnect_doc(self) -> None:
        for signal, handler in self._doc_connections:
            try:
                signal.disconnect(handler)
            except Exception:  # noqa: BLE001 - stale refs are harmless
                logger.debug("Dropping stale signal connection")
        self._doc_connections.clear()

    def _connect_editor(self) -> None:
        subscriptions = [
            (
                self._editor.notification_requested,
                self._notify_handler,
            ),
            (
                self._editor.processing_state_changed,
                self._processing_handler,
            ),
            (self._editor.document_settled, self._settled_handler),
            (self._editor.document_changed, self._document_changed_handler),
        ]
        for signal, handler in subscriptions:
            signal.connect(handler)
            self._editor_connections.append((signal, handler))

    def _disconnect_editor(self) -> None:
        for signal, handler in self._editor_connections:
            try:
                signal.disconnect(handler)
            except Exception:  # noqa: BLE001 - stale refs are harmless
                logger.debug("Dropping stale signal connection")
        self._editor_connections.clear()

    def _connect_machine(self) -> None:
        machine = self._context.machine
        if machine is None:
            return
        subscriptions = [
            (
                machine.state_changed,
                lambda sender, **kw: self.publish_machine_event(),
            ),
            (
                machine.connection_status_changed,
                lambda sender, **kw: self.publish_machine_event(force=True),
            ),
            (machine.changed, self.publish_viewport_event),
            (machine.wcs_updated, self.publish_viewport_event),
            (
                machine.job_finished,
                lambda sender, **kw: self.publish_job_event(),
            ),
            (
                self._machine_cmd.job_started,
                lambda sender, **kw: self.publish_job_event(),
            ),
        ]
        for signal, handler in subscriptions:
            signal.connect(handler)
            self._machine_connections.append((signal, handler))
        self._machine = machine

    def _disconnect_machine(self) -> None:
        for signal, handler in self._machine_connections:
            try:
                signal.disconnect(handler)
            except Exception:  # noqa: BLE001 - stale refs are harmless
                logger.debug("Dropping stale signal connection")
        self._machine_connections.clear()
        self._machine = None


def _job_snapshot(status) -> dict | None:
    if status is None:
        return None
    return {
        "phase": status.phase.name.lower(),
        "progress_fraction": status.progress_fraction,
        "eta_seconds": status.eta_seconds,
        "elapsed_seconds": status.elapsed_seconds,
    }


def _initial_snapshot(editor, context) -> list[dict]:
    events: list[dict] = [
        {"type": "doc", "kind": "snapshot", "data": tree_digest(editor)}
    ]
    machine = context.machine
    if machine is not None:
        events.append({"type": "machine", "data": machine_state(machine)})
        events.append(
            {
                "type": "viewport",
                "data": viewport_contract_dict(context),
            }
        )
    return events


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    app = websocket.app
    token = app.state.token
    if token and not _ws_token_ok(websocket, token):
        await websocket.close(code=1008)
        return
    hub: EventHub = app.state.hub
    await hub.connect(websocket)
    editor = app.state.editor
    context = app.state.context
    try:
        for event in _initial_snapshot(editor, context):
            if not await hub.send_to(websocket, event):
                return
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub.disconnect(websocket)


def _ws_token_ok(websocket: WebSocket, token: str) -> bool:
    header = websocket.headers.get("Authorization")
    if header == f"Bearer {token}":
        return True
    return websocket.query_params.get("token") == token

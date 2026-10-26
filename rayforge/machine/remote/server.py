"""
Headless job server for remote execution.

Runs on the machine that is physically attached to the laser cutter.
It accepts fully computed :class:`JobPayload`s from a client over
TCP, executes them on the configured driver through the regular
driver code paths, and streams job progress back. Jobs are owned by
the server once accepted: the client may disconnect at any time
(e.g. close the laptop) and reconnect later to re-attach to a
running job via ``get_job``.

Single user, single job: one job runs at a time and progress is
broadcast to all connected clients.

Threading model: the network loop is owned by :meth:`serve`, while
jobs and drivers run on the task manager's event loop, as they do in
the GUI application. Events cross from the task loop to the network
loop via :meth:`asyncio.loop.call_soon_threadsafe`, and control
requests cross back via :func:`asyncio.run_coroutine_threadsafe`.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from typing import TYPE_CHECKING, Any, cast

from rayforge import __version__

from ..job_monitor import JobMonitor
from ..job_status import JobOutcome
from .payload import JobPayload
from .protocol import (
    PROTOCOL_VERSION,
    encode_message,
    read_message,
    write_message,
)

if TYPE_CHECKING:
    from ...context import RayforgeContext
    from ...core.doc import Doc
    from ...shared.tasker.manager import TaskManager
    from ..models.machine import Machine

logger = logging.getLogger(__name__)


class RemoteJobServer:
    """Serves job submission and progress over TCP."""

    def __init__(
        self,
        context: RayforgeContext,
        task_manager: TaskManager,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        self._context = context
        self._task_manager = task_manager
        self._host = host
        self._port = port
        self._server: asyncio.Server | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connections: set[asyncio.StreamWriter] = set()
        self._job: _RunningJob | None = None

    @property
    def port(self) -> int:
        """The port the server listens on."""
        assert self._server is not None, "Server not started"
        return self._server.sockets[0].getsockname()[1]

    async def start(self) -> None:
        """Starts listening for client connections."""
        self._loop = asyncio.get_running_loop()
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port
        )
        logger.info(f"Remote job server listening on {self._host}:{self.port}")

    async def stop(self) -> None:
        """Closes the listening socket and all client connections."""
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        for writer in list(self._connections):
            writer.close()
        self._connections.clear()
        self._server = None

    async def serve_forever(self) -> None:
        """Runs until the server is stopped or the process exits."""
        assert self._server is not None, "Server not started"
        async with self._server:
            await self._server.serve_forever()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        logger.info(f"Client connected: {peer}")
        self._connections.add(writer)
        try:
            while True:
                message = await read_message(reader)
                for response in await self._dispatch(message):
                    await write_message(writer, response)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception:
            logger.exception("Error while handling client")
            try:
                await write_message(
                    writer, {"type": "error", "message": "internal error"}
                )
            except Exception:
                logger.debug("Could not send error to client", exc_info=True)
        finally:
            self._connections.discard(writer)
            writer.close()
            logger.info(f"Client disconnected: {peer}")

    async def _dispatch(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Handles one client message, returning the replies."""
        mtype = message.get("type")
        if mtype == "hello":
            return [self._welcome(message)]
        if mtype == "get_job":
            return [self._job_state()]
        if mtype == "submit_job":
            return [await self._submit_job(message)]
        if mtype == "cancel_job":
            return [self._control_job(message, "cancel_job")]
        if mtype == "set_hold":
            return [self._control_job(message, "set_hold")]
        return [
            {
                "type": "error",
                "message": f"Unknown message type: {mtype!r}",
            }
        ]

    def _welcome(self, message: dict[str, Any]) -> dict[str, Any]:
        version = message.get("version")
        if version != PROTOCOL_VERSION:
            return {
                "type": "error",
                "message": (
                    f"Protocol version mismatch: client {version}, "
                    f"server {PROTOCOL_VERSION}."
                ),
            }
        machines = [
            {
                "id": machine.id,
                "name": machine.name,
                "driver_name": machine.driver_name,
                "connected": machine.is_connected(),
            }
            for machine in self._context.machine_mgr.get_machines()
        ]
        return {
            "type": "welcome",
            "version": PROTOCOL_VERSION,
            "app_version": __version__,
            "machines": machines,
        }

    def _job_state(self) -> dict[str, Any]:
        if self._job is None:
            return {"type": "job_state", "job": None}
        return {"type": "job_state", "job": self._job.state()}

    async def _submit_job(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._job is not None and self._job.running:
            return {
                "type": "error",
                "message": "A job is already running.",
            }

        machine_id = message.get("machine_id")
        machine = self._context.machine_mgr.get_machine_by_id(machine_id)
        if machine is None:
            return {
                "type": "error",
                "message": f"No machine with id {machine_id!r}.",
            }

        try:
            payload = JobPayload.from_bytes(
                base64.b64decode(message["payload"])
            )
        except (KeyError, ValueError, TypeError) as e:
            return {"type": "error", "message": f"Bad payload: {e}"}
        if payload.ops is None:
            return {
                "type": "error",
                "message": "Payload contains no ops.",
            }

        job = _RunningJob(self, machine.id, payload)
        self._job = job
        # The job runs on the task manager's loop, where the drivers
        # and their transports live; it survives client disconnects.
        job.future = asyncio.run_coroutine_threadsafe(
            self._execute_job(job, machine), self._task_manager.loop
        )
        return {"type": "job_accepted", "job_id": job.id}

    def _control_job(
        self, message: dict[str, Any], action: str
    ) -> dict[str, Any]:
        job = self._job
        if job is None or job.id != message.get("job_id"):
            return {"type": "error", "message": "No such job."}
        if not job.running:
            return {"type": "error", "message": "Job is not running."}

        machine = self._context.machine_mgr.get_machine_by_id(job.machine_id)
        if machine is None:
            return {"type": "error", "message": "Machine is gone."}

        if action == "cancel_job":
            coroutine = machine.driver.cancel()
        else:
            coroutine = machine.driver.set_hold(bool(message.get("hold")))
        asyncio.run_coroutine_threadsafe(coroutine, self._task_manager.loop)
        return {"type": "ok", "action": action}

    async def _execute_job(self, job: _RunningJob, machine: Machine) -> None:
        """Runs a submitted job on the task manager's event loop."""
        payload = job.payload
        ops = payload.ops
        assert ops is not None
        try:
            if not machine.is_connected():
                await machine.connect()

            estimated_seconds = payload.estimated_seconds
            if estimated_seconds is None:
                estimated_seconds = ops.estimate_time(
                    default_feed_rate=machine.max_cut_speed,
                    default_rapid_rate=machine.max_travel_speed,
                    acceleration=machine.acceleration,
                )
            monitor = JobMonitor(
                ops,
                machine=machine,
                kind=payload.kind,
                context=self._context,
                estimated_seconds=estimated_seconds,
                default_feed_rate=machine.max_cut_speed,
                default_rapid_rate=machine.max_travel_speed,
                acceleration=machine.acceleration,
            )
            job.monitor = monitor
            monitor.progress_updated.connect(job.on_progress)
            monitor.begin()
            try:
                on_command_done = (
                    monitor.update_progress
                    if machine.reports_granular_progress
                    else None
                )
                # Drivers only consume the doc during encoding, which
                # happened on the client; the server has no document.
                doc = cast("Doc", None)
                await machine.driver.run(
                    payload.encoded,
                    doc,
                    ops,
                    on_command_done=on_command_done,
                )
            except asyncio.CancelledError:
                monitor.end(JobOutcome.CANCELLED)
                job.finish("CANCELLED", "")
                raise
            except Exception as e:
                logger.exception("Remote job failed")
                monitor.end(JobOutcome.FAILED, str(e))
                job.finish("FAILED", str(e))
            else:
                monitor.end(JobOutcome.COMPLETED)
                job.finish("COMPLETED", "")
                machine.add_machine_hours(estimated_seconds / 3600.0)
        except Exception as e:
            logger.exception("Remote job could not be started")
            job.finish("FAILED", str(e))

    def _broadcast(self, message: dict[str, Any]) -> None:
        """Sends a message to all clients; runs on the network loop."""
        if self._loop is None or self._loop.is_closed():
            return
        data = encode_message(message)
        for writer in list(self._connections):
            try:
                writer.write(data)
            except Exception:
                logger.debug("Dropping disconnected client", exc_info=True)
                self._connections.discard(writer)

    def _broadcast_threadsafe(self, message: dict[str, Any]) -> None:
        """Marshals a broadcast from the task loop to the network loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._broadcast, message)


class _RunningJob:
    """Tracks the lifecycle of a job accepted by the server."""

    def __init__(
        self,
        server: RemoteJobServer,
        machine_id: str,
        payload: JobPayload,
    ):
        self.id = str(uuid.uuid4())
        self.machine_id = machine_id
        self.payload = payload
        self.running = True
        self.outcome: str | None = None
        self.message = ""
        self.metrics: dict[str, Any] | None = None
        self.monitor: JobMonitor | None = None
        self.future: Any = None
        self._server = server

    def on_progress(self, sender, metrics: dict[str, Any]) -> None:
        """JobMonitor signal handler; runs on the task manager loop."""
        self.metrics = metrics
        self._server._broadcast_threadsafe(
            {"type": "job_progress", "job_id": self.id, "metrics": metrics}
        )

    def finish(self, outcome: str, message: str) -> None:
        """Marks the job as ended and notifies the clients."""
        self.running = False
        self.outcome = outcome
        self.message = message
        self._server._broadcast_threadsafe(
            {
                "type": "job_finished",
                "job_id": self.id,
                "outcome": outcome,
                "message": message,
            }
        )

    def state(self) -> dict[str, Any]:
        """Returns a JSON-safe snapshot for the ``get_job`` reply."""
        return {
            "job_id": self.id,
            "machine_id": self.machine_id,
            "doc_name": self.payload.doc_name,
            "kind": self.payload.kind.name,
            "running": self.running,
            "outcome": self.outcome,
            "message": self.message,
            "metrics": self.metrics,
        }

"""
Client side of the remote split.

A :class:`RemoteClient` owns the single TCP connection to a
rayforge-server: it issues correlated requests, and routes the
server's event stream to the machine proxies and job listeners
registered with it.

The bootstrap helpers build the client machine manager:
:class:`ClientMachineController` substitutes the machine's driver
with a :class:`~rayforge.machine.remote.driver_proxy.RemoteDriverProxy`
during driver rebuilds, so the whole application treats the remote
machine exactly like a local one. :class:`ClientMachineManager` does
not persist its machines: they are mirrors of the server's
configuration, refreshed on every start.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ...context import get_context
from ...shared.tasker import task_mgr
from ..models.controller import MachineController
from ..models.manager import MachineManager
from .driver_proxy import RemoteDriverProxy
from .protocol import PROTOCOL_VERSION, read_message, write_message

if TYPE_CHECKING:
    from .payload import JobPayload

logger = logging.getLogger(__name__)


class RemoteError(Exception):
    """The server rejected a request."""


class RemoteClient:
    """One client connection to a rayforge-server."""

    def __init__(self):
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = itertools.count(1)
        self._proxies: dict[str, RemoteDriverProxy] = {}
        self._job_queues: dict[str, asyncio.Queue] = {}

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self, host: str, port: int) -> dict[str, Any]:
        """Connects to the server and completes the handshake.

        Returns the server's hello result (version, machines).
        Raises :class:`OSError` when the server is unreachable and
        :class:`RemoteError` when it speaks another protocol version.
        """
        self._reader, self._writer = await asyncio.open_connection(host, port)
        self._read_task = asyncio.get_running_loop().create_task(
            self._read_loop()
        )
        try:
            result = await self.request(
                {"type": "hello", "version": PROTOCOL_VERSION}
            )
        except Exception:
            await self.close()
            raise
        logger.info(f"Connected to rayforge-server at {host}:{port}.")
        return result

    async def close(self) -> None:
        """Closes the connection; in-flight requests fail."""
        if self._read_task is not None:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._reader = None
        for future in self._pending.values():
            if not future.done():
                future.set_exception(
                    RemoteError("Connection to the server was closed.")
                )
        self._pending.clear()

    async def request(self, message: dict[str, Any]) -> Any:
        """Sends a request and awaits the server's result.

        Raises :class:`RemoteError` when the server reports an error.
        """
        if not self.connected:
            raise RemoteError("Not connected to the server.")
        request_id = next(self._next_id)
        message["id"] = request_id
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            writer = self._writer
            assert writer is not None
            await write_message(writer, message)
            reply = await future
        finally:
            self._pending.pop(request_id, None)
        if not reply.get("ok"):
            raise RemoteError(reply.get("error", "Request failed."))
        return reply.get("result")

    # -- High-level requests ------------------------------------------------

    async def get_machine_configs(self) -> dict[str, Any]:
        """Returns the server's machine configurations by id."""
        result = await self.request({"type": "get_machine_configs"})
        return result["configs"]

    async def machine_op(self, machine_id: str, op: str, **kwargs: Any) -> Any:
        """Forwards a driver-level operation to the server."""
        return await self.request(
            {
                "type": "machine_op",
                "machine_id": machine_id,
                "op": op,
                "kwargs": kwargs,
            }
        )

    async def submit_job(self, machine_id: str, payload: JobPayload) -> str:
        """Submits a job; returns the server's job id."""
        result = await self.request(
            {
                "type": "submit_job",
                "machine_id": machine_id,
                "payload": base64.b64encode(payload.to_bytes()).decode(
                    "ascii"
                ),
            }
        )
        return result["job_id"]

    # -- Event routing -------------------------------------------------------

    def register_proxy(
        self, machine_id: str, proxy: RemoteDriverProxy
    ) -> None:
        self._proxies[machine_id] = proxy

    def unregister_proxy(
        self, machine_id: str, proxy: RemoteDriverProxy
    ) -> None:
        if self._proxies.get(machine_id) is proxy:
            del self._proxies[machine_id]

    def job_events(self, job_id: str) -> asyncio.Queue:
        """Returns the event queue for a submitted job."""
        queue: asyncio.Queue = asyncio.Queue()
        self._job_queues[job_id] = queue
        return queue

    def remove_job_events(self, job_id: str) -> None:
        self._job_queues.pop(job_id, None)

    async def _read_loop(self) -> None:
        """Routes replies to pending requests and events to listeners."""
        assert self._reader is not None
        try:
            while True:
                message = await read_message(self._reader)
                mtype = message.get("type")
                if mtype == "reply":
                    reply_id = message.get("id")
                    future = self._pending.get(cast(int, reply_id))
                    if future is not None and not future.done():
                        future.set_result(message)
                    else:
                        logger.debug(f"Dropping unmatched reply: {message}")
                elif mtype == "driver_event":
                    machine_id = message.get("machine_id")
                    proxy = self._proxies.get(cast(str, machine_id))
                    if proxy is not None:
                        proxy._handle_event(
                            message.get("event", ""),
                            message.get("payload", {}),
                        )
                elif mtype in ("job_progress", "job_finished"):
                    job_id = message.get("job_id")
                    queue = self._job_queues.get(cast(str, job_id))
                    if queue is not None:
                        queue.put_nowait(message)
                else:
                    logger.debug(f"Dropping unknown message: {message}")
        except (asyncio.IncompleteReadError, ConnectionError):
            logger.info("Connection to rayforge-server lost.")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error reading from rayforge-server")
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(
                        RemoteError("Connection to the server was lost.")
                    )


class ClientMachineController(MachineController):
    """A controller whose driver is a proxy to the server.

    The machine's configured driver class is only used for
    introspection (encoding, validation); the actual instance is a
    :class:`RemoteDriverProxy` forwarding to the server. Driver
    prechecks are skipped: their arguments reference hardware
    resources that exist on the server, not on this computer.
    """

    skip_driver_precheck = True

    def __init__(self, machine, context, scheduler, client: RemoteClient):
        self._remote_client = client
        super().__init__(machine, context, scheduler)

    def _instantiate_driver(self, driver_cls):
        return RemoteDriverProxy(
            self.context,
            self.machine,
            self._remote_client,
            self.machine.id,
            self.machine.driver_name,
        )


class ClientMachineManager(MachineManager):
    """A machine manager mirroring the machines of a rayforge-server.

    Machines are populated from the server's configurations and
    never persisted locally; their drivers are proxies to the
    server's drivers.
    """

    def __init__(self, base_dir: Path, remote_client: RemoteClient):
        self.remote_client = remote_client
        super().__init__(base_dir, persist=False)

    def load(self):
        # Machines come from the server, not from disk.
        pass

    def get_controller(self, machine_id: str) -> MachineController:
        """Gets the controller, creating a client controller on demand."""
        if machine_id in self.controllers:
            return self.controllers[machine_id]

        machine = self.get_machine_by_id(machine_id)
        if not machine:
            raise ValueError(f"No machine found with ID {machine_id}")

        logger.debug(
            f"Creating client controller for machine '{machine.name}'."
        )
        controller = ClientMachineController(
            machine,
            get_context(),
            task_mgr.schedule_on_main_thread,
            self.remote_client,
        )
        machine._connect_controller_signals(controller)
        self.controllers[machine_id] = controller
        return controller

    def populate_from_configs(self, configs: dict[str, Any]) -> None:
        """Creates machine mirrors from the server's configurations."""
        from ..models.machine import Machine

        for config in configs.values():
            machine = Machine.from_dict(config)
            self.add_machine(machine)
            logger.info(
                f"Mirrored remote machine '{machine.name}' (id: {machine.id})"
            )

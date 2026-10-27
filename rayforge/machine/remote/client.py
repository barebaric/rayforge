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
    from ..models.machine import Machine
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
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The event loop the connection runs on."""
        return self._loop

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self, address: str | tuple[str, int]) -> dict[str, Any]:
        """Connects to the server and completes the handshake.

        The address is a Unix domain socket path or a ``(host,
        port)`` tuple. Returns the server's hello result (version,
        machines). Raises :class:`OSError` when the server is
        unreachable and :class:`RemoteError` when it speaks another
        protocol version.
        """
        if isinstance(address, str):
            self._reader, self._writer = await asyncio.open_unix_connection(
                path=address
            )
            where = address
        else:
            host, port = address
            self._reader, self._writer = await asyncio.open_connection(
                host, port
            )
            where = f"{host}:{port}"
        self._loop = asyncio.get_running_loop()
        self._read_task = self._loop.create_task(self._read_loop())
        try:
            result = await self.request(
                {"type": "hello", "version": PROTOCOL_VERSION}
            )
        except Exception:
            await self.close()
            raise
        logger.info(f"Connected to rayforge-server at {where}.")
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

    async def manager_op(self, op: str, **kwargs: Any) -> Any:
        """Sends a machine-configuration change to the server."""
        return await self.request(
            {"type": "manager_op", "op": op, "kwargs": kwargs}
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
    server's drivers. The server owns the configuration, so machine
    changes made here are forwarded to it: creations and removals
    immediately, edits debounced.
    """

    #: Seconds to wait after the last change before syncing an
    #: edited machine configuration to the server.
    SYNC_DEBOUNCE_SECONDS = 1.0

    def __init__(self, base_dir: Path, remote_client: RemoteClient):
        self.remote_client = remote_client
        # Set before super().__init__(): the base constructor calls
        # load(), which is overridden to no-op.
        self._populating = False
        self._sync_timers: dict[str, Any] = {}
        super().__init__(base_dir, persist=False)

    def load(self):
        # Machines come from the server, not from disk.
        pass

    def add_machine(self, machine: Machine):
        super().add_machine(machine)
        if self._populating:
            return
        self._send_config_change(
            "create_machine", config=self._config_of(machine)
        )

    def remove_machine(self, machine_id: str):
        existed = machine_id in self.machines
        super().remove_machine(machine_id)
        self._sync_timers.pop(machine_id, None)
        if self._populating or not existed:
            return
        self._send_config_change("remove_machine", machine_id=machine_id)

    def on_machine_changed(self, machine: Machine, **kwargs: Any):
        super().on_machine_changed(machine, **kwargs)
        self._schedule_update_sync(machine)

    def _config_of(self, machine: Machine) -> dict[str, Any]:
        """Serializes a mirror machine for the server."""
        config = machine.to_dict(include_frozen_dialect=False)
        config.setdefault("machine", {})["id"] = machine.id
        return config

    def _schedule_update_sync(self, machine: Machine) -> None:
        """Debounces machine edits into one update per machine.

        Called from arbitrary threads (e.g. the GTK main loop);
        marshals the timer onto the connection's event loop.
        """
        if self._populating:
            return
        loop = self.remote_client.loop
        if loop is None or not loop.is_running():
            return
        loop.call_soon_threadsafe(self._arm_update_sync, machine.id)

    def _arm_update_sync(self, machine_id: str) -> None:
        """Arms the debounce timer; runs on the connection's loop."""
        loop = self.remote_client.loop
        assert loop is not None
        timer = self._sync_timers.pop(machine_id, None)
        if timer is not None:
            timer.cancel()
        self._sync_timers[machine_id] = loop.call_later(
            self.SYNC_DEBOUNCE_SECONDS,
            lambda: self._send_update_sync(machine_id),
        )

    def _send_update_sync(self, machine_id: str) -> None:
        """Sends the machine's final state to the server."""
        self._sync_timers.pop(machine_id, None)
        current = self.get_machine_by_id(machine_id)
        if current is None:
            return
        self._send_config_change(
            "update_machine",
            machine_id=machine_id,
            config=self._config_of(current),
        )

    def _send_config_change(self, op: str, **kwargs: Any) -> None:
        """Forwards a configuration change on the connection's loop.

        Failures are logged: configuration sync is best-effort and
        must never disturb the UI that triggered it.
        """
        loop = self.remote_client.loop
        if loop is None or not loop.is_running():
            logger.warning(f"Cannot sync '{op}' to the server: not connected.")
            return

        async def _send() -> None:
            try:
                await self.remote_client.manager_op(op, **kwargs)
            except Exception:
                logger.exception(
                    "Syncing a machine change to the server failed"
                )

        loop.call_soon_threadsafe(lambda: loop.create_task(_send()))

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

        self._populating = True
        try:
            for config in configs.values():
                machine = Machine.from_dict(config)
                self.add_machine(machine)
                logger.info(
                    f"Mirrored remote machine '{machine.name}' "
                    f"(id: {machine.id})"
                )
        finally:
            self._populating = False

"""
Unified application bootstrap: one path, local or remote.

Every Rayforge frontend drives its machines through the remote
protocol — there is no separate "directly attached" code path. When
no ``--client`` address is given, the backend runs inside the same
process, owning the real machine configuration and drivers, and the
frontend connects to it over a Unix domain socket (a loopback TCP
connection on platforms without Unix sockets). With ``--client``, the
same frontend machinery talks to a backend on another machine.

The backend owns the machine configuration: machines created or
edited on the frontend are synced to it (see
:meth:`~rayforge.machine.remote.client.ClientMachineManager`), and it
persists them to the application's configuration directory.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
import tempfile
from typing import Any, NamedTuple

from ...context import RayforgeContext
from ...shared.tasker import task_mgr
from .client import ClientMachineManager, RemoteClient
from .server import RemoteJobServer

logger = logging.getLogger(__name__)

# The setup of a locally started backend, kept for shutdown.
_local_setup: LocalSetup | None = None


class LocalSetup(NamedTuple):
    """The pieces of an in-process backend plus its frontend client."""

    server: RemoteJobServer
    client: RemoteClient
    backend_context: RayforgeContext
    frontend_context: RayforgeContext


def supports_unix_sockets() -> bool:
    """Whether Unix domain sockets are usable on this platform."""
    return hasattr(socket, "AF_UNIX") and sys.platform != "win32"


def local_socket_path() -> str:
    """A per-process socket path for the in-process backend."""
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return os.path.join(runtime_dir, f"rayforge-backend-{os.getpid()}.sock")


async def start_local_backend(
    context: RayforgeContext,
) -> tuple[RemoteJobServer, str | tuple[str, int]]:
    """Starts the in-process backend server on the task loop."""
    if supports_unix_sockets():
        server = RemoteJobServer(
            context, task_mgr, socket_path=local_socket_path()
        )
    else:
        server = RemoteJobServer(context, task_mgr, host="127.0.0.1")
    await server.start()
    return server, server.address


def _run_on_task_loop(coro: Any, timeout: float = 30.0) -> Any:
    """Runs a coroutine on the task loop, blocking the caller."""
    future = asyncio.run_coroutine_threadsafe(coro, task_mgr.loop)
    return future.result(timeout=timeout)


def bootstrap_local_app() -> LocalSetup:
    """Starts the in-process backend and installs the frontend context.

    Creates two contexts in this process: the backend context owns the
    machine configuration and drivers (it is what ``rayforge
    --server`` runs on its own), the frontend context serves the UI.
    The process-wide context singleton ends up as the frontend
    context, with a :class:`ClientMachineManager` mirroring the
    backend's machines.
    """
    from ... import config
    from ... import context as context_module
    from ...machine.models.dialect_manager import DialectManager

    global _local_setup
    if _local_setup is not None:
        raise RuntimeError("The local backend is already running.")

    backend_context = RayforgeContext()
    previous = context_module._context_instance
    context_module._context_instance = backend_context
    try:
        backend_context.initialize_lite_context(config.MACHINE_DIR)
        backend_context._dialect_mgr = DialectManager(config.DIALECT_DIR)
    finally:
        # The frontend reuses the context created early in main(), so
        # anything a --script registered on it survives; if none
        # exists yet, create one.
        frontend_context = previous or RayforgeContext()
        context_module._context_instance = frontend_context
        if frontend_context._dialect_mgr is None:
            frontend_context._dialect_mgr = DialectManager(config.DIALECT_DIR)

    server, address = _run_on_task_loop(start_local_backend(backend_context))

    async def _connect() -> Any:
        client = RemoteClient()
        await client.connect(address)
        return client

    client = _run_on_task_loop(_connect())
    configs = _run_on_task_loop(client.get_machine_configs())

    manager = ClientMachineManager(config.MACHINE_DIR, client)
    frontend_context.install_machine_manager(manager)
    manager.populate_from_configs(configs)

    _local_setup = LocalSetup(
        server=server,
        client=client,
        backend_context=backend_context,
        frontend_context=frontend_context,
    )
    logger.info(
        f"Unified local backend started: {len(configs)} machine(s), "
        f"frontend connected via {address!r}."
    )
    return _local_setup


async def shutdown_local_backend() -> None:
    """Stops the in-process backend, if this process runs one."""
    global _local_setup
    setup = _local_setup
    _local_setup = None
    if setup is None:
        return
    logger.info("Shutting down the local backend.")
    await setup.client.close()
    await setup.server.stop()
    await setup.backend_context.machine_mgr.shutdown()

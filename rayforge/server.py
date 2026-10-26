"""
Entry point for the headless remote job server.

Runs on the machine that is physically attached to the laser
cutter. It loads the regular machine configuration, connects the
configured drivers, and serves job submission and progress over TCP
(see :mod:`rayforge.machine.remote`). No GTK is imported.

Usage::

    rayforge-server [--host HOST] [--port PORT]

Bind to a LAN address only on trusted networks; the server has no
authentication. For remote access over untrusted networks, tunnel it
(e.g. via SSH) instead of exposing the port.
"""

import argparse
import asyncio
import logging
import signal
from typing import cast

from . import config
from .context import get_context
from .machine.remote.server import RemoteJobServer
from .shared import tasker
from .shared.tasker.manager import TaskManagerProxy
from .worker_init import initialize_worker

logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="rayforge-server",
        description="Headless Rayforge job server.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Address to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2517,
        help="Port to bind to (default: 2517)",
    )
    return parser.parse_args(argv)


async def _run(args) -> None:
    context = get_context()
    context.initialize_lite_context(config.MACHINE_DIR)
    task_mgr_proxy = cast(TaskManagerProxy, tasker.task_mgr)
    task_mgr_proxy.initialize()
    task_mgr = tasker.task_mgr

    machine_mgr = context.machine_mgr
    if not machine_mgr.machines:
        logger.warning(
            "No machines configured. Use the Rayforge GUI to set up "
            "the machine this server should drive."
        )
    else:
        machine_mgr.initialize_connections()

    server = RemoteJobServer(context, task_mgr, host=args.host, port=args.port)
    await server.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)
    logger.info(
        f"rayforge-server ready: {len(machine_mgr.machines)} machine(s), "
        f"listening on {args.host}:{server.port}"
    )
    await stop_event.wait()
    logger.info("Shutting down.")
    await server.stop()
    await machine_mgr.shutdown()


def main(argv=None):
    initialize_worker()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

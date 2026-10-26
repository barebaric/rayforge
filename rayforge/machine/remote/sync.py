"""
Syncing machine configurations from a remote rayforge-server.

The client must encode jobs exactly the way the server's driver
expects. Instead of manually replicating a machine profile on the
laptop, the server publishes its machine configuration (see the
``get_machine_config`` protocol message) and the client imports it as
a local mirror driven by :class:`RemoteDriver`.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import uuid
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from .driver import DEFAULT_PORT, RemoteDriver
from .protocol import PROTOCOL_VERSION, read_message, write_message

if TYPE_CHECKING:
    from ..models.machine import Machine
    from ..models.manager import MachineManager

logger = logging.getLogger(__name__)


async def _connect_and_hello(
    host: str, port: int
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, dict[str, Any]]:
    """Connects to a server and completes the version handshake.

    Raises :class:`OSError` when the server cannot be reached and
    :class:`ValueError` when it speaks a different protocol version.
    """
    reader, writer = await asyncio.open_connection(host, port)
    try:
        await write_message(
            writer, {"type": "hello", "version": PROTOCOL_VERSION}
        )
        welcome = await read_message(reader)
        if welcome.get("type") != "welcome":
            raise ValueError(welcome.get("message", _("Handshake failed.")))
    except Exception:
        writer.close()
        raise
    return reader, writer, welcome


async def fetch_server_machines(
    host: str, port: int = DEFAULT_PORT
) -> list[dict[str, Any]]:
    """Returns the machines hosted by a remote server.

    Each entry carries the machine's ``id``, ``name``, ``driver_name``
    and ``connected`` state.
    """
    _reader, writer, welcome = await _connect_and_hello(host, port)
    writer.close()
    return welcome.get("machines", [])


async def fetch_machine_config(
    host: str, port: int = DEFAULT_PORT, machine_id: str = ""
) -> dict[str, Any]:
    """Fetches one machine configuration from a remote server.

    Raises :class:`OSError` when the server cannot be reached and
    :class:`ValueError` when the server rejects the request.
    """
    reader, writer, _welcome = await _connect_and_hello(host, port)
    try:
        await write_message(
            writer,
            {"type": "get_machine_config", "machine_id": machine_id},
        )
        reply = await read_message(reader)
        if reply.get("type") != "machine_config":
            raise ValueError(
                reply.get("message", _("Server did not send a machine."))
            )
        return reply["config"]
    finally:
        writer.close()


def import_remote_machine(
    machine_mgr: MachineManager,
    config: dict[str, Any],
    host: str,
    port: int = DEFAULT_PORT,
    machine_id: str = "",
) -> Machine:
    """Creates a local machine mirroring a server machine.

    The imported machine carries the server machine's entire profile
    (kinematics, heads, dialect, speeds) but is driven by
    :class:`RemoteDriver` pointing at the server. It gets a fresh
    local id: the id of the machine on the server travels in the
    driver arguments instead.
    """
    from ..models.machine import Machine

    data = copy.deepcopy(config)
    machine_data = data.setdefault("machine", {})
    remote_id = machine_data.get("id") or machine_id
    machine_data["driver"] = RemoteDriver.__name__
    machine_data["driver_args"] = {
        "host": host,
        "port": port,
        "machine_id": remote_id,
    }
    machine = Machine.from_dict(data)
    machine.id = str(uuid.uuid4())
    machine_mgr.add_machine(machine)
    logger.info(
        f"Imported remote machine '{machine.name}' (remote id: {remote_id})"
    )
    return machine

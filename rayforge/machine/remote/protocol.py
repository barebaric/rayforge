"""
Message framing for the remote protocol.

The client and the server exchange length-prefixed JSON messages over
a TCP stream. Both sides run the same application version, so the
protocol only carries a version for a friendly mismatch error, not
for backward compatibility.

Requests carry a client-chosen ``id``; the server answers with a
``reply`` echoing the id, so several requests from the same
connection can be in flight at once. Messages that are not replies
are events: the server pushes them to every connected client at any
time, tagged with the ``machine_id`` or ``job_id`` they belong to.

Client → server (all carry a client-chosen ``id``):

    hello              ``{"id": n, "type": "hello", "version": int}``
    get_machine_configs
                       ``{"id": n, "type": "get_machine_configs"}``
    submit_job         ``{"id": n, "type": "submit_job",
                        "machine_id": str, "payload": <base64>}``
    machine_op        ``{"id": n, "type": "machine_op",
                        "machine_id": str, "op": str, "kwargs": {...}}``
    get_job           ``{"id": n, "type": "get_job"}``

Server → client:

    reply             ``{"type": "reply", "id": n, "ok": bool,
                        "result": ... | "error": str}`` — the result
                        of a request; ``hello`` yields the machine
                        list, ``get_machine_configs`` a mapping of
                        machine ids to configuration dictionaries.
    driver_event      ``{"machine_id": str, "event": str,
                        "payload": {...}}`` — a re-emitted driver
                       signal, e.g. state_changed with a serialized
                       DeviceState.
    job_progress      ``{"job_id": str, "op_index": int,
                        "metrics": {...}}``
    job_finished      ``{"job_id": str, "outcome": str,
                        "message": str}``

The ``machine_id`` arguments are optional when the server hosts a
single machine; the server then uses that one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import Any

logger = logging.getLogger(__name__)


PROTOCOL_VERSION = 2

#: The default port the server and the clients use.
DEFAULT_PORT = 2517

#: Messages larger than this are rejected before being buffered.
MAX_MESSAGE_BYTES = 512 * 1024 * 1024


def encode_message(message: dict[str, Any]) -> bytes:
    """Frames a message as a length-prefixed JSON body."""
    body = json.dumps(message).encode("utf-8")
    return struct.pack("!I", len(body)) + body


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Reads one framed message from the stream.

    Raises :class:`asyncio.IncompleteReadError` when the peer
    disconnects mid-message, and :class:`ConnectionError` on an
    oversized message.
    """
    header = await reader.readexactly(4)
    (length,) = struct.unpack("!I", header)
    if length > MAX_MESSAGE_BYTES:
        raise ConnectionError(f"Received oversized message ({length} bytes).")
    body = await reader.readexactly(length)
    return json.loads(body.decode("utf-8"))


async def write_message(
    writer: asyncio.StreamWriter, message: dict[str, Any]
) -> None:
    """Writes one framed message to the stream."""
    writer.write(encode_message(message))
    await writer.drain()

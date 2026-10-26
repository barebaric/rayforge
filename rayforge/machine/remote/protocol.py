"""
Message framing for the remote job server protocol.

The client and the server exchange length-prefixed JSON messages over
a TCP stream. Both sides run the same application version, so the
protocol only carries a version for a friendly mismatch error, not
for backward compatibility.

Message types
-------------

Client → server:

    hello             ``{"type": "hello", "version": int}``
    get_machine_config
                       ``{"type": "get_machine_config",
                        "machine_id": str (optional)}``
    submit_job        ``{"type": "submit_job", "machine_id": str,
                        "payload": <base64 JobPayload>}``
    run_raw           ``{"type": "run_raw", "machine_id": str,
                        "code": str}``
    cancel_job         ``{"type": "cancel_job", "job_id": str}``
    set_hold          ``{"type": "set_hold", "job_id": str,
                        "hold": bool}``

Server → client:

    welcome          Sent in response to ``hello``; lists machines.
    machine_config    Sent in response to ``get_machine_config``.
    job_accepted     ``{"type": "job_accepted", "job_id": str}``
    job_progress     ``{"type": "job_progress", "job_id": str,
                        "op_index": int, "metrics": {...}}``
    job_finished     ``{"type": "job_finished", "job_id": str,
                        "outcome": str, "message": str}``
    ok / error       Replies for request/response style messages.

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


PROTOCOL_VERSION = 1

#: Messages larger than this are rejected before being buffered.
MAX_MESSAGE_BYTES = 512 * 1024 * 1024


def encode_message(message: dict[str, Any]) -> bytes:
    """Frames a message as a length-prefixed JSON body."""
    body = json.dumps(message).encode("utf-8")
    return struct.pack("!I", len(body)) + body


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Reads one framed message from the stream.

    Raises :class:`asyncio.IncompleteReadError` when the peer
    disconnects mid-message, and :class:`ConnectionError` on a clean
    EOF between messages.
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

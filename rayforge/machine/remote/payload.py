"""
Binary serialization of fully computed jobs.

A :class:`JobPayload` captures everything a machine driver needs to
execute a job — the encoded machine code, the op↔line map, per-command
time estimates, and job metadata — in a self-contained byte string.
It is the unit of work that the computing side of a client/server
split hands to the executing side: both run the same version, so the
format only needs a version check, not backward compatibility.

Wire format (all integers big-endian):

    ``"RFPY"`` | u32 format version | u32 header length |
    header JSON (utf-8) | zlib(blob section)

The blob section concatenates, in the order recorded by the header:

    text (utf-8) | op-map spans (i32 pairs) | op-map line→op (i32)
    | command times (little-endian f64) | ops dict (JSON, utf-8)
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from raygeo.ops import Ops

from ...pipeline.encoder.base import EncodedOutput, MachineCodeOpMap
from ..job_status import JobKind

if TYPE_CHECKING:
    from ...machine.models.machine import Machine
    from ...pipeline.artifact.job import JobArtifact


_MAGIC = b"RFPY"
_FORMAT_VERSION = 1
_PREFIX = struct.Struct("!4sII")


@dataclass
class JobPayload:
    """A fully computed job, serializable for execution elsewhere.

    Attributes:
        encoded: The encoder output (machine code, op↔line map, and
            driver-specific data).
        ops: The command stream the encoded output was built from, in
            the same space the encoder consumed it in. The executing
            side uses it for time estimates and progress tracking.
        doc_name: Name of the document the job was computed from, for
            display and logging on the executing side.
        kind: Whether the job is a send or a frame job.
        distance_mm: Total travel distance of the job.
        estimated_seconds: Total time estimate, if known.
        command_times: Per-command time estimates in seconds, in op
            order. Drivers use these to pace and report progress
            without access to the original :class:`Ops` stream.
    """

    encoded: EncodedOutput
    ops: Ops | None = None
    doc_name: str = ""
    kind: JobKind = JobKind.SEND
    distance_mm: float = 0.0
    estimated_seconds: float | None = None
    command_times: list[float] | None = None

    @property
    def op_count(self) -> int:
        """The number of ops the encoded output was built from."""
        return self.encoded.op_map.op_count

    @classmethod
    def from_job_artifact(
        cls,
        artifact: JobArtifact,
        doc_name: str = "",
        machine: Machine | None = None,
        kind: JobKind = JobKind.SEND,
    ) -> JobPayload:
        """Builds a payload from a pipeline job artifact.

        When *machine* is given, per-command time estimates are
        computed with the machine's motion parameters, mirroring what
        the GRBL driver derives from the ``Ops`` stream at run time.
        """
        encoded = artifact.encoded_output
        if encoded is None:
            raise ValueError("Job artifact has no encoded output.")

        command_times = None
        if machine is not None:
            command_times = [
                float(t)
                for t in artifact.ops.estimate_command_times(
                    default_feed_rate=machine.max_cut_speed,
                    default_rapid_rate=machine.max_travel_speed,
                    acceleration=machine.acceleration,
                )
            ]

        return cls(
            encoded=encoded,
            ops=artifact.ops,
            doc_name=doc_name,
            kind=kind,
            distance_mm=artifact.distance,
            estimated_seconds=artifact.time_estimate,
            command_times=command_times,
        )

    def to_bytes(self) -> bytes:
        """Serializes the payload to a self-contained byte string."""
        text_bytes = self.encoded.text.encode("utf-8")
        spans = bytes(self.encoded.op_map.op_to_machine_code_bytes)
        line_to_op = bytes(self.encoded.op_map.machine_code_to_op_bytes)
        if self.command_times is None:
            times = b""
        else:
            times = np.asarray(self.command_times, dtype="<f8").tobytes()
        if self.ops is None:
            ops_bytes = b""
        else:
            ops_bytes = json.dumps(
                _tag_driver_data(self.ops.to_dict())
            ).encode("utf-8")

        header = {
            "format_version": _FORMAT_VERSION,
            "doc_name": self.doc_name,
            "kind": self.kind.name,
            "distance_mm": self.distance_mm,
            "estimated_seconds": self.estimated_seconds,
            "text_len": len(text_bytes),
            "spans_len": len(spans),
            "line_to_op_len": len(line_to_op),
            "command_times_len": len(times),
            "ops_len": len(ops_bytes),
            "driver_data": _tag_driver_data(self.encoded.driver_data),
        }
        header_bytes = json.dumps(header).encode("utf-8")
        return (
            _PREFIX.pack(_MAGIC, _FORMAT_VERSION, len(header_bytes))
            + header_bytes
            + zlib.compress(
                text_bytes + spans + line_to_op + times + ops_bytes
            )
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> JobPayload:
        """Reconstructs a payload serialized by :meth:`to_bytes`."""
        if len(data) < _PREFIX.size:
            raise ValueError("Job payload is truncated.")
        magic, version, header_len = _PREFIX.unpack_from(data)
        if magic != _MAGIC:
            raise ValueError("Not a RayForge job payload.")
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported job payload version {version} "
                f"(expected {_FORMAT_VERSION})."
            )

        header_end = _PREFIX.size + header_len
        if header_end > len(data):
            raise ValueError("Job payload header is truncated.")
        try:
            header = json.loads(
                data[_PREFIX.size : header_end].decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Corrupt job payload header: {e}") from e

        lengths = [
            header["text_len"],
            header["spans_len"],
            header["line_to_op_len"],
            header["command_times_len"],
            header.get("ops_len", 0),
        ]
        if min(lengths) < 0:
            raise ValueError("Job payload header has negative blob lengths.")
        try:
            blobs = zlib.decompress(data[header_end:])
        except zlib.error as e:
            raise ValueError(f"Corrupt job payload data: {e}") from e
        if len(blobs) != sum(lengths):
            raise ValueError("Job payload blob section has unexpected length.")

        pos = 0

        def _take(length: int) -> bytes:
            nonlocal pos
            chunk = blobs[pos : pos + length]
            pos += length
            return chunk

        text = _take(lengths[0]).decode("utf-8")
        spans = bytearray(_take(lengths[1]))
        line_to_op = bytearray(_take(lengths[2]))
        times_len = lengths[3]
        command_times = None
        if times_len:
            if times_len % 8 != 0:
                raise ValueError(
                    "Job payload command times have bad alignment."
                )
            command_times = np.frombuffer(
                _take(times_len), dtype="<f8"
            ).tolist()

        ops = None
        ops_len = lengths[4]
        if ops_len:
            try:
                ops_dict = json.loads(_take(ops_len).decode("utf-8"))
                ops = Ops.from_dict(_untag_driver_data(ops_dict))
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ) as e:
                raise ValueError(f"Corrupt job payload ops: {e}") from e

        encoded = EncodedOutput(
            text=text,
            op_map=MachineCodeOpMap(
                op_to_machine_code=spans,
                machine_code_to_op=line_to_op,
            ),
            driver_data=_untag_driver_data(header["driver_data"]),
        )
        return cls(
            encoded=encoded,
            ops=ops,
            doc_name=header["doc_name"],
            kind=JobKind[header["kind"]],
            distance_mm=header["distance_mm"],
            estimated_seconds=header["estimated_seconds"],
            command_times=command_times,
        )


def _tag_driver_data(value: Any) -> Any:
    """Converts a driver_data value into JSON-safe data.

    Bytes and numpy arrays are wrapped in tagged dicts. Values that
    JSON cannot represent raise :class:`TypeError` so that no payload
    is silently lossy. Dict keys are stringified on the way through.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "__type__": "bytes",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": base64.b64encode(value.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return _tag_driver_data(value.item())
    if isinstance(value, dict):
        return {
            str(key): _tag_driver_data(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_tag_driver_data(item) for item in value]
    if value is None or isinstance(value, (str, int, float)):
        return value
    raise TypeError(
        f"Cannot serialize driver_data value of type {type(value).__name__}."
    )


def _untag_driver_data(value: Any) -> Any:
    """Inverse of :func:`_tag_driver_data`."""
    if isinstance(value, list):
        return [_untag_driver_data(item) for item in value]
    if isinstance(value, dict):
        tag = value.get("__type__")
        if tag == "bytes":
            return base64.b64decode(value["data"])
        if tag == "ndarray":
            raw = base64.b64decode(value["data"])
            array = np.frombuffer(raw, dtype=np.dtype(value["dtype"]))
            shape = value["shape"]
            if shape:
                array = array.reshape(shape)
            return array
        return {key: _untag_driver_data(item) for key, item in value.items()}
    return value

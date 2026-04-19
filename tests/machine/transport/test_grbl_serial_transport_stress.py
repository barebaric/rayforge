"""
Stress tests for GrblSerialTransport buffer tracking and ack interleaving.

Exercises random serial packet boundaries — data arrives in arbitrary-
sized chunks, acks can be split mid-byte across reads, status reports
can be interleaved at any position.  The transport must keep buffer
accounting correct regardless of fragmentation.

Run with:  pixi run test -m "stress"
"""

import random
import pytest
from unittest.mock import MagicMock, AsyncMock

from rayforge.machine.transport.grbl import (
    GrblSerialTransport,
    GRBL_RX_BUFFER_SIZE,
)
from rayforge.machine.transport import SerialTransport


pytestmark = pytest.mark.stress


def _make_mock_transport():
    mock = MagicMock(spec=SerialTransport)
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.send = AsyncMock()
    mock.received = MagicMock()
    mock.status_changed = MagicMock()
    mock.is_connected = True
    mock.port = "/dev/ttyUSB0"
    return mock


def _make_grbl_transport():
    return GrblSerialTransport(_make_mock_transport())


def _fragment(data: bytes, rng: random.Random) -> list[bytes]:
    """Split *data* into random-sized chunks (1–max_chunk bytes)."""
    if not data:
        return []
    chunks = []
    i = 0
    while i < len(data):
        chunk_len = rng.randint(1, min(12, len(data) - i))
        chunks.append(data[i : i + chunk_len])
        i += chunk_len
    return chunks


STATUS_REPORTS = [
    b"<Idle|MPos:0.000,0.000,0.000|FS:0,0>\r\n",
    b"<Run|MPos:1.234,5.678,0.000|FS:500,0>\r\n",
    b"<Jog|MPos:10.0,20.0,0.0|FS:1000,0>\r\n",
    b"<Home|MPos:0,0,0|FS:0,0>\r\n",
]


def _build_response_stream(
    num_acks: int,
    error_positions: set[int],
    rng: random.Random,
) -> list[tuple[bytes, str]]:
    """
    Build a mixed response stream as (raw_bytes, kind) tuples.

    *kind* is one of ``'ok'``, ``'error'``, ``'status'`` so the test
    can track how many pending commands are freed.
    """
    parts: list[tuple[bytes, str]] = []
    for i in range(num_acks):
        if i in error_positions:
            code = rng.choice([1, 5, 10, 20, 33])
            parts.append((f"error:{code}\r\n".encode(), "error"))
        else:
            parts.append((b"ok\r\n", "ok"))
        if rng.random() < 0.35:
            parts.append((rng.choice(STATUS_REPORTS), "status"))
    return parts


# ------------------------------------------------------------------ #
#  Transport-layer tests: buffer tracking under random fragmentation #
# ------------------------------------------------------------------ #


class TestBufferTrackingFuzz:
    """
    Fuzz the transport's ``parse_incoming`` with random packet
    boundaries and verify buffer accounting stays exact.
    """

    @pytest.mark.asyncio
    async def test_single_byte_delivery_500_commands(self):
        transport = _make_grbl_transport()

        for i in range(500):
            line = f"G1 X{i} Y{i}\n".encode()
            await transport.send_gcode(line, op_index=i)

            ack_stream = b"ok\r\n"
            for byte in ack_stream:
                transport.parse_incoming(bytes([byte]))

            assert transport.buffer_count == 0, f"iter {i}"

    @pytest.mark.asyncio
    async def test_random_chunk_size_2000_commands(self):
        transport = _make_grbl_transport()
        rng = random.Random(99)

        for i in range(2000):
            line = f"G1 X{i % 999}\n".encode()
            await transport.send_gcode(line, op_index=i)

            chunks = _fragment(b"ok\r\n", rng)
            for chunk in chunks:
                transport.parse_incoming(chunk)

            assert transport.buffer_count == 0, f"iter {i}"

    @pytest.mark.asyncio
    async def test_fill_near_limit_then_fragmented_drain(self):
        transport = _make_grbl_transport()
        rng = random.Random(7)

        for cycle in range(200):
            sent: list[int] = []
            total = 0
            while total < GRBL_RX_BUFFER_SIZE - 2:
                line_len = rng.randint(5, 20)
                line = b"G0 " + b"A" * (line_len - 5) + b"\n"
                await transport.send_gcode(line, op_index=None)
                sent.append(len(line))
                total += len(line)

            assert transport.buffer_count == total

            ack_data = b"ok\r\n" * len(sent)
            chunks = _fragment(ack_data, rng)
            for chunk in chunks:
                transport.parse_incoming(chunk)

            assert transport.buffer_count == 0, f"cycle {cycle}"
            assert transport.pending_queue.empty()

    @pytest.mark.asyncio
    async def test_partial_ack_delivery_with_buffer_pressure(self):
        transport = _make_grbl_transport()
        rng = random.Random(2024)

        all_sent: list[int] = []
        total = 0

        for i in range(500):
            line = f"G1 X{i} Y{i}\n".encode()
            if total + len(line) > GRBL_RX_BUFFER_SIZE:
                num_acks = rng.randint(1, 4)
                num_acks = min(num_acks, len(all_sent))
                ack_data = b"ok\r\n" * num_acks
                for chunk in _fragment(ack_data, rng):
                    transport.parse_incoming(chunk)
                for _ in range(num_acks):
                    total -= all_sent.pop(0)

            await transport.send_gcode(line, op_index=i)
            all_sent.append(len(line))
            total += len(line)
            assert transport.buffer_count == total

        remainder = b"ok\r\n" * len(all_sent)
        for chunk in _fragment(remainder, rng):
            transport.parse_incoming(chunk)

        assert transport.buffer_count == 0
        assert transport.pending_queue.empty()

    @pytest.mark.asyncio
    async def test_many_fill_drain_cycles_random_sizes(self):
        transport = _make_grbl_transport()
        rng = random.Random(31415)

        for cycle in range(1000):
            num_cmds = rng.randint(1, 25)
            sent: list[int] = []
            total = 0

            for j in range(num_cmds):
                line_len = rng.randint(5, 40)
                line = b"G0 " + b"X" * (line_len - 5) + b"\n"
                if total + len(line) > GRBL_RX_BUFFER_SIZE:
                    ack_data = b"ok\r\n" * len(sent)
                    for chunk in _fragment(ack_data, rng):
                        transport.parse_incoming(chunk)
                    total = 0
                    sent.clear()

                await transport.send_gcode(line, op_index=None)
                sent.append(len(line))
                total += len(line)

            ack_data = b"ok\r\n" * len(sent)
            for chunk in _fragment(ack_data, rng):
                transport.parse_incoming(chunk)

            assert transport.buffer_count == 0, f"cycle {cycle}"
            assert transport.pending_queue.empty()

    @pytest.mark.asyncio
    async def test_reset_at_random_points(self):
        transport = _make_grbl_transport()
        rng = random.Random(2718)

        for i in range(1500):
            line = f"G1 X{i}\n".encode()
            await transport.send_gcode(line, op_index=i)

            if rng.random() < 0.1:
                transport.reset()
                assert transport.buffer_count == 0
                assert transport.pending_queue.empty()
            else:
                ack_data = b"ok\r\n"
                for chunk in _fragment(ack_data, rng):
                    transport.parse_incoming(chunk)
                assert transport.buffer_count == 0

    @pytest.mark.asyncio
    async def test_spurious_acks_dont_corrupt_buffer(self):
        transport = _make_grbl_transport()
        rng = random.Random(1618)

        for i in range(1300):
            extra_acks = b"ok\r\n" * rng.randint(0, 3)
            for chunk in _fragment(extra_acks, rng):
                transport.parse_incoming(chunk)

            line = f"G1 X{i}\n".encode()
            await transport.send_gcode(line, op_index=i)

            ack = b"ok\r\n"
            for chunk in _fragment(ack, rng):
                transport.parse_incoming(chunk)
            assert transport.buffer_count == 0

    @pytest.mark.asyncio
    async def test_needs_space_invariant_under_fragmented_acks(self):
        transport = _make_grbl_transport()
        rng = random.Random(42)

        for i in range(500):
            line = f"G0 X{i % 100}\n".encode()
            assert not transport.needs_space(len(line))

            await transport.send_gcode(line, op_index=i)

            overflow = GRBL_RX_BUFFER_SIZE - transport.buffer_count + 1
            assert transport.needs_space(overflow)

            for chunk in _fragment(b"ok\r\n", rng):
                transport.parse_incoming(chunk)
            assert transport.buffer_count == 0


# ------------------------------------------------------------------ #
#  Transport-layer tests: interleaved acks + random fragmentation   #
# ------------------------------------------------------------------ #


class TestAckInterleavingFuzz:
    """
    Verify that acks embedded within or between status reports,
    errors, and alarms are extracted correctly even when the
    serial stream is fragmented at random byte boundaries.
    """

    @pytest.mark.asyncio
    async def test_interleaved_ok_status_random_fragmentation(self):
        transport = _make_grbl_transport()
        rng = random.Random(1001)

        for i in range(10000):
            line = f"G1 X{i % 100}\n".encode()
            await transport.send_gcode(line, op_index=i)

            stream_parts = _build_response_stream(1, set(), rng)
            raw = b"".join(part for part, _ in stream_parts)
            chunks = _fragment(raw, rng)
            for chunk in chunks:
                transport.parse_incoming(chunk)

            assert transport.buffer_count == 0, f"iter {i}"

    @pytest.mark.asyncio
    async def test_bulk_commands_interleaved_fragmented(self):
        transport = _make_grbl_transport()
        rng = random.Random(2002)

        total_bytes = 0
        for i in range(2000):
            line = f"G1 X{i} Y{i}\n".encode()
            await transport.send_gcode(line, op_index=i)
            total_bytes += len(line)

        assert transport.buffer_count == total_bytes

        error_positions = set(rng.sample(range(2000), 5))
        stream_parts = _build_response_stream(2000, error_positions, rng)
        raw = b"".join(part for part, _ in stream_parts)
        chunks = _fragment(raw, rng)
        for chunk in chunks:
            transport.parse_incoming(chunk)

        assert transport.buffer_count == 0
        assert transport.pending_queue.empty()

    @pytest.mark.asyncio
    async def test_ok_glued_to_status_report_no_separator(self):
        transport = _make_grbl_transport()
        rng = random.Random(3003)

        for i in range(500):
            await transport.send_gcode(f"G0 X{i}\n".encode(), op_index=i)

            report = rng.choice(STATUS_REPORTS)
            glued = report[:-2] + b"ok\r\n\r\n"
            chunks = _fragment(glued, rng)
            for chunk in chunks:
                transport.parse_incoming(chunk)

            assert transport.buffer_count == 0, f"iter {i}"

    @pytest.mark.asyncio
    async def test_error_interleaved_fragmented(self):
        transport = _make_grbl_transport()
        rng = random.Random(4004)

        num_cmds = 500
        for i in range(num_cmds):
            await transport.send_gcode(f"G1 X{i}\n".encode(), op_index=i)

        error_positions = {5, 15, 30, 45}
        stream_parts = _build_response_stream(num_cmds, error_positions, rng)
        raw = b"".join(part for part, _ in stream_parts)
        chunks = _fragment(raw, rng)
        for chunk in chunks:
            transport.parse_incoming(chunk)

        assert transport.buffer_count == 0
        assert transport.pending_queue.empty()

    @pytest.mark.asyncio
    async def test_null_corrupted_ok_with_random_splits(self):
        transport = _make_grbl_transport()
        rng = random.Random(5005)

        for i in range(2000):
            await transport.send_gcode(f"G0 X{i}\n".encode(), op_index=i)

            corrupted = b"o\x00k\r\n"
            chunks = _fragment(corrupted, rng)
            for chunk in chunks:
                transport.parse_incoming(chunk)

            assert transport.buffer_count == 0, f"iter {i}"

    @pytest.mark.asyncio
    async def test_multi_null_corrupted_ok(self):
        transport = _make_grbl_transport()
        rng = random.Random(6006)

        for i in range(1000):
            await transport.send_gcode(f"G1 X{i}\n".encode(), op_index=i)

            num_nulls = rng.randint(1, 5)
            corrupted = b"o" + b"\x00" * num_nulls + b"k\r\n"
            chunks = _fragment(corrupted, rng)
            for chunk in chunks:
                transport.parse_incoming(chunk)

            assert transport.buffer_count == 0, f"iter {i}"

    @pytest.mark.asyncio
    async def test_mixed_ok_error_alarm_fragmented_stream(self):
        transport = _make_grbl_transport()
        rng = random.Random(7007)

        for i in range(300):
            await transport.send_gcode(f"G0 X{i}\n".encode(), op_index=i)

        stream = bytearray()
        ack_count = 0
        for i in range(300):
            if rng.random() < 0.05:
                stream.extend(b"ALARM:1\r\n")
            if i in {50, 150}:
                stream.extend(b"error:20\r\n")
            stream.extend(b"ok\r\n")
            ack_count += 1
            if rng.random() < 0.3:
                stream.extend(rng.choice(STATUS_REPORTS))

        raw = bytes(stream)
        chunks = _fragment(raw, rng)
        for chunk in chunks:
            transport.parse_incoming(chunk)

        assert transport.buffer_count == 0
        assert transport.pending_queue.empty()

    @pytest.mark.asyncio
    async def test_ack_split_inside_ok_marker(self):
        transport = _make_grbl_transport()
        rng = random.Random(8008)

        for i in range(500):
            await transport.send_gcode(f"G1 X{i}\n".encode(), op_index=i)

            ok = b"ok\r\n"
            split = rng.randint(1, len(ok) - 1)
            transport.parse_incoming(ok[:split])
            transport.parse_incoming(ok[split:])

            assert transport.buffer_count == 0, f"iter {i}"

    @pytest.mark.asyncio
    async def test_every_possible_split_point(self):
        transport = _make_grbl_transport()

        response = (
            b"<Idle|MPos:0,0,0|FS:0,0>\r\nok\r\n<Run|MPos:1,1,0|FS:500,0>\r\n"
        )

        for split in range(1, len(response)):
            transport.reset()
            await transport.send_gcode(b"G0 X0\n", op_index=0)

            transport.parse_incoming(response[:split])
            transport.parse_incoming(response[split:])

            assert transport.buffer_count == 0, f"split={split}"

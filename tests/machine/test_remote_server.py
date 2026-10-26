"""End-to-end tests for the remote protocol server."""

import asyncio
import base64

import pytest
from raygeo.ops import Ops

from rayforge.machine.remote import JobPayload, RemoteJobServer
from rayforge.machine.remote.protocol import (
    PROTOCOL_VERSION,
    read_message,
    write_message,
)
from rayforge.pipeline.encoder.base import EncodedOutput, MachineCodeOpMap


def _make_encoded() -> EncodedOutput:
    return EncodedOutput(
        text="G0 X10\nG1 X20\nG1 X30\n",
        op_map=MachineCodeOpMap.from_lists(
            [(0, 1), (1, 1), (2, 1)], [0, 1, 2]
        ),
    )


def _make_payload() -> JobPayload:
    ops = Ops()
    ops.move_to(10, 10, 0)
    ops.line_to(20, 10, 0)
    ops.line_to(20, 20, 0)
    return JobPayload(
        encoded=_make_encoded(),
        ops=ops,
        doc_name="remote-test.rfz",
        distance_mm=10.0,
        estimated_seconds=1.0,
    )


class TestRemoteJobServer:
    @pytest.mark.asyncio
    async def test_hello_lists_machines(self, lite_context, task_mgr):
        machine = lite_context.machine_mgr.get_machines()[0]

        server = RemoteJobServer(lite_context, task_mgr)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", server.port
            )
            await write_message(
                writer, {"type": "hello", "version": PROTOCOL_VERSION}
            )
            welcome = await asyncio.wait_for(read_message(reader), 10)
            assert welcome["type"] == "reply"
            assert welcome["ok"]
            assert any(
                m["id"] == machine.id for m in welcome["result"]["machines"]
            )
            writer.close()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_hello_rejects_version_mismatch(
        self, lite_context, task_mgr
    ):
        server = RemoteJobServer(lite_context, task_mgr)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", server.port
            )
            await write_message(
                writer, {"type": "hello", "version": PROTOCOL_VERSION + 1}
            )
            reply = await asyncio.wait_for(read_message(reader), 10)
            assert reply["type"] == "reply"
            assert not reply["ok"]
            assert "version mismatch" in reply["error"]
            writer.close()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_get_machine_configs(self, lite_context, task_mgr):
        machine = lite_context.machine_mgr.get_machines()[0]

        server = RemoteJobServer(lite_context, task_mgr)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", server.port
            )
            await write_message(
                writer, {"type": "get_machine_configs", "id": 1}
            )
            reply = await asyncio.wait_for(read_message(reader), 10)
            assert reply["type"] == "reply"
            assert reply["ok"]
            config = reply["result"]["configs"][machine.id]
            assert config["machine"]["id"] == machine.id
            writer.close()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_machine_op_rejects_unknown_op(self, lite_context, task_mgr):
        machine = lite_context.machine_mgr.get_machines()[0]

        server = RemoteJobServer(lite_context, task_mgr)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", server.port
            )
            await write_message(
                writer,
                {
                    "type": "machine_op",
                    "id": 1,
                    "machine_id": machine.id,
                    "op": "does_not_exist",
                    "kwargs": {},
                },
            )
            reply = await asyncio.wait_for(read_message(reader), 10)
            assert reply["type"] == "reply"
            assert not reply["ok"]
            assert "Unknown machine op" in reply["error"]
            writer.close()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_submit_job_runs_to_completion(
        self, lite_context, task_mgr, machine
    ):
        server = RemoteJobServer(lite_context, task_mgr)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", server.port
            )
            await write_message(
                writer,
                {
                    "type": "submit_job",
                    "id": 1,
                    "machine_id": machine.id,
                    "payload": base64.b64encode(
                        _make_payload().to_bytes()
                    ).decode("ascii"),
                },
            )
            accepted = await asyncio.wait_for(read_message(reader), 10)
            assert accepted["type"] == "reply"
            assert accepted["ok"]
            job_id = accepted["result"]["job_id"]

            finished = None
            while finished is None:
                message = await asyncio.wait_for(read_message(reader), 30)
                if message["type"] == "job_progress":
                    assert message["job_id"] == job_id
                    assert 0 <= message["op_index"] <= 2
                elif message["type"] == "job_finished":
                    finished = message
            assert finished["job_id"] == job_id
            assert finished["outcome"] == "COMPLETED"

            await write_message(writer, {"type": "get_job", "id": 2})
            state = await asyncio.wait_for(read_message(reader), 10)
            assert state["type"] == "reply"
            assert state["result"]["job_id"] == job_id
            assert state["result"]["running"] is False
            assert state["result"]["outcome"] == "COMPLETED"
            writer.close()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_submit_job_survives_client_disconnect(
        self, lite_context, task_mgr, machine
    ):
        server = RemoteJobServer(lite_context, task_mgr)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", server.port
            )
            await write_message(
                writer,
                {
                    "type": "submit_job",
                    "id": 1,
                    "machine_id": machine.id,
                    "payload": base64.b64encode(
                        _make_payload().to_bytes()
                    ).decode("ascii"),
                },
            )
            accepted = await asyncio.wait_for(read_message(reader), 10)
            assert accepted["ok"]
            job_id = accepted["result"]["job_id"]

            # The client goes away mid-job.
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.5)

            # A reconnecting client can still observe the job.
            reader2, writer2 = await asyncio.open_connection(
                "127.0.0.1", server.port
            )
            deadline = asyncio.get_event_loop().time() + 30
            state = None
            while True:
                await write_message(writer2, {"type": "get_job", "id": 1})
                reply = await asyncio.wait_for(read_message(reader2), 10)
                assert reply["type"] == "reply"
                assert reply["ok"]
                state = reply["result"]
                if not state["running"]:
                    break
                if asyncio.get_event_loop().time() > deadline:
                    pytest.fail("Job did not finish after disconnect")
                await asyncio.sleep(0.1)
            assert state["job_id"] == job_id
            assert state["outcome"] == "COMPLETED"
            writer2.close()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_submit_job_rejects_unknown_machine(
        self, lite_context, task_mgr
    ):
        server = RemoteJobServer(lite_context, task_mgr)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", server.port
            )
            await write_message(
                writer,
                {
                    "type": "submit_job",
                    "id": 1,
                    "machine_id": "does-not-exist",
                    "payload": base64.b64encode(
                        _make_payload().to_bytes()
                    ).decode("ascii"),
                },
            )
            reply = await asyncio.wait_for(read_message(reader), 10)
            assert reply["type"] == "reply"
            assert not reply["ok"]
            writer.close()
        finally:
            await server.stop()


def test_message_framing_roundtrip():
    from rayforge.machine.remote.protocol import encode_message

    message = {"type": "hello", "version": 1, "nested": {"a": [1, 2, 3]}}
    data = encode_message(message)
    assert data[:4] == len(data[4:]).to_bytes(4, "big")

    import json

    assert json.loads(data[4:].decode("utf-8")) == message

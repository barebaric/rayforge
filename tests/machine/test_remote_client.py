"""End-to-end tests for the remote client (controller-level split)."""

import asyncio

import pytest
from raygeo.ops import Ops
from raygeo.ops.axis import Axis

from rayforge import config
from rayforge import context as context_module
from rayforge.context import RayforgeContext
from rayforge.machine.models.dialect_manager import DialectManager
from rayforge.machine.remote import (
    ClientMachineManager,
    JobPayload,
    RemoteClient,
    RemoteDriverProxy,
    RemoteJobServer,
)
from rayforge.pipeline.encoder.base import EncodedOutput, MachineCodeOpMap


def _make_encoded() -> EncodedOutput:
    return EncodedOutput(
        text="G0 X10\nG1 X20\nG1 X30\n",
        op_map=MachineCodeOpMap.from_lists(
            [(0, 1), (1, 1), (2, 1)], [0, 1, 2]
        ),
    )


def _make_ops() -> Ops:
    ops = Ops()
    ops.move_to(10, 10, 0)
    ops.line_to(20, 10, 0)
    ops.line_to(20, 20, 0)
    return ops


@pytest.mark.asyncio
async def test_client_mirrors_and_drives_machine(
    lite_context, task_mgr, machine
):
    """A client mirrors the server machine and drives it fully."""
    # The server machine must carry a driver for the client to proxy.
    machine.driver_name = "NoDeviceDriver"

    server = RemoteJobServer(lite_context, task_mgr)
    await server.start()

    client = RemoteClient()
    client_context = None
    original_context = context_module._context_instance
    try:
        await client.connect("127.0.0.1", server.port)
        configs = await client.get_machine_configs()
        assert machine.id in configs

        # The client stack gets its own context so it never touches
        # the server's machine manager.
        client_context = RayforgeContext()
        client_context._headless = True
        client_context._dialect_mgr = DialectManager(config.DIALECT_DIR)
        context_module._context_instance = client_context

        mirror_mgr = ClientMachineManager(config.MACHINE_DIR, client)
        client_context.install_machine_manager(mirror_mgr)
        mirror_mgr.populate_from_configs(configs)
        mirror = mirror_mgr.machines[machine.id]
        assert mirror.driver_name == "NoDeviceDriver"

        controller = mirror_mgr.get_controller(mirror.id)
        task_mgr.wait_until_settled(5000)
        assert isinstance(controller.driver, RemoteDriverProxy)

        # Connecting the mirror connects the machine on the server.
        await mirror.connect()
        assert mirror.is_connected()
        assert machine.is_connected()

        # Interactive commands forward as machine ops.
        await controller.jog({Axis.X: 5.0}, 1000)
        await controller.move_to(10, 10)
        await controller.clear_alarm()

        # Job execution runs on the server with streamed progress.
        done_ops: list[int] = []
        await controller.run(
            _make_encoded(),
            None,
            _make_ops(),
            on_command_done=done_ops.append,
        )
        assert sorted(done_ops) == [0, 1, 2]
    finally:
        context_module._context_instance = original_context
        if client_context is not None:
            mirror_mgr = client_context.machine_mgr
            await mirror_mgr.shutdown()
        await client.close()
        await server.stop()


@pytest.mark.asyncio
async def test_client_survives_reconnect(lite_context, task_mgr, machine):
    """A client that disconnects mid-job can re-attach via get_job."""
    machine.driver_name = "NoDeviceDriver"

    server = RemoteJobServer(lite_context, task_mgr)
    await server.start()
    client = RemoteClient()
    try:
        await client.connect("127.0.0.1", server.port)
        configs = await client.get_machine_configs()

        client_context = RayforgeContext()
        client_context._headless = True
        client_context._dialect_mgr = DialectManager(config.DIALECT_DIR)
        context_module._context_instance = client_context
        try:
            mirror_mgr = ClientMachineManager(config.MACHINE_DIR, client)
            client_context.install_machine_manager(mirror_mgr)
            mirror_mgr.populate_from_configs(configs)

            # Submit a job, then drop the connection mid-flight.
            job_id = await client.submit_job(
                machine.id,
                JobPayload(
                    encoded=_make_encoded(),
                    ops=_make_ops(),
                    doc_name="test.rfz",
                    distance_mm=10.0,
                    estimated_seconds=1.0,
                ),
            )
            await client.close()

            # Wait for the server to finish the job on its own.
            await asyncio.sleep(0.5)

            # Reconnect and ask for the job state.
            client2 = RemoteClient()
            try:
                await client2.connect("127.0.0.1", server.port)
                state = await asyncio.wait_for(
                    client2.request({"type": "get_job"}), 10
                )
                assert state["job_id"] == job_id
                assert state["running"] is False
                assert state["outcome"] == "COMPLETED"
            finally:
                await client2.close()
        finally:
            context_module._context_instance = None
    finally:
        await server.stop()

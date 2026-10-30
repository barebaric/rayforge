"""Tests for the unified local path (in-process backend + frontend).

Plain ``rayforge`` runs an in-process backend and drives it through
the same protocol a network frontend uses. These tests exercise that
path over a Unix domain socket: mirroring, machine CRUD syncing to
the backend, and job execution.
"""

import asyncio
import os
import socket
import sys

import pytest
from raygeo.ops import Ops

from rayforge import config
from rayforge import context as context_module
from rayforge.context import RayforgeContext
from rayforge.machine.models.dialect_manager import DialectManager
from rayforge.machine.models.machine import Machine
from rayforge.machine.remote import (
    ClientMachineManager,
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


def _unix_socket_supported() -> bool:
    return (
        hasattr(socket, "AF_UNIX")
        and sys.platform != "win32"
        and hasattr(asyncio, "open_connection")
    )


@pytest.fixture
def socket_path(tmp_path):
    return str(tmp_path / "backend.sock")


@pytest.mark.asyncio
async def test_local_backend_over_socket(
    lite_context, task_mgr, machine, monkeypatch, socket_path
):
    """The frontend path works against an in-process backend."""
    machine.driver_name = "NoDeviceDriver"

    server = RemoteJobServer(lite_context, task_mgr, socket_path=socket_path)
    await server.start()
    client = RemoteClient()
    client_context = None
    original_context = context_module._context_instance
    try:
        await client.connect(socket_path)
        configs = await client.get_machine_configs()
        assert machine.id in configs

        client_context = RayforgeContext()
        client_context._headless = True
        client_context._dialect_mgr = DialectManager(config.DIALECT_DIR)
        context_module._context_instance = client_context

        mirror_mgr = ClientMachineManager(config.MACHINE_DIR, client)
        monkeypatch.setattr(mirror_mgr, "SYNC_DEBOUNCE_SECONDS", 0.05)
        client_context.install_machine_manager(mirror_mgr)
        mirror_mgr.populate_from_configs(configs)
        mirror = mirror_mgr.machines[machine.id]

        controller = mirror_mgr.get_controller(mirror.id)
        task_mgr.wait_until_settled(5000)
        assert isinstance(controller.driver, RemoteDriverProxy)

        # Connecting the mirror connects the machine on the backend.
        await mirror.connect()
        assert mirror.is_connected()
        assert machine.is_connected()

        # Job execution runs on the backend.
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
            await client_context.machine_mgr.shutdown()
        await client.close()
        await server.stop()
    assert not os.path.exists(socket_path)


@pytest.mark.asyncio
async def test_machine_edits_sync_to_backend(
    lite_context, task_mgr, machine, monkeypatch, socket_path
):
    """Machine changes on the frontend persist on the backend."""
    machine.driver_name = "NoDeviceDriver"

    server = RemoteJobServer(lite_context, task_mgr, socket_path=socket_path)
    await server.start()
    client = RemoteClient()
    client_context = None
    original_context = context_module._context_instance
    try:
        await client.connect(socket_path)
        configs = await client.get_machine_configs()

        client_context = RayforgeContext()
        client_context._headless = True
        client_context._dialect_mgr = DialectManager(config.DIALECT_DIR)
        context_module._context_instance = client_context
        mirror_mgr = ClientMachineManager(config.MACHINE_DIR, client)
        monkeypatch.setattr(mirror_mgr, "SYNC_DEBOUNCE_SECONDS", 0.05)
        client_context.install_machine_manager(mirror_mgr)
        mirror_mgr.populate_from_configs(configs)
        mirror = mirror_mgr.machines[machine.id]

        # An edit on the mirror syncs to the backend after the
        # debounce window.
        mirror.name = "Renamed Machine"
        mirror.changed.send(mirror)
        await asyncio.sleep(0.3)

        backend_machine = lite_context.machine_mgr.get_machine_by_id(
            machine.id
        )
        assert backend_machine is not None
        assert backend_machine.name == "Renamed Machine"
    finally:
        context_module._context_instance = original_context
        if client_context is not None:
            await client_context.machine_mgr.shutdown()
        await client.close()
        await server.stop()


@pytest.mark.asyncio
async def test_machine_create_and_remove_sync_to_backend(
    lite_context, task_mgr, monkeypatch, socket_path
):
    """Machines created and deleted on the frontend reach the backend."""
    server = RemoteJobServer(lite_context, task_mgr, socket_path=socket_path)
    await server.start()
    client = RemoteClient()
    client_context = None
    original_context = context_module._context_instance
    try:
        await client.connect(socket_path)

        client_context = RayforgeContext()
        client_context._headless = True
        client_context._dialect_mgr = DialectManager(config.DIALECT_DIR)
        context_module._context_instance = client_context
        mirror_mgr = ClientMachineManager(config.MACHINE_DIR, client)
        client_context.install_machine_manager(mirror_mgr)

        # Adding a machine on the frontend (like the setup wizard
        # does) creates it on the backend.
        new_machine = Machine(client_context)
        new_machine.name = "Wizard Machine"
        mirror_mgr.add_machine(new_machine)
        await asyncio.sleep(0.2)

        backend_machine = lite_context.machine_mgr.get_machine_by_id(
            new_machine.id
        )
        assert backend_machine is not None
        assert backend_machine.name == "Wizard Machine"
        backend_ids = set(lite_context.machine_mgr.machines)
        assert new_machine.id in backend_ids

        # Removing it on the frontend removes it on the backend.
        mirror_mgr.remove_machine(new_machine.id)
        await asyncio.sleep(0.2)
        assert (
            lite_context.machine_mgr.get_machine_by_id(new_machine.id) is None
        )
    finally:
        context_module._context_instance = original_context
        if client_context is not None:
            await client_context.machine_mgr.shutdown()
        await client.close()
        await server.stop()


@pytest.mark.skipif(
    not _unix_socket_supported(), reason="no Unix domain sockets"
)
def test_socket_fallback_helpers():
    from rayforge.machine.remote.bootstrap import (
        local_socket_path,
        supports_unix_sockets,
    )

    assert supports_unix_sockets()
    path = local_socket_path()
    assert "rayforge-backend-" in path
    assert str(os.getpid()) in path

"""Test for the full unified bootstrap (in-process backend + frontend)."""

import asyncio
from typing import TYPE_CHECKING, cast

import pytest
from raygeo.ops import Ops

from rayforge import config
from rayforge import context as context_module
from rayforge.context import RayforgeContext
from rayforge.machine.models.controller import MachineController
from rayforge.machine.models.machine import Machine
from rayforge.pipeline.encoder.base import (
    EncodedOutput,
    MachineCodeOpMap,
)
from rayforge.shared import tasker
from rayforge.shared.tasker.manager import TaskManager, TaskManagerProxy

if TYPE_CHECKING:
    from rayforge.core.doc import Doc


@pytest.fixture
def local_task_manager(monkeypatch):
    """A fresh TaskManager, as the application would have one."""
    tm = TaskManager(main_thread_scheduler=lambda f, *a, **kw: f(*a, **kw))
    proxy = cast(TaskManagerProxy, tasker.task_mgr)
    monkeypatch.setattr(proxy, "_instance", tm)
    monkeypatch.setattr(proxy, "_init_kwargs", {})
    yield tm
    tm.shutdown()


@pytest.mark.asyncio
async def test_bootstrap_local_app(tmp_path, monkeypatch, local_task_manager):
    """The unified bootstrap wires backend, frontend, and mirrors."""
    from rayforge.machine.models.manager import MachineManager
    from rayforge.machine.remote import (
        ClientMachineManager,
        RemoteDriverProxy,
    )
    from rayforge.machine.remote import bootstrap as remote_bootstrap

    temp_config_dir = tmp_path / "config"
    monkeypatch.setattr(config, "CONFIG_DIR", temp_config_dir)
    monkeypatch.setattr(config, "MACHINE_DIR", temp_config_dir / "machines")
    monkeypatch.setattr(config, "DIALECT_DIR", temp_config_dir / "dialects")

    # Pre-create a machine whose driver precheck fails, like a laser
    # whose serial port is not attached. Its rebuild on the backend
    # must never touch the frontend context (regression: the backend
    # controller used to bind to the process-wide context singleton
    # and lazily created a machine manager on the frontend).
    from rayforge.machine.models.dialect_manager import (
        DialectManager,
    )
    from rayforge.machine.models.machine import Machine

    context_module._context_instance = RayforgeContext()
    setup_ctx = context_module._context_instance
    setup_ctx._dialect_mgr = DialectManager(config.DIALECT_DIR)
    preconfigured = Machine(setup_ctx)
    preconfigured.name = "TestLaser"
    preconfigured.driver_name = "GrblSerialDriver"
    preconfigured.driver_args = {"port": "/dev/does-not-exist"}
    setup_ctx.machine_mgr.add_machine(preconfigured)
    context_module._context_instance = None

    monkeypatch.setattr(context_module, "_context_instance", RayforgeContext())

    setup = remote_bootstrap.bootstrap_local_app()
    tm = local_task_manager
    try:
        frontend = context_module._context_instance
        backend = setup.backend_context
        assert frontend is setup.frontend_context
        assert frontend is not None
        assert isinstance(frontend.machine_mgr, ClientMachineManager)
        assert isinstance(backend.machine_mgr, MachineManager)

        # The frontend mirrors the backend's machines.
        backend_ids = set(backend.machine_mgr.machines)
        assert set(frontend.machine_mgr.machines) == backend_ids
        assert preconfigured.id in backend_ids

        # Controllers bind to their machine's context, not the
        # process-wide singleton: the backend machine's controller
        # must be bound to the backend context.
        backend_controller = backend.machine_mgr.get_controller(
            preconfigured.id
        )
        backend_machine = backend.machine_mgr.get_machine_by_id(
            preconfigured.id
        )
        assert backend_machine is not None
        assert backend_controller.context is backend_machine.context
        assert backend_controller.context is backend

        # A machine edit on the frontend persists on the backend.
        mirror = next(iter(frontend.machine_mgr.machines.values()))
        # Keep the test fast: shrink the edit sync debounce.
        frontend.machine_mgr.SYNC_DEBOUNCE_SECONDS = 0.05
        mirror.name = "Unified Machine"
        mirror.changed.send(mirror)
        await asyncio.sleep(0.5)
        backend_machine = backend.machine_mgr.get_machine_by_id(mirror.id)
        assert backend_machine is not None
        assert backend_machine.name == "Unified Machine"

        # Configuring a driver on the frontend machine also reaches
        # the backend through the edit sync.
        mirror.driver_name = "NoDeviceDriver"
        mirror.changed.send(mirror)
        await asyncio.sleep(0.5)
        backend_machine = backend.machine_mgr.get_machine_by_id(mirror.id)
        assert backend_machine is not None
        assert backend_machine.driver_name == "NoDeviceDriver"

        # The frontend drives the machine through the protocol.
        assert frontend is not None
        controller: MachineController = frontend.machine_mgr.get_controller(
            mirror.id
        )
        tm.wait_until_settled(5000)
        assert isinstance(controller.driver, RemoteDriverProxy)

        loop = tm.loop
        future = asyncio.run_coroutine_threadsafe(
            _drive_machine(controller, mirror), loop
        )
        assert sorted(future.result(timeout=30)) == [0, 1]
    finally:
        shutdown = asyncio.run_coroutine_threadsafe(
            remote_bootstrap.shutdown_local_backend(), tm.loop
        )
        shutdown.result(timeout=30)


async def _drive_machine(
    controller: MachineController, mirror: Machine
) -> list[int]:
    """Connects the mirror machine and runs a minimal job on it."""
    await mirror.connect()
    assert mirror.is_connected()
    encoded = EncodedOutput(
        text="G0 X10\nG1 X20\n",
        op_map=MachineCodeOpMap.from_lists([(0, 1), (1, 1)], [0, 1]),
    )
    ops = Ops()
    ops.move_to(10, 10, 0)
    ops.line_to(20, 10, 0)
    done_ops: list[int] = []
    await controller.run(
        encoded,
        cast("Doc", None),
        ops,
        on_command_done=done_ops.append,
    )
    return done_ops

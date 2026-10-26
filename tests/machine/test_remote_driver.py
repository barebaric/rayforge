"""End-to-end tests for the client-side remote driver."""

from typing import TYPE_CHECKING, cast

import pytest
from raygeo.ops import Ops

from rayforge.machine.driver.driver import DriverPrecheckError
from rayforge.machine.remote import RemoteDriver, RemoteJobServer
from rayforge.machine.remote.sync import (
    fetch_machine_config,
    import_remote_machine,
)
from rayforge.pipeline.encoder.base import EncodedOutput, MachineCodeOpMap

if TYPE_CHECKING:
    from rayforge.core.doc import Doc


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


class TestRemoteDriver:
    @pytest.mark.asyncio
    async def test_import_remote_machine(self, lite_context, task_mgr):
        server = RemoteJobServer(lite_context, task_mgr)
        await server.start()
        try:
            config = await fetch_machine_config("127.0.0.1", server.port)
            mirror = import_remote_machine(
                lite_context.machine_mgr, config, "127.0.0.1", server.port
            )
            try:
                assert mirror.driver_name == "RemoteDriver"
                args = mirror.driver_args
                assert args["host"] == "127.0.0.1"
                assert args["port"] == server.port
                assert args["machine_id"]
            finally:
                await mirror.shutdown()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_remote_driver_runs_job(
        self, lite_context, task_mgr, machine
    ):
        server = RemoteJobServer(lite_context, task_mgr)
        await server.start()
        mirror = None
        driver = None
        try:
            config = await fetch_machine_config(
                "127.0.0.1", server.port, machine.id
            )
            mirror = import_remote_machine(
                lite_context.machine_mgr, config, "127.0.0.1", server.port
            )

            driver = RemoteDriver(lite_context, mirror)
            driver.setup(**mirror.driver_args)
            await driver.connect()

            done_ops: list[int] = []
            await driver.run(
                _make_encoded(),
                cast("Doc", None),
                _make_ops(),
                on_command_done=done_ops.append,
            )
            assert sorted(done_ops) == [0, 1, 2]
        finally:
            if driver is not None:
                await driver.cleanup()
            if mirror is not None:
                await mirror.shutdown()
            await server.stop()

    @pytest.mark.asyncio
    async def test_remote_driver_run_raw(
        self, lite_context, task_mgr, machine
    ):
        server = RemoteJobServer(lite_context, task_mgr)
        await server.start()
        mirror = None
        driver = None
        try:
            config = await fetch_machine_config(
                "127.0.0.1", server.port, machine.id
            )
            mirror = import_remote_machine(
                lite_context.machine_mgr, config, "127.0.0.1", server.port
            )

            driver = RemoteDriver(lite_context, mirror)
            driver.setup(**mirror.driver_args)
            await driver.connect()
            await driver.run_raw("G0 X0")
        finally:
            if driver is not None:
                await driver.cleanup()
            if mirror is not None:
                await mirror.shutdown()
            await server.stop()

    @pytest.mark.asyncio
    async def test_remote_driver_rejects_unknown_machine(
        self, lite_context, task_mgr
    ):
        server = RemoteJobServer(lite_context, task_mgr)
        await server.start()
        try:
            with pytest.raises(ValueError):
                await fetch_machine_config(
                    "127.0.0.1", server.port, "does-not-exist"
                )
        finally:
            await server.stop()

    def test_precheck_requires_host(self):
        with pytest.raises(DriverPrecheckError):
            RemoteDriver.precheck()

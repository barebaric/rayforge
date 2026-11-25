"""Ruida driver binding tests.

The protocol stack itself is covered by raydriver's suite; these tests
cover the Rayforge binding: setup/config mapping, connection over the
native emulator, job submission with the compiled payload, live
motion/job control, WCS policy and the encoder context.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from raydriver.ruida import RuidaDriver as NativeRuidaDriver
from raydriver.ruida.emulator import RuidaEmulator
from raygeo.ops.axis import Axis

from rayforge.machine.driver.driver import (
    DriverPrecheckError,
    ZHomingMode,
)
from rayforge.machine.driver.ruidarpa import RuidaRPAAdapter
from rayforge.pipeline.artifact.encoded import (
    EncodedOutput,
    MachineCodeOpMap,
)
from rayforge.pipeline.encode_inputs import build_ruida_spec


async def wait_until(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"condition not met within {timeout:.1f}s")


async def wait_for_connected(driver, timeout=5.0):
    connected = asyncio.Event()

    def _on_status(sender, status=None, message=None, **kwargs):
        if status is not None and status.name == "CONNECTED":
            connected.set()

    driver.connection_status_changed.connect(_on_status)
    try:
        await driver.connect()
        await asyncio.wait_for(connected.wait(), timeout=timeout)
    finally:
        driver.connection_status_changed.disconnect(_on_status)


@pytest_asyncio.fixture
async def emulator():
    device = RuidaEmulator()
    await device.start()
    try:
        yield device
    finally:
        await device.stop()


@pytest_asyncio.fixture
async def driver(lite_context, machine, emulator):
    drv = RuidaRPAAdapter(lite_context, machine)
    drv.setup(
        udp_host="127.0.0.1",
        port=emulator.port,
        connection="network",
        tcp_proto=True,
    )
    try:
        yield drv
    finally:
        await drv.cleanup()


@pytest_asyncio.fixture
async def connected_driver(driver):
    await wait_for_connected(driver)
    await wait_until(lambda: driver.state.status.name == "IDLE")
    yield driver


class TestMetadata:
    def test_native_class(self):
        assert RuidaRPAAdapter.NATIVE is NativeRuidaDriver
        assert RuidaRPAAdapter.native_encoder == "ruida"
        assert RuidaRPAAdapter.uses_gcode is False

    def test_setup_vars(self):
        keys = set(RuidaRPAAdapter.get_setup_vars().keys())
        assert {
            "connection",
            "tcp_proto",
            "udp_host",
            "usb_device",
            "magic_number",
            "power_floor",
            "power_floor_enabled",
            "image_power_bias",
            "power_scaling_enabled",
        } <= keys
        assert "tui" not in keys

    def test_precheck_requires_an_endpoint(self):
        with pytest.raises(DriverPrecheckError, match="At least one"):
            RuidaRPAAdapter.precheck(connection="auto")
        with pytest.raises(DriverPrecheckError, match="Hostname"):
            RuidaRPAAdapter.precheck(connection="network", udp_host="")
        with pytest.raises(DriverPrecheckError, match="USB"):
            RuidaRPAAdapter.precheck(connection="usb", usb_device="")

    def test_precheck_rejects_tui(self):
        with pytest.raises(DriverPrecheckError, match="TUI"):
            RuidaRPAAdapter.precheck(
                connection="network", udp_host="host", tui=True
            )

    def test_build_config_network_tcp(self, lite_context, machine):
        drv = RuidaRPAAdapter(lite_context, machine)
        config = drv.build_config(
            {
                "connection": "network",
                "udp_host": "192.168.1.5",
                "usb_device": "ignored",
                "tcp_proto": True,
                "magic_number": "0x99",
            }
        )
        assert config == {
            "host": "192.168.1.5",
            "usb_device": "",
            "protocol": "tcp",
            "magic_number": "0x99",
            "protect": True,
        }

    def test_build_config_usb(self, lite_context, machine):
        drv = RuidaRPAAdapter(lite_context, machine)
        config = drv.build_config(
            {
                "connection": "usb",
                "udp_host": "ignored",
                "usb_device": "/dev/ttyUSB0",
            }
        )
        assert config["host"] == ""
        assert config["usb_device"] == "/dev/ttyUSB0"
        assert config["protocol"] == "udp"

    def test_build_config_rejects_bad_magic(self, lite_context, machine):
        drv = RuidaRPAAdapter(lite_context, machine)
        with pytest.raises(Exception, match="Magic"):
            drv.build_config(
                {
                    "connection": "network",
                    "udp_host": "host",
                    "magic_number": "nope",
                }
            )

    def test_setup_seeds_machine_speeds(self, lite_context, machine):
        machine.set_max_cut_speed(1000)
        machine.set_max_travel_speed(3000)
        drv = RuidaRPAAdapter(lite_context, machine)
        drv.setup(host="x", connection="network")
        assert machine.max_cut_speed == 24000
        assert machine.max_travel_speed == 36000


@pytest.mark.asyncio
class TestConnection:
    async def test_connect_reports_idle(self, connected_driver):
        assert connected_driver.session is not None
        assert connected_driver.state.status.name == "IDLE"

    async def test_status_updates(self, connected_driver, emulator):
        emulator.set_job_running(True)
        await wait_until(lambda: connected_driver.state.status.name == "RUN")
        emulator.set_paused(True)
        await wait_until(lambda: connected_driver.state.status.name == "HOLD")
        emulator.set_job_running(False)
        emulator.set_paused(False)
        await wait_until(lambda: connected_driver.state.status.name == "IDLE")


@pytest.mark.asyncio
class TestJobs:
    async def test_run_uses_the_compiled_payload(
        self, connected_driver, emulator
    ):
        payload = b"SPEED_LASER_1 100.0mm/S\nEND_JOB"
        encoded = EncodedOutput(
            text="ignored transcript",
            op_map=MachineCodeOpMap(),
            driver_data={"payload": payload},
        )
        await connected_driver.run(encoded, MagicMock(), MagicMock())
        await wait_until(
            lambda: any("e505" in line for line in emulator.commands)
        )

    async def test_run_raw(self, connected_driver, emulator):
        await connected_driver.run_raw("SPEED_LASER_1 100.0mm/S\nEND_JOB")
        await wait_until(
            lambda: any("e505" in line for line in emulator.commands)
        )


@pytest.mark.asyncio
class TestMovement:
    async def test_move_to_sets_speed_and_absolute_jog(
        self, connected_driver, emulator
    ):
        await connected_driver.move_to(10.0, 20.0, speed=6000)
        assert any("d910" in line for line in emulator.commands)
        assert any("1c20" in line for line in emulator.commands)

    async def test_jog_relative(self, connected_driver, emulator):
        await connected_driver.jog(600, x=1.0, y=2.0, z=3.0)
        assert any("d910" in line for line in emulator.commands)
        assert any("d902" in line for line in emulator.commands)

    async def test_frame_uses_beam_off_moves(self, connected_driver, emulator):
        corners = [(0.0, 0.0, None), (10.0, 0.0, None)]
        await connected_driver.frame(corners, 600.0, MagicMock())
        assert any("d910" in line for line in emulator.commands)
        assert not any("a9" in line for line in emulator.commands)


@pytest.mark.asyncio
class TestHoming:
    async def test_home_xy(self, connected_driver, emulator):
        await connected_driver.home()
        assert any("d910" in line for line in emulator.commands)

    async def test_home_z_switch(self, connected_driver, emulator, machine):
        machine.set_z_homing_mode(ZHomingMode.SWITCH)
        await connected_driver.home(Axis.Z)
        assert any("d82c" in line for line in emulator.commands)

    async def test_home_z_probe(self, connected_driver, emulator, machine):
        machine.set_z_homing_mode(ZHomingMode.PROBE)
        await connected_driver.home(Axis.Z)
        assert any("d82e" in line for line in emulator.commands)


@pytest.mark.asyncio
class TestWcs:
    async def test_supported_wcs(self, connected_driver):
        assert connected_driver.supported_wcs == [
            "MACHINE",
            "ANCHOR",
            "CURRENT",
            "SET_POINT",
        ]
        assert connected_driver.machine_space_wcs == "MACHINE"

    async def test_select_and_read_parser_state(self, connected_driver):
        await connected_driver.select_wcs("CURRENT")
        assert await connected_driver.read_parser_state() == "CURRENT"
        with pytest.raises(ValueError):
            await connected_driver.select_wcs("G54")

    async def test_read_offsets_emits_signal(self, connected_driver):
        offsets = await connected_driver.read_wcs_offsets()
        assert set(offsets) == {
            "MACHINE",
            "ANCHOR",
            "CURRENT",
            "SET_POINT",
        }

    async def test_set_offset_unsupported(self, connected_driver):
        with pytest.raises(NotImplementedError):
            await connected_driver.set_wcs_offset("G54", 1, 2, 3)


@pytest.mark.asyncio
class TestProtect:
    async def test_toggle(self, connected_driver):
        assert await connected_driver.get_protect() is True
        await connected_driver.set_protect(False)
        assert await connected_driver.get_protect() is False
        await connected_driver.set_protect(True)
        assert await connected_driver.get_protect() is True


class TestEncoderContext:
    def test_context_maps_layers_and_wcs(self, machine):
        from rayforge.core.doc import Doc

        doc = Doc()
        doc.layers[0].name = "Cut"
        doc.layers[0].color = "#ff6600"
        machine.active_wcs = "ANCHOR"
        spec = build_ruida_spec(machine, doc)
        import json

        context = json.loads(spec.context_json)
        assert context["ref_point"] == "ABSOLUTE"
        assert context["layers"][0]["name"] == "Cut"
        assert context["layers"][0]["color"] == "#ff6600"

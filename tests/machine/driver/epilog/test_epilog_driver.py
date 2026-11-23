"""Epilog driver tests.

The driver is a thin shell over the raydriver LPD session: it builds
the session from the setup vars, uploads the encoder's byte stream, and
reports every machine operation the print protocol cannot perform.

The session is mocked with an ``AsyncMock`` spec'd against the real
``EpilogSession``; the encoder itself is covered by raygeo's tests.
"""

from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from rayforge.core.doc import Doc
from rayforge.machine.driver.driver import (
    Axis,
    DriverPrecheckError,
    DriverSetupError,
)
from rayforge.machine.driver.epilog import EpilogZingDriver
from rayforge.machine.driver.epilog.epilog_driver import DEFAULT_PORT
from rayforge.machine.models.laser import LaserHead
from rayforge.machine.models.machine import Machine
from rayforge.pipeline.encoder.base import EncodedOutput, MachineCodeOpMap


@pytest_asyncio.fixture
async def machine(lite_context) -> Machine:
    m = Machine(lite_context)
    m.name = "Epilog"
    m.add_head(LaserHead())
    m.set_axis_extents(609.6, 304.8)
    return m


@pytest.fixture
def setup_args() -> dict:
    return {
        "host": "10.0.0.1",
        "port": DEFAULT_PORT,
        "resolution": "500",
        "frequency": 1000,
    }


@pytest.fixture
def doc() -> Doc:
    doc = Doc()
    doc.name = "Test Job"
    return doc


@pytest.fixture
def session() -> Mock:
    # A bare AsyncMock: ``spec=EpilogSession`` would inspect the Rust
    # extension's method descriptors and treat the coroutine methods as
    # synchronous.
    session = Mock()
    session.upload = AsyncMock()
    session.cancel = AsyncMock()
    return session


@pytest.fixture
def driver(lite_context, machine, setup_args, session) -> EpilogZingDriver:
    driver = EpilogZingDriver(lite_context, machine)
    driver.setup(**setup_args)
    driver._session = session
    return driver


def _encoded(payload: bytes | None) -> EncodedOutput:
    return EncodedOutput(
        text="listing",
        op_map=MachineCodeOpMap(),
        driver_data={} if payload is None else {"payload": payload},
    )


class TestCapabilities:
    def test_is_not_a_gcode_driver(self):
        assert EpilogZingDriver.uses_gcode is False

    def test_no_homing_jogging_or_probing(self, driver):
        assert driver.can_home() is False
        assert driver.can_jog() is False
        assert driver.supports_probing is False

    def test_can_frame(self, driver):
        assert driver.can_frame() is True

    def test_single_machine_wcs(self, driver):
        assert driver.machine_space_wcs == "MACHINE"
        assert driver.supported_wcs == ["MACHINE"]

    def test_no_travel_speed(self):
        assert EpilogZingDriver.supports_travel_speed(None) is False

    @pytest.mark.asyncio
    async def test_read_parser_state_is_machine(self, driver):
        assert await driver.read_parser_state() == "MACHINE"
        assert await driver.read_wcs_offsets() == {"MACHINE": (0.0, 0.0, 0.0)}


class TestPrecheck:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"host": ""},
            {"host": "not a host"},
            {"port": 0},
            {"port": 70000},
            {"port": "515"},
            {"resolution": "300"},
            {"frequency": 5},
            {"frequency": 6000},
        ],
    )
    def test_invalid_args(self, setup_args, overrides):
        args = {**setup_args, **overrides}
        with pytest.raises(DriverPrecheckError):
            EpilogZingDriver.precheck(**args)

    def test_valid_args(self, setup_args):
        EpilogZingDriver.precheck(**setup_args)

    def test_setup_without_host_records_error(self, lite_context, machine):
        driver = EpilogZingDriver(lite_context, machine)
        driver.setup(host="", port=DEFAULT_PORT)
        assert driver.session is None
        assert driver.state.error is not None


class TestSetup:
    def test_creates_session(self, lite_context, machine, setup_args):
        driver = EpilogZingDriver(lite_context, machine)
        driver.setup(**setup_args)
        assert driver.session is not None
        assert driver.state.error is None

    def test_no_settings_pages(self, driver):
        assert driver.get_setting_vars() == []

    def test_queue_defaults_to_laser(self):
        keys = EpilogZingDriver.get_setup_vars().keys()
        assert "queue" in keys
        var = EpilogZingDriver.get_setup_vars().get("queue")
        assert var is not None
        assert var.default == "Laser"


class TestRun:
    @pytest.mark.asyncio
    async def test_uploads_payload(self, driver, session, doc):
        payload = b"\x1b%-12345X@PJL JOB" + bytes(16)
        await driver.run(_encoded(payload), doc, None)
        session.upload.assert_awaited_once_with(payload, doc.name)

    @pytest.mark.asyncio
    async def test_missing_payload_fails(self, driver, doc):
        with pytest.raises(DriverSetupError):
            await driver.run(_encoded(None), doc, None)

    @pytest.mark.asyncio
    async def test_upload_error_propagates(self, driver, session, doc):
        session.upload.side_effect = ConnectionError("refused")
        with pytest.raises(ConnectionError):
            await driver.run(_encoded(b"payload"), doc, None)

    @pytest.mark.asyncio
    async def test_cancel_delegates(self, driver, session):
        await driver.cancel()
        session.cancel.assert_awaited_once_with(False)


class TestPanelOnlyOperations:
    @pytest.mark.parametrize(
        "call",
        [
            lambda d: d.run_raw("G0 X0"),
            lambda d: d.set_hold(True),
            lambda d: d.home(None),
            lambda d: d.move_to(1.0, 2.0),
            lambda d: d.jog(1000, X=1.0),
            lambda d: d.clear_alarm(),
            lambda d: d.read_settings(),
            lambda d: d.write_setting("a", "b"),
            lambda d: d.set_power(Mock(), 0.5),
            lambda d: d.set_focus_power(Mock(), 0.5),
            lambda d: d.set_wcs_offset("MACHINE", 0, 0, None),
            lambda d: d.run_probe_cycle(Axis.X, 10, 100),
        ],
    )
    @pytest.mark.asyncio
    async def test_raise_not_implemented(self, driver, call):
        with pytest.raises(NotImplementedError):
            await call(driver)

    @pytest.mark.asyncio
    async def test_select_default_tool_is_allowed(self, driver, machine):
        await driver.select_tool(machine.get_default_head().tool_number)

    @pytest.mark.asyncio
    async def test_select_other_tool_fails(self, driver):
        with pytest.raises(NotImplementedError):
            await driver.select_tool(7)

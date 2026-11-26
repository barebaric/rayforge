"""Epilog driver tests.

The driver is a thin shell over raydriver's ``EpilogDriver``: it builds
the native driver from the setup vars, uploads the encoder's byte
stream, and reports every machine operation the print protocol cannot
perform.

The session is mocked with an ``AsyncMock`` for ``upload``/``cancel``;
the encoder itself is covered by raydriver's tests.
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from raygeo.ops import Ops

from rayforge.core.doc import Doc
from rayforge.core.varset import ChoiceVar, TextAreaVar
from rayforge.machine.driver.driver import (
    Axis,
    DriverPrecheckError,
    DriverSetupError,
)
from rayforge.machine.driver.epilog import (
    EpilogHelixDriver,
    EpilogZingDriver,
)
from rayforge.machine.driver.epilog.epilog_driver import DEFAULT_PORT
from rayforge.machine.models.laser import LaserHead
from rayforge.machine.models.machine import Machine, Origin
from rayforge.pipeline.artifact.encoded import (
    EncodedOutput,
    MachineCodeOpMap,
)


@pytest.fixture
def setup_args() -> dict:
    return {
        "host": "10.0.0.1",
        "port": DEFAULT_PORT,
        "resolution": "500",
        "frequency": 1000,
        "speed_reference": 6000,
    }


@pytest_asyncio.fixture
async def machine(lite_context, setup_args) -> Machine:
    m = Machine(lite_context)
    m.name = "Epilog"
    m.add_head(LaserHead())
    m.set_axis_extents(609.6, 304.8)
    m.driver_name = EpilogZingDriver.__name__
    m.driver_args = dict(setup_args)
    m.set_origin(Origin.TOP_LEFT)
    return m


@pytest.fixture
def doc() -> Doc:
    doc = Doc()
    doc.name = "Test Job"
    return doc


@pytest.fixture
def session() -> Mock:
    session = Mock()
    session.upload = AsyncMock()
    session.cancel = AsyncMock()
    session.read_parser_state = AsyncMock(return_value="MACHINE")
    session.read_wcs_offsets = AsyncMock(
        return_value={"MACHINE": (0.0, 0.0, 0.0)}
    )
    return session


@pytest.fixture
def driver(lite_context, machine, setup_args, session) -> EpilogZingDriver:
    driver = EpilogZingDriver(lite_context, machine)
    driver.setup(**setup_args)
    driver._session = session
    return driver


@pytest.fixture
def helix_setup_args(setup_args) -> dict:
    return {**setup_args, "resolution": "400"}


@pytest_asyncio.fixture
async def helix_machine(lite_context, helix_setup_args) -> Machine:
    m = Machine(lite_context)
    m.name = "Epilog Helix"
    m.add_head(LaserHead())
    m.set_axis_extents(609.6, 304.8)
    m.driver_name = EpilogHelixDriver.__name__
    m.driver_args = dict(helix_setup_args)
    m.set_origin(Origin.TOP_LEFT)
    return m


@pytest.fixture
def helix_driver(
    lite_context, helix_machine, helix_setup_args, session
) -> EpilogHelixDriver:
    driver = EpilogHelixDriver(lite_context, helix_machine)
    driver.setup(**helix_setup_args)
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


class TestModels:
    def test_labels_are_model_specific(self):
        assert "Zing" in EpilogZingDriver.label
        assert "Helix" in EpilogHelixDriver.label

    def test_zing_resolution_choices(self):
        var = EpilogZingDriver.get_setup_vars().get("resolution")
        assert isinstance(var, ChoiceVar)
        assert "500" in var.choices
        assert "75" not in var.choices

    def test_helix_resolution_choices(self):
        var = EpilogHelixDriver.get_setup_vars().get("resolution")
        assert isinstance(var, ChoiceVar)
        assert "400" in var.choices
        assert "500" not in var.choices

    def test_precheck_rejects_cross_model_resolution(self):
        with pytest.raises(DriverPrecheckError):
            EpilogHelixDriver.precheck(
                host="10.0.0.1", port=515, resolution="500", frequency=1000
            )
        with pytest.raises(DriverPrecheckError):
            EpilogZingDriver.precheck(
                host="10.0.0.1", port=515, resolution="75", frequency=1000
            )

    def test_encoder_spec_carries_the_model(self, machine, helix_machine, doc):
        zing = EpilogZingDriver.create_encoder_spec(machine, doc, Ops())
        helix = EpilogHelixDriver.create_encoder_spec(
            helix_machine, doc, Ops()
        )
        assert json.loads(zing.context_json)["model"] == "zing"
        assert json.loads(helix.context_json)["model"] == "helix"


class TestQueue:
    def test_default_is_empty(self):
        var = EpilogZingDriver.get_setup_vars().get("queue")
        assert var is not None
        assert var.default == ""

    def test_job_prefix_setting(self):
        keys = list(EpilogZingDriver.get_setup_vars().keys())
        assert "job_prefix" in keys
        var = EpilogZingDriver.get_setup_vars().get("job_prefix")
        assert var is not None
        assert var.default == ""
        assert var.optional is True
        assert keys.index("job_prefix") == keys.index("queue") + 1

    def test_autofocus_setting(self):
        keys = list(EpilogZingDriver.get_setup_vars().keys())
        assert "autofocus" in keys
        var = EpilogZingDriver.get_setup_vars().get("autofocus")
        assert var is not None
        assert var.default is False
        assert var.var_type is bool
        assert keys.index("autofocus") == keys.index("job_prefix") + 1

    def test_is_a_single_line_field(self):
        var = EpilogZingDriver.get_setup_vars().get("queue")
        assert var is not None
        assert not isinstance(var, TextAreaVar)

    def test_empty_queue_is_optional(self):
        var = EpilogZingDriver.get_setup_vars().get("queue")
        assert var is not None
        assert var.optional is True
        assert var.is_required({}) is False

    def test_empty_queue_is_preserved(self, lite_context, machine):
        driver = EpilogZingDriver(lite_context, machine)
        config = driver.build_config(
            {"host": "10.0.0.1", "queue": "", "port": 515}
        )
        assert config["queue"] == ""
        config = driver.build_config({"host": "10.0.0.1", "port": 515})
        assert config["queue"] == ""

    def test_named_queue_is_preserved(self, lite_context, machine):
        driver = EpilogZingDriver(lite_context, machine)
        config = driver.build_config(
            {"host": "10.0.0.1", "queue": "Engraver", "port": 515}
        )
        assert config["queue"] == "Engraver"


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


class TestRun:
    @pytest.mark.asyncio
    async def test_uploads_payload(self, driver, session, doc):
        payload = b"\x1b%-12345X@PJL JOB" + bytes(16)
        await driver.run(_encoded(payload), doc, Ops())
        session.upload.assert_awaited_once_with(payload, doc.name)

    @pytest.mark.asyncio
    async def test_uploads_payload_with_job_prefix(
        self, lite_context, machine, setup_args, session, doc
    ):
        driver = EpilogZingDriver(lite_context, machine)
        driver.setup(**{**setup_args, "job_prefix": "lab-"})
        driver._session = session
        payload = b"\x1b%-12345X@PJL JOB" + bytes(16)
        await driver.run(_encoded(payload), doc, Ops())
        session.upload.assert_awaited_once_with(payload, f"lab-{doc.name}")

    @pytest.mark.asyncio
    async def test_missing_payload_fails(self, driver, doc):
        with pytest.raises(DriverSetupError):
            await driver.run(_encoded(None), doc, Ops())

    @pytest.mark.asyncio
    async def test_upload_error_propagates(self, driver, session, doc):
        session.upload.side_effect = ConnectionError("refused")
        with pytest.raises(ConnectionError):
            await driver.run(_encoded(b"payload"), doc, Ops())

    @pytest.mark.asyncio
    async def test_cancel_delegates(self, driver, session):
        await driver.cancel()
        session.cancel.assert_awaited_once_with(False)

    @pytest.mark.asyncio
    async def test_frame_uploads_beam_off_outline(self, driver, session, doc):
        corners = [
            (0.0, 0.0, None),
            (10.0, 0.0, None),
            (10.0, 10.0, None),
            (0.0, 10.0, None),
            (0.0, 0.0, None),
        ]
        await driver.frame(corners, 3000.0, doc)

        session.upload.assert_awaited_once()
        payload, title = session.upload.await_args.args
        assert title == doc.name
        assert payload.startswith(b"\x1b%-12345X@PJL JOB")
        assert b"PD" in payload
        assert b"YP000;" in payload

    @pytest.mark.asyncio
    async def test_helix_frame_uploads_beam_off_outline(
        self, helix_driver, session, doc
    ):
        corners = [
            (0.0, 0.0, None),
            (10.0, 0.0, None),
            (10.0, 10.0, None),
            (0.0, 10.0, None),
            (0.0, 0.0, None),
        ]
        await helix_driver.frame(corners, 3000.0, doc)

        session.upload.assert_awaited_once()
        payload, title = session.upload.await_args.args
        assert title == doc.name
        assert payload.startswith(b"\x1b%-12345X@PJL JOB")
        assert b"PD" in payload
        assert b"YP000;" in payload


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
            lambda d: d.select_wcs("G54"),
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

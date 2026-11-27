from typing import Any, cast

import pytest
import pytest_asyncio
from raygeo.driver.octoprint.emulator import OctoPrintEmulator

from rayforge.machine.driver.octoprint import OctoPrintDriver
from rayforge.machine.driver.octoprint.octoprint_util import (
    build_octoprint_profile,
)

API_KEY = "test-api-key"


@pytest_asyncio.fixture
async def open_device():
    """An OctoPrint server with access control disabled."""
    emulator = OctoPrintEmulator(require_auth=False)
    await emulator.start()
    try:
        yield emulator
    finally:
        await emulator.stop()


class TestBuildOctoPrintProfile:
    def test_full_profile(self):
        version_info = {"server": "1.9.0", "version": "1.9.0"}
        printer_info = {
            "state": {"text": "Operational", "flags": {}},
            "printer": {"name": "My Prusa"},
        }
        profile, warnings = build_octoprint_profile(version_info, printer_info)
        assert profile.meta.name == "My Prusa"
        dc = profile.machine_config.driver_config
        assert dc is not None
        assert dc["server_version"] == "1.9.0"
        assert warnings == []

    def test_transient_state_does_not_leak_into_name(self):
        version_info = {"version": "1.8.1"}
        printer_info = {
            "state": {"text": "Heating bed", "flags": {}},
            "printer": {},
        }
        profile, _ = build_octoprint_profile(version_info, printer_info)
        assert profile.meta.name == "OctoPrint"

    def test_no_printer_name_keeps_default(self):
        printer_info = {
            "state": {"text": "Operational", "flags": {}},
            "printer": {},
        }
        profile, _ = build_octoprint_profile({"version": "1"}, printer_info)
        assert profile.meta.name == "OctoPrint"

    def test_dimensions_populate_axis_extents(self):
        version_info = {"version": "1.9.0"}
        printer_info = {
            "state": {"text": "Operational", "flags": {}},
            "printer": {},
            "dimensions": {"x_length": 603, "y_length": 402},
        }
        profile, _ = build_octoprint_profile(version_info, printer_info)
        assert profile.machine_config.axis_extents == (603.0, 402.0)

    def test_incomplete_or_invalid_dimensions_ignored(self):
        for dimensions in (
            {"x_length": 603},
            {"x_length": "wide", "y_length": 402},
            {"x_length": -1, "y_length": 402},
            [603, 402],
        ):
            printer_info = {
                "state": {},
                "printer": {},
                "dimensions": dimensions,
            }
            profile, _ = build_octoprint_profile(None, printer_info)
            assert profile.machine_config.axis_extents is None

    def test_malformed_printer_section_never_raises(self):
        version_info = {"version": "1.9.0"}
        printer_info = {
            "state": "Operational",
            "printer": None,
            "dimensions": [603, 402],
        }
        profile, warnings = build_octoprint_profile(version_info, printer_info)
        assert profile.meta.name == "OctoPrint"
        assert profile.machine_config.axis_extents is None
        assert warnings == []

    def test_no_version_info_warns(self):
        profile, warnings = build_octoprint_profile(None, None)
        assert profile.meta.name == "OctoPrint"
        assert profile.machine_config.driver_config is None
        assert len(warnings) == 1
        assert "version" in warnings[0].lower()

    def test_version_without_version_key(self):
        profile, _ = build_octoprint_profile({"server": "1.0"}, None)
        assert profile.machine_config.driver_config is None

    def test_non_dict_payloads_never_raise(self):
        # A malformed JSON body (wrong shape) must not raise; it is
        # simply treated as missing data.
        version_info = cast("dict[str, Any] | None", ["junk"])
        profile, warnings = build_octoprint_profile(
            version_info, {"printer": 42}
        )
        assert profile.meta.name == "OctoPrint"
        assert profile.machine_config.axis_extents is None
        assert len(warnings) == 0


class TestDriverProbe:
    @pytest.mark.asyncio
    async def test_probe_without_api_key(
        self, context_initializer, open_device
    ):
        profile, warnings = await OctoPrintDriver.probe(
            context_initializer,
            host="127.0.0.1",
            port=open_device.http_port,
            path="/",
        )

        assert profile.meta.name == "TestPrinter"
        assert profile.machine_config.driver == "OctoPrintDriver"
        assert profile.machine_config.driver_args == {
            "host": "127.0.0.1",
            "port": open_device.http_port,
            "path": "/",
        }
        dc = profile.machine_config.driver_config
        assert dc is not None
        assert dc["server_version"] == "1.9.3"
        assert warnings == []

    @pytest.mark.asyncio
    async def test_probe_connection_failure(self, context_initializer):
        """When the server is unreachable, the probe returns a
        minimal profile with a warning instead of raising."""
        emulator = OctoPrintEmulator(require_auth=False)
        await emulator.start()
        port = emulator.http_port
        await emulator.stop()

        profile, warnings = await OctoPrintDriver.probe(
            context_initializer,
            host="127.0.0.1",
            port=port,
        )

        assert profile.meta.name == "OctoPrint"
        assert profile.machine_config.driver_config is None
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_probe_with_path_prefix(self, context_initializer):
        emulator = OctoPrintEmulator(
            require_auth=False, path_prefix="/octoprint"
        )
        await emulator.start()
        try:
            profile, warnings = await OctoPrintDriver.probe(
                context_initializer,
                host="127.0.0.1",
                port=emulator.http_port,
                path="/octoprint/",
            )
        finally:
            await emulator.stop()

        assert profile.meta.name == "TestPrinter"
        driver_args = profile.machine_config.driver_args
        assert driver_args is not None
        assert driver_args["path"] == "/octoprint/"
        assert profile.machine_config.driver_config == {
            "server_version": "1.9.3"
        }
        assert warnings == []

import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from rayforge.machine.driver.octoprint import OctoPrintDriver
from rayforge.machine.driver.octoprint.octoprint_util import (
    build_octoprint_profile,
)

_AIOHTTP_PATCH = (
    "rayforge.machine.driver.octoprint.octoprint_driver.aiohttp.ClientSession"
)


def _mock_aiohttp_session(responses: dict[str, Any]):
    """Builds a mock aiohttp.ClientSession that returns canned JSON
    per request URL path. ``responses`` maps path substrings to return
    values (dicts). A path not in the map returns 200 with an empty
    dict. Paths mapped to ``None`` return a 403 (auth failure)."""

    def _make_response(url):
        for path, data in responses.items():
            if path in url:
                if data is None:
                    resp = AsyncMock()
                    resp.status = 403
                    resp.raise_for_status = MagicMock(
                        side_effect=Exception("forbidden")
                    )
                    ctx = AsyncMock()
                    ctx.__aenter__ = AsyncMock(return_value=resp)
                    ctx.__aexit__ = AsyncMock(return_value=False)
                    return ctx
                resp = AsyncMock()
                resp.status = 200
                resp.content_type = "application/json"
                resp.raise_for_status = MagicMock()
                resp.json = AsyncMock(return_value=data)
                resp.text = AsyncMock(return_value=json.dumps(data))
                ctx = AsyncMock()
                ctx.__aenter__ = AsyncMock(return_value=resp)
                ctx.__aexit__ = AsyncMock(return_value=False)
                return ctx
        # Default: empty 200
        resp = AsyncMock()
        resp.status = 200
        resp.content_type = "application/json"
        resp.raise_for_status = MagicMock()
        resp.json = AsyncMock(return_value={})
        resp.text = AsyncMock(return_value="{}")
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.request = MagicMock(
        side_effect=lambda method, url, **kw: _make_response(url)
    )
    return mock_session


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
    async def test_probe_without_api_key(self, context_initializer, mocker):
        responses = {
            "/api/version": {"server": "1.9.0", "version": "1.9.0"},
            "/api/printer": {
                "state": {"text": "Operational", "flags": {}},
                "printer": {"name": "Workshop Printer"},
            },
        }
        mocker.patch(
            _AIOHTTP_PATCH, return_value=_mock_aiohttp_session(responses)
        )

        profile, warnings = await OctoPrintDriver.probe(
            context_initializer,
            host="octoprint.local",
            port=80,
            path="/",
        )

        assert profile.meta.name == "Workshop Printer"
        assert profile.machine_config.driver == "OctoPrintDriver"
        assert profile.machine_config.driver_args == {
            "host": "octoprint.local",
            "port": 80,
            "path": "/",
        }
        dc = profile.machine_config.driver_config
        assert dc is not None
        assert dc["server_version"] == "1.9.0"
        assert warnings == []

    @pytest.mark.asyncio
    async def test_probe_connection_failure(self, context_initializer, mocker):
        """When the server is unreachable, the probe returns a
        minimal profile with a warning instead of raising."""
        responses = {"/api/version": None}
        mocker.patch(
            _AIOHTTP_PATCH, return_value=_mock_aiohttp_session(responses)
        )

        profile, warnings = await OctoPrintDriver.probe(
            context_initializer,
            host="unreachable.local",
            port=80,
        )

        assert profile.meta.name == "OctoPrint"
        assert profile.machine_config.driver_config is None
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_probe_with_path_prefix(self, context_initializer, mocker):
        responses = {
            "/api/version": {"version": "1.8.0"},
            "/api/printer": {
                "state": {"text": "Operational", "flags": {}},
                "printer": {},
            },
        }
        mocker.patch(
            _AIOHTTP_PATCH, return_value=_mock_aiohttp_session(responses)
        )

        profile, _ = await OctoPrintDriver.probe(
            context_initializer,
            host="host",
            port=80,
            path="/octoprint",
        )

        assert profile.machine_config.driver_args == {
            "host": "host",
            "port": 80,
            "path": "/octoprint",
        }
        assert profile.machine_config.driver_config == {
            "server_version": "1.8.0"
        }

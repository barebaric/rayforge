import json

import pytest

from rayforge.core.varset import AppKeyVar, HostnameVar, PortVar
from rayforge.machine.driver.driver import (
    DriverMaturity,
    DriverPrecheckError,
)
from rayforge.machine.driver.octoprint import OctoPrintDriver
from rayforge.machine.driver.octoprint.octoprint_driver import (
    _extract_api_key,
    _normalize_path_prefix,
)

API_KEY = json.dumps({"api_key": "mykey"})


@pytest.fixture
def driver(context_initializer, machine):
    return OctoPrintDriver(context_initializer, machine)


@pytest.fixture
def setup_driver(driver):
    driver.setup(host="octoprint.local", port=80, api_key=API_KEY)
    return driver


class TestClassAttributes:
    def test_label(self):
        assert OctoPrintDriver.label == "OctoPrint"

    def test_subtitle(self):
        assert "OctoPrint" in OctoPrintDriver.subtitle

    def test_maturity(self):
        assert OctoPrintDriver.maturity == DriverMaturity.STABLE

    def test_uses_gcode(self):
        assert OctoPrintDriver.uses_gcode is True

    def test_reports_granular_progress(self):
        assert OctoPrintDriver.reports_granular_progress is False

    def test_supports_settings(self):
        assert OctoPrintDriver.supports_settings is False

    def test_mdns_services(self):
        spec = OctoPrintDriver.DISCOVERY
        assert spec is not None and spec.mdns is not None
        assert spec.mdns.services == ("_octoprint._tcp.local.",)


class TestProperties:
    def test_machine_space_wcs(self, setup_driver):
        assert setup_driver.machine_space_wcs == "G53"

    def test_machine_space_wcs_display_name(self, setup_driver):
        assert "G53" in setup_driver.machine_space_wcs_display_name

    def test_resource_uri(self, setup_driver):
        assert setup_driver.resource_uri == "tcp://octoprint.local:80"

    def test_resource_uri_no_host(self, driver):
        assert driver.resource_uri is None


class TestPrecheck:
    def test_valid_hostname(self):
        OctoPrintDriver.precheck(host="octoprint.local")

    def test_valid_ip(self):
        OctoPrintDriver.precheck(host="192.168.1.100")

    def test_empty_hostname_ok(self):
        OctoPrintDriver.precheck(host="")

    def test_invalid_hostname_raises(self):
        with pytest.raises(DriverPrecheckError):
            OctoPrintDriver.precheck(host="!!!invalid!!!")

    def test_none_hostname_ok(self):
        OctoPrintDriver.precheck(host=None)


class TestGetSetupVars:
    def test_returns_varset(self):
        vs = OctoPrintDriver.get_setup_vars()
        keys = list(vs.keys())
        assert "host" in keys
        assert "port" in keys
        assert "api_key" in keys
        assert isinstance(vs.get("host"), HostnameVar)
        assert isinstance(vs.get("port"), PortVar)
        assert isinstance(vs.get("api_key"), AppKeyVar)


class TestExtractApiKey:
    def test_from_appkey_json(self):
        data = json.dumps({"api_key": "abc123"})
        assert _extract_api_key(data) == "abc123"

    def test_from_oauth_json_backward_compat(self):
        data = json.dumps({"access_token": "abc123", "expires_at": None})
        assert _extract_api_key(data) == "abc123"

    def test_api_key_takes_precedence(self):
        data = json.dumps({"api_key": "new", "access_token": "old"})
        assert _extract_api_key(data) == "new"

    def test_raw_key_string(self):
        assert _extract_api_key("raw-key-xyz") == "raw-key-xyz"

    def test_empty_string(self):
        assert _extract_api_key("") is None

    def test_none(self):
        assert _extract_api_key(None) is None

    def test_whitespace_stripped(self):
        assert _extract_api_key("  key  ") == "key"


class TestPathPrefix:
    def test_root_is_empty(self):
        assert _normalize_path_prefix("/") == ""

    def test_empty_is_empty(self):
        assert _normalize_path_prefix("") == ""

    def test_custom_prefix_kept(self):
        assert _normalize_path_prefix("/octoprint") == "/octoprint"

    def test_trailing_slash_stripped(self):
        assert _normalize_path_prefix("/octoprint/") == "/octoprint"


class TestSetup:
    def test_setup_success(self, driver):
        driver.setup(
            host="192.168.1.50",
            port=5000,
            api_key=json.dumps({"api_key": "mykey"}),
        )
        assert driver.did_setup is True
        assert driver.resource_uri == "tcp://192.168.1.50:5000"

    def test_setup_no_host_raises(self, driver):
        driver.setup(host="", port=80, api_key=API_KEY)
        assert driver.state.error is not None
        assert "Hostname" in driver.state.error.title

    def test_setup_no_api_key_raises(self, driver):
        driver.setup(host="x", port=80, api_key="")
        assert driver.state.error is not None
        assert "API key" in driver.state.error.title

    def test_setup_raw_api_key(self, driver):
        driver.setup(host="x", port=80, api_key="plain-key")
        assert driver.did_setup is True

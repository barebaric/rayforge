import json

import pytest

from rayforge.core.varset import VarSet, Var
from rayforge.core.varset.appkeyvar import AppKeyVar


class TestAppKeyVarCreation:
    def test_creation_defaults(self):
        var = AppKeyVar(key="ak", label="API Key", app_name="TestApp")
        assert var.key == "ak"
        assert var.label == "API Key"
        assert var.var_type is str
        assert var.value == ""
        assert var.app_name == "TestApp"
        assert var.probe_url is None
        assert var.request_url is None
        assert var.poll_url is None

    def test_creation_with_urls(self):
        var = AppKeyVar(
            key="ak",
            label="Key",
            app_name="RayForge",
            probe_url="http://{host}:{port}/probe",
            request_url="http://{host}:{port}/request",
            poll_url="http://{host}:{port}/poll/{app_token}",
        )
        assert var.probe_url == "http://{host}:{port}/probe"
        assert var.request_url == "http://{host}:{port}/request"
        assert var.poll_url == "http://{host}:{port}/poll/{app_token}"


class TestGetApiKey:
    def test_empty_value(self):
        var = AppKeyVar(key="ak", label="Key", app_name="App")
        assert var.get_api_key() is None
        assert not var.has_key()

    def test_json_api_key(self):
        var = AppKeyVar(
            key="ak",
            label="Key",
            app_name="App",
            value=json.dumps({"api_key": "my-secret-key"}),
        )
        assert var.get_api_key() == "my-secret-key"
        assert var.has_key()

    def test_plain_string(self):
        var = AppKeyVar(
            key="ak",
            label="Key",
            app_name="App",
            value="raw-key-123",
        )
        assert var.get_api_key() == "raw-key-123"
        assert var.has_key()

    def test_invalid_json(self):
        var = AppKeyVar(
            key="ak",
            label="Key",
            app_name="App",
            value="not-json",
        )
        assert var.get_api_key() == "not-json"

    def test_json_without_api_key(self):
        var = AppKeyVar(
            key="ak",
            label="Key",
            app_name="App",
            value=json.dumps({"other": "data"}),
        )
        assert var.get_api_key() is None
        assert not var.has_key()

    def test_whitespace_string(self):
        var = AppKeyVar(
            key="ak",
            label="Key",
            app_name="App",
            value="  ",
        )
        assert not var.get_api_key()
        assert not var.has_key()


class TestResolveConfig:
    def test_resolve_with_sibling_vars(self):
        var = AppKeyVar(
            key="ak",
            label="Key",
            app_name="RayForge",
            probe_url="http://{host}:{port}/probe",
            request_url="http://{host}:{port}/request",
            poll_url="http://{host}:{port}/poll/{{app_token}}",
        )
        host_var = Var(
            key="host",
            label="Host",
            var_type=str,
            value="octoprint.local",
        )
        port_var = Var(
            key="port",
            label="Port",
            var_type=str,
            value="80",
        )
        VarSet(vars=[host_var, port_var, var])
        config = var.resolve_config()
        assert config["probe_url"] == ("http://octoprint.local:80/probe")
        assert config["request_url"] == ("http://octoprint.local:80/request")
        assert config["poll_url"] == (
            "http://octoprint.local:80/poll/{app_token}"
        )

    def test_resolve_raises_on_unknown_placeholder(self):
        var = AppKeyVar(
            key="ak",
            label="Key",
            app_name="App",
            request_url="http://{host}/req/{unknown}",
        )
        host_var = Var(
            key="host",
            label="Host",
            var_type=str,
            value="myhost",
        )
        VarSet(vars=[host_var, var])
        with pytest.raises(KeyError):
            var.resolve_config()

    def test_resolve_no_siblings(self):
        var = AppKeyVar(
            key="ak",
            label="Key",
            app_name="App",
            request_url="http://fixed.host/request",
        )
        config = var.resolve_config()
        assert config["request_url"] == ("http://fixed.host/request")

    def test_resolve_none_urls(self):
        var = AppKeyVar(
            key="ak",
            label="Key",
            app_name="App",
        )
        config = var.resolve_config()
        assert config["probe_url"] is None
        assert config["request_url"] is None
        assert config["poll_url"] is None


class TestSerialization:
    def test_to_dict_without_value(self):
        var = AppKeyVar(
            key="ak",
            label="Key",
            app_name="RayForge",
            probe_url="http://{host}/probe",
        )
        data = var.to_dict()
        assert data["key"] == "ak"
        assert data["class"] == "AppKeyVar"
        assert data["app_name"] == "RayForge"
        assert data["probe_url"] == "http://{host}/probe"
        assert "value" not in data

    def test_to_dict_with_value(self):
        var = AppKeyVar(
            key="ak",
            label="Key",
            app_name="App",
            value=json.dumps({"api_key": "k"}),
        )
        data = var.to_dict(include_value=True)
        assert data["value"] == json.dumps({"api_key": "k"})

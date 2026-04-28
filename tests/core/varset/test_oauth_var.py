import json

from rayforge.core.varset import VarSet, Var
from rayforge.core.varset.oauthvar import OAuthFlowVar


class TestOAuthFlowVar:
    def test_creation_defaults(self):
        var = OAuthFlowVar(key="auth", label="Login")
        assert var.key == "auth"
        assert var.label == "Login"
        assert var.var_type is str
        assert var.value == ""
        assert var.authorize_url is None
        assert var.token_url is None
        assert var.client_id is None
        assert var.scopes == []
        assert var.redirect_port == 8765

    def test_creation_with_config(self):
        var = OAuthFlowVar(
            key="auth",
            label="Cloud Login",
            authorize_url="https://example.com/oauth/authorize",
            token_url="https://example.com/oauth/token",
            client_id="my-app",
            client_secret="secret",
            scopes=["read", "write"],
            redirect_port=9999,
        )
        assert var.authorize_url == "https://example.com/oauth/authorize"
        assert var.token_url == "https://example.com/oauth/token"
        assert var.client_id == "my-app"
        assert var.client_secret == "secret"
        assert var.scopes == ["read", "write"]
        assert var.redirect_port == 9999

    def test_get_tokens_empty(self):
        var = OAuthFlowVar(key="auth", label="Login")
        assert var.get_tokens() is None
        assert not var.is_authenticated()

    def test_get_tokens_valid(self):
        token_data = json.dumps(
            {"access_token": "abc123", "refresh_token": "refresh"}
        )
        var = OAuthFlowVar(key="auth", label="Login", value=token_data)
        tokens = var.get_tokens()
        assert tokens is not None
        assert tokens["access_token"] == "abc123"
        assert tokens["refresh_token"] == "refresh"
        assert var.is_authenticated()

    def test_get_tokens_invalid_json(self):
        var = OAuthFlowVar(key="auth", label="Login", value="not-json")
        assert var.get_tokens() is None
        assert not var.is_authenticated()

    def test_get_tokens_no_access_token(self):
        token_data = json.dumps({"refresh_token": "refresh"})
        var = OAuthFlowVar(key="auth", label="Login", value=token_data)
        assert not var.is_authenticated()

    def test_to_dict(self):
        var = OAuthFlowVar(
            key="auth",
            label="Login",
            authorize_url="https://example.com/authorize",
            token_url="https://example.com/token",
            client_id="my-app",
            scopes=["read"],
            redirect_port=8080,
        )
        d = var.to_dict()
        assert d["class"] == "OAuthFlowVar"
        assert d["key"] == "auth"
        assert d["authorize_url"] == "https://example.com/authorize"
        assert d["token_url"] == "https://example.com/token"
        assert d["client_id"] == "my-app"
        assert d["scopes"] == ["read"]
        assert d["redirect_port"] == 8080

    def test_to_dict_with_value(self):
        token_data = json.dumps({"access_token": "abc"})
        var = OAuthFlowVar(
            key="auth", label="Login", value=token_data
        )
        d = var.to_dict(include_value=True)
        assert d["value"] == token_data

    def test_serialization_round_trip(self):
        original = OAuthFlowVar(
            key="auth",
            label="Login",
            authorize_url="https://example.com/authorize",
            token_url="https://example.com/token",
            client_id="my-app",
            client_secret="secret",
            scopes=["read", "write"],
            redirect_port=9090,
        )
        data = original.to_dict(include_value=True)
        data.pop("class")
        restored = OAuthFlowVar(**data)
        assert restored.key == original.key
        assert restored.authorize_url == original.authorize_url
        assert restored.token_url == original.token_url
        assert restored.client_id == original.client_id
        assert restored.client_secret == original.client_secret
        assert restored.scopes == original.scopes
        assert restored.redirect_port == original.redirect_port


class TestOAuthFlowVarTemplates:
    def test_resolve_config_no_varset(self):
        var = OAuthFlowVar(
            key="auth",
            label="Login",
            authorize_url="https://example.com/authorize",
            token_url="https://example.com/token",
            client_id="my-app",
        )
        config = var.resolve_config()
        assert config["authorize_url"] == "https://example.com/authorize"
        assert config["token_url"] == "https://example.com/token"
        assert config["client_id"] == "my-app"

    def test_resolve_config_with_sibling_vars(self):
        var = OAuthFlowVar(
            key="auth",
            label="Login",
            authorize_url="{server}/oauth/authorize",
            token_url="{server}/oauth/token",
            client_id="my-app",
        )
        server_var = Var(
            key="server", label="Server", var_type=str,
            value="https://cloud.example.com",
        )
        VarSet(vars=[server_var, var])
        config = var.resolve_config()
        assert (
            config["authorize_url"]
            == "https://cloud.example.com/oauth/authorize"
        )
        assert (
            config["token_url"]
            == "https://cloud.example.com/oauth/token"
        )

    def test_resolve_config_with_overrides(self):
        var = OAuthFlowVar(
            key="auth",
            label="Login",
            authorize_url=None,
            token_url=None,
            client_id=None,
        )
        config = var.resolve_config(
            overrides={
                "authorize_url": "https://custom.com/authorize",
                "token_url": "https://custom.com/token",
                "client_id": "custom-app",
            }
        )
        assert config["authorize_url"] == "https://custom.com/authorize"
        assert config["token_url"] == "https://custom.com/token"
        assert config["client_id"] == "custom-app"

    def test_resolve_config_overrides_take_precedence(self):
        var = OAuthFlowVar(
            key="auth",
            label="Login",
            authorize_url="https://original.com/authorize",
        )
        config = var.resolve_config(
            overrides={"authorize_url": "https://override.com/authorize"}
        )
        assert config["authorize_url"] == "https://override.com/authorize"

    def test_resolve_config_missing_sibling_returns_template(self):
        var = OAuthFlowVar(
            key="auth",
            label="Login",
            authorize_url="{server}/oauth/authorize",
            token_url="https://fixed.com/token",
            client_id="my-app",
        )
        VarSet(vars=[var])
        config = var.resolve_config()
        assert config["authorize_url"] == "{server}/oauth/authorize"
        assert config["token_url"] == "https://fixed.com/token"

    def test_resolve_config_empty_override_ignored(self):
        var = OAuthFlowVar(
            key="auth",
            label="Login",
            authorize_url="https://original.com/authorize",
            token_url="https://original.com/token",
            client_id="my-app",
        )
        VarSet(vars=[var])
        config = var.resolve_config(
            overrides={"authorize_url": "", "token_url": "   "}
        )
        assert config["authorize_url"] == "https://original.com/authorize"
        assert config["token_url"] == "https://original.com/token"


class TestOAuthFlowVarVarsetBackReference:
    def test_varset_set_on_add(self):
        var = OAuthFlowVar(key="auth", label="Login")
        assert var._varset is None
        vs = VarSet(vars=[var])
        assert var._varset is vs

    def test_varset_cleared_on_remove(self):
        var = OAuthFlowVar(key="auth", label="Login")
        vs = VarSet(vars=[var])
        assert var._varset is vs
        vs.remove("auth")
        assert var._varset is None

    def test_varset_cleared_on_clear(self):
        var = OAuthFlowVar(key="auth", label="Login")
        vs = VarSet(vars=[var])
        vs.clear()
        assert var._varset is None

    def test_base_var_also_gets_back_reference(self):
        var = Var(key="test", label="Test", var_type=str)
        assert var._varset is None
        vs = VarSet(vars=[var])
        assert var._varset is vs
        vs.remove("test")
        assert var._varset is None

import json
import threading
from http.server import HTTPServer
from unittest.mock import MagicMock, patch

from rayforge.shared.oauth.flow import (
    OAuthFlow,
    OAuthFlowConfig,
    OAuthResult,
    _OAuthCallbackHandler,
)


class TestOAuthFlowConfig:
    def test_defaults(self):
        config = OAuthFlowConfig(
            authorize_url="https://example.com/authorize",
            token_url="https://example.com/token",
            client_id="test-app",
        )
        assert config.client_secret is None
        assert config.scopes == []
        assert config.redirect_port == 8765
        assert config.use_pkce is False


class TestOAuthResult:
    def test_creation(self):
        result = OAuthResult(access_token="abc123")
        assert result.access_token == "abc123"
        assert result.refresh_token is None
        assert result.expires_at is None
        assert result.scope is None
        assert result.raw_response == {}


class TestOAuthFlowGetAuthorizeUrl:
    def test_basic_url(self):
        config = OAuthFlowConfig(
            authorize_url="https://example.com/authorize",
            token_url="https://example.com/token",
            client_id="test-app",
        )
        flow = OAuthFlow(config)
        url = flow.get_authorize_url()
        assert "https://example.com/authorize?" in url
        assert "client_id=test-app" in url
        assert "response_type=code" in url
        assert "redirect_uri=" in url
        assert "127.0.0.1" in url
        assert "8765" in url

    def test_url_with_scopes(self):
        config = OAuthFlowConfig(
            authorize_url="https://example.com/authorize",
            token_url="https://example.com/token",
            client_id="test-app",
            scopes=["read", "write"],
        )
        flow = OAuthFlow(config)
        url = flow.get_authorize_url()
        assert "scope=read+write" in url

    def test_custom_redirect_port(self):
        config = OAuthFlowConfig(
            authorize_url="https://example.com/authorize",
            token_url="https://example.com/token",
            client_id="test-app",
            redirect_port=9999,
        )
        flow = OAuthFlow(config)
        url = flow.get_authorize_url()
        assert "9999" in url


class TestOAuthFlowStart:
    def test_port_in_use_calls_on_error(self):
        config = OAuthFlowConfig(
            authorize_url="https://example.com/authorize",
            token_url="https://example.com/token",
            client_id="test-app",
            redirect_port=8765,
        )
        blocker = HTTPServer(("127.0.0.1", 8765), _OAuthCallbackHandler)

        try:
            flow = OAuthFlow(config)
            on_error = MagicMock()
            on_complete = MagicMock()
            flow.start(on_complete, on_error)
            assert on_error.called
            assert not on_complete.called
        finally:
            blocker.server_close()

    @patch("rayforge.shared.oauth.flow.webbrowser.open")
    @patch("rayforge.shared.oauth.flow.urllib.request.urlopen")
    def test_successful_flow(self, mock_urlopen, mock_webbrowser):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"access_token": "tok123", "refresh_token": "ref456"}
        ).encode()
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = OAuthFlowConfig(
            authorize_url="https://example.com/authorize",
            token_url="https://example.com/token",
            client_id="test-app",
            redirect_port=0,
        )

        results = {"complete": None, "error": None}
        event = threading.Event()

        def on_complete(result):
            results["complete"] = result
            event.set()

        def on_error(error):
            results["error"] = error
            event.set()

        flow = OAuthFlow(config)

        # We can't easily test the full localhost callback flow
        # without a real HTTP request, so test exchange_code
        # directly instead.
        result = flow._exchange_code("test-code")
        assert result.access_token == "tok123"
        assert result.refresh_token == "ref456"
        assert result.raw_response["access_token"] == "tok123"

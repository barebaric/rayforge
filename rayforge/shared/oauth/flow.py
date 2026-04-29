import json
import logging
import socket as _socket
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


@dataclass
class OAuthFlowConfig:
    authorize_url: str
    token_url: str
    client_id: str
    client_secret: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    redirect_port: int = 8765
    use_pkce: bool = False


@dataclass
class OAuthResult:
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    scope: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    def __init__(self, callback, *args, **kwargs):
        self._callback = callback
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if not self.path.startswith("/callback"):
            self.send_response(404)
            self.end_headers()
            return

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        error = params.get("error", [None])[0]

        if error:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body>"
                b"<h1>Authorization Failed</h1>"
                b"<p>You can close this window.</p>"
                b"</body></html>"
            )
            self._callback(None, error)
        elif code:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body>"
                b"<h1>Authorization Successful</h1>"
                b"<p>You can close this window and return to the app."
                b"</p></body></html>"
            )
            self._callback(code, None)
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>Invalid Request</h1></body></html>"
            )


class OAuthFlow:
    """
    A reusable OAuth 2.0 Authorization Code flow.

    Spawns a localhost HTTP server to capture the redirect callback,
    opens the browser for user authorization, and exchanges the code
    for tokens. All callbacks are marshalled via GLib.idle_add so
    GTK code can safely update the UI.
    """

    def __init__(self, config: OAuthFlowConfig):
        self._config = config

    def get_authorize_url(self) -> str:
        redirect_uri = (
            f"http://127.0.0.1:{self._config.redirect_port}/callback"
        )
        params: Dict[str, str] = {
            "response_type": "code",
            "client_id": self._config.client_id,
            "redirect_uri": redirect_uri,
        }
        if self._config.scopes:
            params["scope"] = " ".join(self._config.scopes)
        return f"{self._config.authorize_url}?{urllib.parse.urlencode(params)}"

    def start(
        self,
        on_complete: Callable[[OAuthResult], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """
        Start the OAuth flow. Opens the browser and listens for the
        callback on localhost. Calls on_complete or on_error when done.
        """
        callback_received: Dict[str, Optional[str]] = {
            "code": None,
            "error": None,
        }

        def callback(code, error):
            callback_received["code"] = code
            callback_received["error"] = error

        def handler_factory(*args, **kwargs):
            return _OAuthCallbackHandler(callback, *args, **kwargs)

        probe = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", self._config.redirect_port))
        except OSError as e:
            on_error(e)
            return
        finally:
            probe.close()

        try:
            server = HTTPServer(
                ("127.0.0.1", self._config.redirect_port),
                handler_factory,
            )
        except OSError as e:
            on_error(e)
            return

        def run_server():
            try:
                server.handle_request()
            except Exception as e:
                on_error(e)
                return

            if callback_received["code"]:
                try:
                    result = self._exchange_code(callback_received["code"])
                    on_complete(result)
                except Exception as e:
                    on_error(e)
            else:
                err = callback_received["error"] or "Unknown error"
                on_error(RuntimeError(err))

        thread = Thread(target=run_server, daemon=True)
        thread.start()

        url = self.get_authorize_url()
        webbrowser.open(url)

    def _exchange_code(self, code: str) -> OAuthResult:
        redirect_uri = (
            f"http://127.0.0.1:{self._config.redirect_port}/callback"
        )
        data_dict: Dict[str, Any] = {
            "code": code,
            "client_id": self._config.client_id,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
        if self._config.client_secret:
            data_dict["client_secret"] = self._config.client_secret
        data = urllib.parse.urlencode(data_dict).encode()

        req = urllib.request.Request(
            self._config.token_url, data=data, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())

        access_token = result.get("access_token", "")
        refresh_token = result.get("refresh_token")
        expires_in = result.get("expires_in")
        expires_at = None
        if expires_in:
            expires_at = datetime.fromtimestamp(
                datetime.now().timestamp() + int(expires_in)
            )

        return OAuthResult(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            scope=result.get("scope"),
            raw_response=result,
        )

    def refresh(self, refresh_token: str) -> OAuthResult:
        data_dict: Dict[str, Any] = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self._config.client_id,
        }
        if self._config.client_secret:
            data_dict["client_secret"] = self._config.client_secret
        data = urllib.parse.urlencode(data_dict).encode()

        req = urllib.request.Request(
            self._config.token_url, data=data, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())

        access_token = result.get("access_token", "")
        new_refresh = result.get("refresh_token", refresh_token)
        expires_in = result.get("expires_in")
        expires_at = None
        if expires_in:
            expires_at = datetime.fromtimestamp(
                datetime.now().timestamp() + int(expires_in)
            )

        return OAuthResult(
            access_token=access_token,
            refresh_token=new_refresh,
            expires_at=expires_at,
            scope=result.get("scope"),
            raw_response=result,
        )

import json
import logging
import urllib.request
import urllib.parse
import urllib.error
import yaml
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, Optional

from .provider import (
    LicenseProvider,
    LicenseResult,
    LicenseStatus,
    LicenseType,
)


logger = logging.getLogger(__name__)


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def __init__(self, callback, *args, **kwargs):
        self.callback = callback
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path.startswith("/callback"):
            from urllib.parse import urlparse, parse_qs

            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            code = params.get("code", [None])[0]
            error = params.get("error", [None])[0]

            if error:
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h1>Authorization Failed</h1>"
                    b"<p>You can close this window.</p></body></html>"
                )
                self.callback(None, error)
            elif code:
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h1>Authorization Successful</h1>"
                    b"<p>You can close this window and return to Rayforge.</p>"
                    b"</body></html>"
                )
                self.callback(code, None)
            else:
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h1>Invalid Request</h1></body></html>"
                )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class PatreonProvider(LicenseProvider):
    AUTHORIZE_URL = "https://www.patreon.com/oauth2/authorize"
    TOKEN_URL = "https://www.patreon.com/api/oauth2/token"
    API_BASE = "https://www.patreon.com/api/oauth2/v2"
    REDIRECT_PORT = 8765

    def __init__(self, licenses_dir: Path, client_id: str):
        self.config_file = licenses_dir / "patreon.yaml"
        self.client_id = client_id
        self._access_token: Optional[str] = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_config()

    @property
    def name(self) -> str:
        return "patreon"

    def is_configured(self) -> bool:
        return self._access_token is not None

    def validate(self, config: Dict) -> LicenseResult:
        if not self._access_token:
            return LicenseResult(
                status=LicenseStatus.NOT_FOUND,
                message="Patreon account not linked",
            )

        tier_ids = config.get("patreon_tier_ids", [])
        if not tier_ids:
            return LicenseResult(
                status=LicenseStatus.NOT_FOUND,
                message="This addon does not support Patreon unlock",
            )

        cached = self._get_valid_cache(tier_ids)
        if cached:
            return LicenseResult(**cached)

        try:
            url = (
                f"{self.API_BASE}/identity"
                f"?include=memberships"
                f"&fields[member]=patron_status,last_charge_date"
                f"&fields[tier]=title"
            )

            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {self._access_token}"}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())

            included = result.get("included", [])
            tier_map = {}
            for item in included:
                if item.get("type") == "tier":
                    tier_map[item["id"]] = item.get("attributes", {}).get(
                        "title", ""
                    )

            for item in included:
                if item.get("type") != "member":
                    continue

                attrs = item.get("attributes", {})

                if attrs.get("patron_status") != "active_patron":
                    continue

                tier_rel = (
                    item.get("relationships", {})
                    .get("currently_entitled_tiers", {})
                    .get("data", [])
                )

                entitled_tier_ids = {t["id"] for t in tier_rel}
                required_tier_ids = set(tier_ids)

                if entitled_tier_ids & required_tier_ids:
                    matched_tiers = list(entitled_tier_ids & required_tier_ids)
                    result_obj = LicenseResult(
                        status=LicenseStatus.VALID,
                        message="Active Patreon supporter",
                        license_type=LicenseType.SUBSCRIPTION,
                        last_validated=datetime.now(),
                        metadata={
                            "tier_ids": matched_tiers,
                        },
                    )

                    cache_key = self._make_cache_key(tier_ids)
                    self._cache_result(cache_key, result_obj)
                    return result_obj

            return LicenseResult(
                status=LicenseStatus.INVALID,
                message="Not a patron at required tier",
            )

        except urllib.error.URLError as e:
            cached = self._get_valid_cache(tier_ids)
            if cached:
                logger.warning(
                    f"Network error during Patreon validation, using cache: "
                    f"{e.reason}"
                )
                return LicenseResult(**cached)
            return LicenseResult(
                status=LicenseStatus.ERROR,
                message=f"Network error: {e.reason}",
            )
        except Exception as e:
            logger.error(f"Patreon validation failed: {e}")
            return LicenseResult(
                status=LicenseStatus.ERROR,
                message=f"Validation failed: {str(e)}",
            )

    def _make_cache_key(self, tier_ids: List[str]) -> str:
        return ",".join(sorted(tier_ids))

    def _get_valid_cache(self, tier_ids: List[str]) -> Optional[Dict]:
        cache_key = self._make_cache_key(tier_ids)
        cached = self._cache.get(cache_key)
        if not cached:
            return None

        cached_result = LicenseResult(**cached)
        if cached_result.is_valid_for_offline():
            return cached
        return None

    def _cache_result(self, cache_key: str, result: LicenseResult) -> None:
        cache_data = {
            "status": result.status.value,
            "message": result.message,
            "license_type": result.license_type.value,
            "customer_email": result.customer_email,
            "last_validated": result.last_validated.isoformat()
            if result.last_validated
            else None,
            "metadata": result.metadata,
        }
        self._cache[cache_key] = cache_data
        self._save_cache()

    def get_oauth_url(self) -> str:
        redirect_uri = f"http://127.0.0.1:{self.REDIRECT_PORT}/callback"
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": "identity identity.memberships",
        }
        return f"{self.AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    def start_oauth_flow(
        self,
        on_complete: Callable[[bool, Optional[str]], None],
    ) -> tuple[int, Thread]:
        """
        Start the OAuth flow by launching a local HTTP server.

        Returns:
            Tuple of (port, thread) where thread is running the server.
        """
        callback_received = {"code": None, "error": None}

        def callback(code, error):
            callback_received["code"] = code
            callback_received["error"] = error

        def handler_factory(*args, **kwargs):
            return OAuthCallbackHandler(callback, *args, **kwargs)

        server = HTTPServer(("127.0.0.1", self.REDIRECT_PORT), handler_factory)

        def run_server():
            server.handle_request()
            if callback_received["code"]:
                success = self.exchange_code(
                    callback_received["code"], self.REDIRECT_PORT
                )
                on_complete(success, None)
            else:
                on_complete(False, callback_received["error"])

        thread = Thread(target=run_server, daemon=True)
        thread.start()

        return self.REDIRECT_PORT, thread

    def exchange_code(self, code: str, redirect_port: int) -> bool:
        redirect_uri = f"http://127.0.0.1:{redirect_port}/callback"
        data = urllib.parse.urlencode(
            {
                "code": code,
                "client_id": self.client_id,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            }
        ).encode()

        try:
            req = urllib.request.Request(
                self.TOKEN_URL, data=data, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())

            self._access_token = result.get("access_token")
            if self._access_token:
                self._save_config()
                return True
        except Exception as e:
            logger.error(f"Failed to exchange Patreon OAuth code: {e}")

        return False

    def unlink(self) -> None:
        self._access_token = None
        self._cache.clear()
        if self.config_file.exists():
            self.config_file.unlink()

    def clear_cache(self) -> None:
        self._cache.clear()
        self._save_cache()

    def _load_config(self) -> None:
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    data = yaml.safe_load(f) or {}
                    self._access_token = data.get("access_token")
                    self._cache = data.get("cache", {})
            except (yaml.YAMLError, IOError) as e:
                logger.warning(f"Failed to load Patreon config: {e}")

    def _save_config(self) -> None:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "access_token": self._access_token,
            "cache": self._cache,
        }
        with open(self.config_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def _save_cache(self) -> None:
        self._save_config()

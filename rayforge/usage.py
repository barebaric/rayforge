import json
import locale
import logging
import os
import platform
import threading
import urllib.error
import urllib.request
import uuid
from typing import Optional

from . import __version__
from .config import UMAMI_URL, UMAMI_WEBSITE_ID

logger = logging.getLogger(__name__)

_BROWSER_UA_TAIL = (
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _get_language() -> str:
    try:
        locale.setlocale(locale.LC_ALL, "")
        lang = locale.getlocale()[0]
        if lang:
            return lang.replace("_", "-")
        return "en-US"
    except locale.Error:
        return "en-US"


def _get_os_info() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            release = platform.freedesktop_os_release()
            distro = release.get("ID", "linux")
            version = release.get("VERSION_ID", "")
            if version:
                return f"linux/{distro}/{version}"
            return f"linux/{distro}"
        except (OSError, KeyError, ValueError):
            return "linux"
    elif system == "Windows":
        return f"windows/{platform.release()}"
    elif system == "Darwin":
        return f"macos/{platform.mac_ver()[0]}"
    return system.lower()


def _get_user_agent() -> str:
    """Returns a browser-style UA whose OS token matches the host OS,
    since the analytics server derives its OS dimension from it."""
    system = platform.system()
    if system == "Windows":
        return f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) {_BROWSER_UA_TAIL}"
    if system == "Darwin":
        return (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            f"{_BROWSER_UA_TAIL}"
        )
    machine = platform.machine()
    return f"Mozilla/5.0 (X11; Linux {machine}) {_BROWSER_UA_TAIL}"


def _get_machine_data(machine) -> dict:
    """Extracts anonymous type properties from a machine. Uses duck
    typing to avoid a dependency on the machine model classes."""
    head = machine.get_default_laser_head()
    laser = head.laser_type.value if head else "none"
    power = int(head.effective_max_power_watts) if head else 0

    width, height = machine.axis_extents
    return {
        "driver": machine.driver_name or "none",
        "laser": laser,
        "power_w": str(power),
        "bed": f"{int(width)}x{int(height)}",
        "rotary": "true" if machine.rotary_modules else "false",
    }


class UsageTracker:
    _instance: Optional["UsageTracker"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._enabled = False
        self._screen = self._get_screen_size()
        self._language = _get_language()
        self._os = _get_os_info()
        self._user_agent = _get_user_agent()
        self._version = __version__ or "unknown"
        self._cache_token: str | None = None
        self._session_id = str(uuid.uuid4())

    def _get_screen_size(self) -> str | None:
        try:
            from .ui_gtk.shared.gtk import get_screen_size

            size = get_screen_size()
            if size:
                return f"{size[0]}x{size[1]}"
        except Exception:
            logger.debug("Failed to get screen size", exc_info=True)
        return None

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if enabled:
            logger.info("Usage tracking enabled")
        else:
            logger.info("Usage tracking disabled")

    def _base_payload(self, title: str, url: str) -> dict:
        # Umami derives the session from website, IP, user agent and
        # the payload's "id" field; a "sessionId" field is ignored.
        payload = {
            "website": UMAMI_WEBSITE_ID,
            "id": self._session_id,
            "language": self._language,
            "title": title,
            "hostname": "",
            "url": url,
            "referrer": "",
        }
        if self._screen:
            payload["screen"] = self._screen
        return payload

    def track_page_view(self, url: str, title: str | None = None):
        if not self._enabled:
            return
        if os.environ.get("RAYFORGE_NO_USAGE_TRACKING"):
            return
        if not url.startswith("/"):
            url = "/" + url
        payload = self._base_payload(title or url, f"file://{url}")
        payload["data"] = {
            "app_version": self._version,
            "os": self._os,
        }
        self._send_event(payload)

    def track_machines(self, machines: list):
        """Reports the type of each configured machine as a named
        event, so the analytics dashboard can break usage down by
        driver, laser type and bed size. Machine names and identifiers
        are never sent."""
        for machine in machines:
            if machine.placeholder:
                continue
            payload = self._base_payload("Machine", "/machine")
            payload["name"] = "machine"
            payload["data"] = _get_machine_data(machine)
            self._send_event(payload)

    def _send_event(self, payload: dict):
        def _send():
            try:
                body = {"type": "event", "payload": payload}
                data = json.dumps(body).encode("utf-8")
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": self._user_agent,
                    "Accept": "*/*",
                    "Origin": "null",
                    "Sec-Fetch-Dest": "empty",
                    "Sec-Fetch-Mode": "cors",
                    "Sec-Fetch-Site": "cross-site",
                }
                if self._cache_token:
                    headers["x-umami-cache"] = self._cache_token

                req = urllib.request.Request(
                    UMAMI_URL, data=data, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    response_body = response.read().decode("utf-8")
                    try:
                        resp_json = json.loads(response_body)
                        if resp_json and "cache" in resp_json:
                            self._cache_token = resp_json["cache"]
                    except json.JSONDecodeError:
                        pass
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8")
                logger.warning(
                    f"Usage tracking request failed: {e.code} {error_body}"
                )
            except urllib.error.URLError as e:
                logger.warning(f"Usage tracking request failed: {e}")
            except (OSError, TimeoutError, ValueError) as e:
                logger.warning(f"Usage tracking error: {e}")

        thread = threading.Thread(target=_send, daemon=True)
        thread.start()


def get_usage_tracker() -> UsageTracker:
    return UsageTracker()

import json
import locale
import logging
import threading
from typing import Optional
import urllib.request
import urllib.error

from .config import UMAMI_URL, UMAMI_WEBSITE_ID

logger = logging.getLogger(__name__)


def _get_language() -> str:
    try:
        lang = locale.getdefaultlocale()[0]
        return lang or "en-US"
    except Exception:
        return "en-US"


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
        self._cache_token: Optional[str] = None

    def _get_screen_size(self) -> str:
        try:
            from .ui_gtk.shared.gtk import get_screen_size

            size = get_screen_size()
            if size:
                return f"{size[0]}x{size[1]}"
        except Exception:
            pass
        return "unknown"

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if enabled:
            logger.info("Usage tracking enabled")
        else:
            logger.info("Usage tracking disabled")

    def track_page_view(self, url: str, title: Optional[str] = None):
        if not self._enabled:
            return
        if not url.startswith("/"):
            url = "/" + url
        full_url = f"file://{url}"
        self._send_event(
            payload={
                "website": UMAMI_WEBSITE_ID,
                "screen": self._screen,
                "language": self._language,
                "title": title or url,
                "hostname": "",
                "url": full_url,
                "referrer": "",
            },
        )

    def _send_event(self, payload: dict):
        def _send():
            try:
                body = {"type": "event", "payload": payload}
                data = json.dumps(body).encode("utf-8")
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
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
            except Exception as e:
                logger.warning(f"Usage tracking error: {e}")

        thread = threading.Thread(target=_send, daemon=True)
        thread.start()


def get_usage_tracker() -> UsageTracker:
    return UsageTracker()

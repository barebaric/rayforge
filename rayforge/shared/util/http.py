"""
Shared resilient HTTP utilities for Rayforge.

Provides ``resilient_get`` and ``resilient_post`` with automatic retry
and exponential backoff for transient server/network failures.
All functions return ``None`` on unrecoverable failure and never
raise; errors are logged with structured ``extra`` fields
(``error_domain``, ``http_url``, ``http_status``, ``attempt``) to
support machine-readable RCA filtering.
"""

import logging
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

logger = logging.getLogger(__name__)

RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})

_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BACKOFF_SECONDS = 0.75
_DEFAULT_TIMEOUT_SECONDS = 10

_BASE_HEADERS: Dict[str, str] = {
    "User-Agent": "rayforge",
}


def resilient_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    backoff: float = _DEFAULT_BACKOFF_SECONDS,
) -> Optional[bytes]:
    """
    HTTP GET with retry/backoff for transient failures.

    Args:
        url: URL to fetch.
        headers: Extra request headers (merged with defaults).
        timeout: Per-attempt timeout in seconds.
        max_attempts: Maximum number of attempts before giving up.
        backoff: Seconds to wait between attempts.

    Returns:
        Response body bytes on success, or ``None`` on failure.
    """
    merged = dict(_BASE_HEADERS)
    if headers:
        merged.update(headers)

    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(url, headers=merged)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()

        except urllib.error.HTTPError as exc:
            retryable = (
                exc.code in RETRYABLE_HTTP_STATUSES
                and attempt < max_attempts
            )
            if retryable:
                logger.warning(
                    "HTTP GET %s returned %s (attempt %s/%s)",
                    url,
                    exc.code,
                    attempt,
                    max_attempts,
                    extra={
                        "error_domain": "http",
                        "http_status": exc.code,
                        "http_url": url,
                        "attempt": attempt,
                    },
                )
                time.sleep(backoff)
                continue

            logger.warning(
                "HTTP GET %s failed: HTTP %s",
                url,
                exc.code,
                extra={
                    "error_domain": "http",
                    "http_status": exc.code,
                    "http_url": url,
                },
            )
            return None

        except urllib.error.URLError as exc:
            if attempt < max_attempts:
                logger.warning(
                    "HTTP GET %s network error (attempt %s/%s): %s",
                    url,
                    attempt,
                    max_attempts,
                    exc.reason,
                    extra={
                        "error_domain": "network",
                        "http_url": url,
                        "attempt": attempt,
                    },
                )
                time.sleep(backoff)
                continue
            logger.error(
                "HTTP GET %s failed after %s attempts: %s",
                url,
                max_attempts,
                exc.reason,
                extra={
                    "error_domain": "network",
                    "http_url": url,
                },
            )
            return None

    return None


def resilient_post(
    url: str,
    data: bytes,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = 1,
    backoff: float = _DEFAULT_BACKOFF_SECONDS,
) -> Optional[bytes]:
    """
    HTTP POST with retry/backoff for transient failures.

    POST defaults to ``max_attempts=1`` (no retry) to avoid accidental
    duplicate submissions.  Pass a higher value only for idempotent
    endpoints (e.g., pure read-like verification calls).

    Args:
        url: URL to post to.
        data: Request body bytes.
        headers: Extra request headers (merged with defaults).
        timeout: Per-attempt timeout in seconds.
        max_attempts: Maximum number of attempts before giving up.
        backoff: Seconds to wait between attempts.

    Returns:
        Response body bytes on success, or ``None`` on failure.
    """
    merged = dict(_BASE_HEADERS)
    if headers:
        merged.update(headers)

    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(
                url, data=data, headers=merged, method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()

        except urllib.error.HTTPError as exc:
            retryable = (
                exc.code in RETRYABLE_HTTP_STATUSES
                and attempt < max_attempts
            )
            if retryable:
                logger.warning(
                    "HTTP POST %s returned %s (attempt %s/%s)",
                    url,
                    exc.code,
                    attempt,
                    max_attempts,
                    extra={
                        "error_domain": "http",
                        "http_status": exc.code,
                        "http_url": url,
                        "attempt": attempt,
                    },
                )
                time.sleep(backoff)
                continue

            logger.warning(
                "HTTP POST %s failed: HTTP %s",
                url,
                exc.code,
                extra={
                    "error_domain": "http",
                    "http_status": exc.code,
                    "http_url": url,
                },
            )
            return None

        except urllib.error.URLError as exc:
            if attempt < max_attempts:
                logger.warning(
                    "HTTP POST %s network error (attempt %s/%s): %s",
                    url,
                    attempt,
                    max_attempts,
                    exc.reason,
                    extra={
                        "error_domain": "network",
                        "http_url": url,
                        "attempt": attempt,
                    },
                )
                time.sleep(backoff)
                continue
            logger.error(
                "HTTP POST %s failed after %s attempts: %s",
                url,
                max_attempts,
                exc.reason,
                extra={
                    "error_domain": "network",
                    "http_url": url,
                },
            )
            return None

    return None

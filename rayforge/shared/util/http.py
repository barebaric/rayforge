"""
Shared resilient HTTP utilities for Rayforge.

Provides ``resilient_get`` and ``resilient_post`` (synchronous, stdlib
urllib-based) and ``resilient_async_get`` and ``resilient_async_post``
(async, aiohttp-based) with automatic retry and exponential backoff
for transient server/network failures. All functions return ``None`` on
unrecoverable failure and never raise; errors are logged with
structured ``extra`` fields (``error_domain``, ``http_url``,
``http_status``, ``attempt``) to support machine-readable RCA
filtering.
"""

import asyncio
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


# ---------------------------------------------------------------------------
# Async variants (aiohttp)
#
# Use these from coroutine code.  Sync callers should use the urllib-based
# resilient_get/resilient_post above to avoid pulling aiohttp into a
# synchronous call path.
# ---------------------------------------------------------------------------

try:
    import aiohttp
except ImportError:  # pragma: no cover - aiohttp is a hard runtime dep
    aiohttp = None  # type: ignore[assignment]


async def resilient_async_get(
    url: str,
    headers=None,
    timeout: float = 15.0,
    max_attempts: int = 3,
    backoff: float = 0.75,
):
    """
    Async HTTP GET with retry/backoff for transient failures.

    Args:
        url: URL to fetch.
        headers: Extra request headers (merged with defaults).
        timeout: Per-attempt timeout in seconds.
        max_attempts: Maximum number of attempts before giving up.
        backoff: Seconds to wait between attempts.

    Returns:
        Response body bytes on success, or ``None`` on failure.

    Raises:
        RuntimeError: If aiohttp is not available.
    """
    if aiohttp is None:
        raise RuntimeError(
            "aiohttp is required for resilient_async_get but is not "
            "installed"
        )

    merged = dict(_BASE_HEADERS)
    if headers:
        merged.update(headers)
    aio_timeout = aiohttp.ClientTimeout(total=timeout)

    for attempt in range(1, max_attempts + 1):
        try:
            async with aiohttp.ClientSession(headers=merged) as session:
                async with session.get(url, timeout=aio_timeout) as response:
                    if response.status == 200:
                        return await response.read()
                    retryable = (
                        response.status in RETRYABLE_HTTP_STATUSES
                        and attempt < max_attempts
                    )
                    if retryable:
                        logger.warning(
                            "HTTP GET %s returned %s (attempt %s/%s)",
                            url,
                            response.status,
                            attempt,
                            max_attempts,
                            extra={
                                "error_domain": "http",
                                "http_status": response.status,
                                "http_url": url,
                                "attempt": attempt,
                            },
                        )
                        await asyncio.sleep(backoff)
                        continue
                    logger.warning(
                        "HTTP GET %s failed: HTTP %s",
                        url,
                        response.status,
                        extra={
                            "error_domain": "http",
                            "http_status": response.status,
                            "http_url": url,
                        },
                    )
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt < max_attempts:
                logger.warning(
                    "HTTP GET %s network error (attempt %s/%s): %s",
                    url,
                    attempt,
                    max_attempts,
                    exc,
                    extra={
                        "error_domain": "network",
                        "http_url": url,
                        "attempt": attempt,
                    },
                )
                await asyncio.sleep(backoff)
                continue
            logger.error(
                "HTTP GET %s failed after %s attempts: %s",
                url,
                max_attempts,
                exc,
                extra={
                    "error_domain": "network",
                    "http_url": url,
                },
            )
            return None

    return None


async def resilient_async_post(
    url: str,
    data: bytes,
    headers=None,
    timeout: float = 15.0,
    max_attempts: int = 1,
    backoff: float = 0.75,
):
    """
    Async HTTP POST with retry/backoff for transient failures.

    POST defaults to ``max_attempts=1`` (no retry) to avoid accidental
    duplicate submissions.  Pass a higher value only for idempotent
    endpoints.

    Args:
        url: URL to post to.
        data: Request body bytes.
        headers: Extra request headers (merged with defaults).
        timeout: Per-attempt timeout in seconds.
        max_attempts: Maximum number of attempts before giving up.
        backoff: Seconds to wait between attempts.

    Returns:
        Response body bytes on success, or ``None`` on failure.

    Raises:
        RuntimeError: If aiohttp is not available.
    """
    if aiohttp is None:
        raise RuntimeError(
            "aiohttp is required for resilient_async_post but is not "
            "installed"
        )

    merged = dict(_BASE_HEADERS)
    if headers:
        merged.update(headers)
    aio_timeout = aiohttp.ClientTimeout(total=timeout)

    for attempt in range(1, max_attempts + 1):
        try:
            async with aiohttp.ClientSession(headers=merged) as session:
                async with session.post(
                    url, data=data, timeout=aio_timeout
                ) as response:
                    if response.status in (200, 201, 204):
                        return await response.read()
                    retryable = (
                        response.status in RETRYABLE_HTTP_STATUSES
                        and attempt < max_attempts
                    )
                    if retryable:
                        logger.warning(
                            "HTTP POST %s returned %s (attempt %s/%s)",
                            url,
                            response.status,
                            attempt,
                            max_attempts,
                            extra={
                                "error_domain": "http",
                                "http_status": response.status,
                                "http_url": url,
                                "attempt": attempt,
                            },
                        )
                        await asyncio.sleep(backoff)
                        continue
                    logger.warning(
                        "HTTP POST %s failed: HTTP %s",
                        url,
                        response.status,
                        extra={
                            "error_domain": "http",
                            "http_status": response.status,
                            "http_url": url,
                        },
                    )
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt < max_attempts:
                logger.warning(
                    "HTTP POST %s network error (attempt %s/%s): %s",
                    url,
                    attempt,
                    max_attempts,
                    exc,
                    extra={
                        "error_domain": "network",
                        "http_url": url,
                        "attempt": attempt,
                    },
                )
                await asyncio.sleep(backoff)
                continue
            logger.error(
                "HTTP POST %s failed after %s attempts: %s",
                url,
                max_attempts,
                exc,
                extra={
                    "error_domain": "network",
                    "http_url": url,
                },
            )
            return None

    return None

"""
Tests for the shared resilient HTTP utility.

Uses mock patching of urllib.request.urlopen and HTTP error types
so no real network calls are made.
"""

from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from rayforge.shared.util.http import (
    RETRYABLE_HTTP_STATUSES,
    resilient_get,
    resilient_post,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _mock_urlopen_ok(body: bytes):
    """Return a context-manager mock that yields a 200 response."""
    resp = MagicMock()
    resp.read.return_value = body
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _http_error(code: int) -> HTTPError:
    return HTTPError(
        url="http://example.com",
        code=code,
        msg="error",
        hdrs={},  # type: ignore[arg-type]
        fp=None,
    )


# ---------------------------------------------------------------------------
# resilient_get – success path
# ---------------------------------------------------------------------------


class TestResilientGetSuccess:
    def test_returns_body_on_200(self):
        body = b"hello"
        with patch(
            "rayforge.shared.util.http.urllib.request.urlopen",
            return_value=_mock_urlopen_ok(body),
        ):
            result = resilient_get("http://example.com")

        assert result == body

    def test_passes_custom_headers(self):
        body = b"ok"
        with patch(
            "rayforge.shared.util.http.urllib.request.urlopen",
            return_value=_mock_urlopen_ok(body),
        ) as mock_open:
            resilient_get(
                "http://example.com",
                headers={"X-Custom": "value"},
            )

        req = mock_open.call_args[0][0]
        assert req.get_header("X-custom") == "value"


# ---------------------------------------------------------------------------
# resilient_get – HTTP error handling / retry
# ---------------------------------------------------------------------------


class TestResilientGetHttpErrors:
    def test_retries_on_retryable_status_then_succeeds(self):
        body = b"data"
        call_count = 0

        def urlopen_side_effect(req, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _http_error(503)
            return _mock_urlopen_ok(body)

        with patch(
            "rayforge.shared.util.http.urllib.request.urlopen",
            side_effect=urlopen_side_effect,
        ), patch("rayforge.shared.util.http.time.sleep"):
            result = resilient_get(
                "http://example.com",
                max_attempts=3,
                backoff=0,
            )

        assert result == body
        assert call_count == 2

    def test_returns_none_after_all_retries_exhausted(self):
        with patch(
            "rayforge.shared.util.http.urllib.request.urlopen",
            side_effect=_http_error(503),
        ), patch("rayforge.shared.util.http.time.sleep"):
            result = resilient_get(
                "http://example.com",
                max_attempts=3,
                backoff=0,
            )

        assert result is None

    def test_returns_none_immediately_on_non_retryable_status(self):
        """404 should not be retried."""
        call_count = 0

        def urlopen_side_effect(req, timeout):
            nonlocal call_count
            call_count += 1
            raise _http_error(404)

        with patch(
            "rayforge.shared.util.http.urllib.request.urlopen",
            side_effect=urlopen_side_effect,
        ):
            result = resilient_get(
                "http://example.com",
                max_attempts=3,
                backoff=0,
            )

        assert result is None
        assert call_count == 1

    def test_retryable_statuses_set_is_correct(self):
        assert 429 in RETRYABLE_HTTP_STATUSES
        assert 500 in RETRYABLE_HTTP_STATUSES
        assert 503 in RETRYABLE_HTTP_STATUSES
        assert 404 not in RETRYABLE_HTTP_STATUSES
        assert 401 not in RETRYABLE_HTTP_STATUSES


# ---------------------------------------------------------------------------
# resilient_get – network error (URLError)
# ---------------------------------------------------------------------------


class TestResilientGetUrlError:
    def test_retries_on_url_error_then_succeeds(self):
        body = b"payload"
        call_count = 0

        def urlopen_side_effect(req, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise URLError("connection refused")
            return _mock_urlopen_ok(body)

        with patch(
            "rayforge.shared.util.http.urllib.request.urlopen",
            side_effect=urlopen_side_effect,
        ), patch("rayforge.shared.util.http.time.sleep"):
            result = resilient_get(
                "http://example.com",
                max_attempts=3,
                backoff=0,
            )

        assert result == body

    def test_returns_none_after_all_network_retries_exhausted(self):
        with patch(
            "rayforge.shared.util.http.urllib.request.urlopen",
            side_effect=URLError("timeout"),
        ), patch("rayforge.shared.util.http.time.sleep"):
            result = resilient_get(
                "http://example.com",
                max_attempts=3,
                backoff=0,
            )

        assert result is None


# ---------------------------------------------------------------------------
# resilient_post
# ---------------------------------------------------------------------------


class TestResilientPost:
    def test_returns_body_on_200(self):
        body = b"response"
        with patch(
            "rayforge.shared.util.http.urllib.request.urlopen",
            return_value=_mock_urlopen_ok(body),
        ):
            result = resilient_post(
                "http://example.com",
                data=b"payload",
            )

        assert result == body

    def test_default_max_attempts_is_one(self):
        call_count = 0

        def urlopen_side_effect(req, timeout):
            nonlocal call_count
            call_count += 1
            raise URLError("network error")

        with patch(
            "rayforge.shared.util.http.urllib.request.urlopen",
            side_effect=urlopen_side_effect,
        ):
            result = resilient_post(
                "http://example.com",
                data=b"data",
            )

        assert result is None
        assert call_count == 1

    def test_retries_on_retryable_status_when_configured(self):
        body = b"ok"
        call_count = 0

        def urlopen_side_effect(req, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _http_error(503)
            return _mock_urlopen_ok(body)

        with patch(
            "rayforge.shared.util.http.urllib.request.urlopen",
            side_effect=urlopen_side_effect,
        ), patch("rayforge.shared.util.http.time.sleep"):
            result = resilient_post(
                "http://example.com",
                data=b"data",
                max_attempts=2,
                backoff=0,
            )

        assert result == body
        assert call_count == 2


# ---------------------------------------------------------------------------
# Async helpers (resilient_async_get / resilient_async_post)
#
# These tests use lightweight aiohttp mocks instead of pulling in
# aioresponses.  The dummy objects implement only the protocol surface that
# resilient_async_get/resilient_async_post actually use.
# ---------------------------------------------------------------------------

import pytest

from rayforge.shared.util.http import (
    resilient_async_get,
    resilient_async_post,
)


class _DummyAsyncResponse:
    def __init__(self, status, body=b""):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def read(self):
        return self._body


class _DummyAsyncSession:
    """
    Mimics aiohttp.ClientSession enough for resilient_async_get/post.

    The ``responses`` queue yields response objects or raises
    ClientError/TimeoutError on demand.  When a request fires, the next
    item in the queue is consumed; this lets a test script a sequence
    of 503, 503, 200-style behaviour.
    """

    def __init__(self, items, headers_seen=None):
        self._items = list(items)
        self._headers_seen = headers_seen

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, *args, **kwargs):
        if self._headers_seen is not None:
            self._headers_seen.append(kwargs.get("headers"))
        return self._next()

    def post(self, *args, **kwargs):
        if self._headers_seen is not None:
            self._headers_seen.append(kwargs.get("headers"))
        return self._next()

    def _next(self):
        if not self._items:
            raise AssertionError("No more scripted responses")
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _client_error(message="boom"):
    import aiohttp

    return aiohttp.ClientError(message)


def _timeout_error():
    return asyncio.TimeoutError()


import asyncio  # noqa: E402  -- placed after fixtures that use it


class TestResilientAsyncGetSuccess:
    @pytest.mark.asyncio
    async def test_returns_body_on_200(self):
        body = b'{"ok": true}'
        session = _DummyAsyncSession([_DummyAsyncResponse(200, body)])
        with patch(
            "rayforge.shared.util.http.aiohttp.ClientSession",
            return_value=session,
        ):
            result = await resilient_async_get(
                "http://example.com/api", max_attempts=1, backoff=0
            )
        assert result == body

    @pytest.mark.asyncio
    async def test_merges_default_user_agent(self):
        # Capture the kwargs passed to ClientSession so we can assert
        # that default + caller-provided headers were merged correctly.
        captured_kwargs = []

        def _capture(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return _DummyAsyncSession([_DummyAsyncResponse(200, b"ok")])

        with patch(
            "rayforge.shared.util.http.aiohttp.ClientSession",
            side_effect=_capture,
        ):
            await resilient_async_get(
                "http://example.com",
                headers={"X-Custom": "yes"},
                max_attempts=1,
                backoff=0,
            )
        session_headers = captured_kwargs[0]["headers"]
        assert session_headers["User-Agent"] == "rayforge"
        assert session_headers["X-Custom"] == "yes"


class TestResilientAsyncGetHttpErrors:
    @pytest.mark.asyncio
    async def test_retry_on_5xx_then_succeed(self):
        body = b"recovered"
        session = _DummyAsyncSession(
            [
                _DummyAsyncResponse(503),
                _DummyAsyncResponse(500),
                _DummyAsyncResponse(200, body),
            ]
        )
        with (
            patch(
                "rayforge.shared.util.http.aiohttp.ClientSession",
                return_value=session,
            ),
            patch("rayforge.shared.util.http.asyncio.sleep"),
        ):
            result = await resilient_async_get(
                "http://example.com",
                max_attempts=3,
                backoff=0,
            )
        assert result == body

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self):
        session = _DummyAsyncSession([_DummyAsyncResponse(404)])
        with (
            patch(
                "rayforge.shared.util.http.aiohttp.ClientSession",
                return_value=session,
            ),
            patch("rayforge.shared.util.http.asyncio.sleep"),
        ):
            result = await resilient_async_get(
                "http://example.com",
                max_attempts=3,
                backoff=0,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_attempts_exhausted(self):
        session = _DummyAsyncSession(
            [
                _DummyAsyncResponse(503),
                _DummyAsyncResponse(503),
                _DummyAsyncResponse(503),
            ]
        )
        with (
            patch(
                "rayforge.shared.util.http.aiohttp.ClientSession",
                return_value=session,
            ),
            patch("rayforge.shared.util.http.asyncio.sleep"),
        ):
            result = await resilient_async_get(
                "http://example.com",
                max_attempts=3,
                backoff=0,
            )
        assert result is None


class TestResilientAsyncGetNetworkErrors:
    @pytest.mark.asyncio
    async def test_retry_on_client_error(self):
        body = b"recovered"
        session = _DummyAsyncSession(
            [_client_error(), _client_error(), _DummyAsyncResponse(200, body)]
        )
        with (
            patch(
                "rayforge.shared.util.http.aiohttp.ClientSession",
                return_value=session,
            ),
            patch("rayforge.shared.util.http.asyncio.sleep"),
        ):
            result = await resilient_async_get(
                "http://example.com",
                max_attempts=3,
                backoff=0,
            )
        assert result == body

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        body = b"recovered"
        session = _DummyAsyncSession(
            [_timeout_error(), _DummyAsyncResponse(200, body)]
        )
        with (
            patch(
                "rayforge.shared.util.http.aiohttp.ClientSession",
                return_value=session,
            ),
            patch("rayforge.shared.util.http.asyncio.sleep"),
        ):
            result = await resilient_async_get(
                "http://example.com",
                max_attempts=3,
                backoff=0,
            )
        assert result == body

    @pytest.mark.asyncio
    async def test_returns_none_after_all_attempts_fail(self):
        session = _DummyAsyncSession(
            [
                _client_error(),
                _client_error(),
                _client_error(),
            ]
        )
        with (
            patch(
                "rayforge.shared.util.http.aiohttp.ClientSession",
                return_value=session,
            ),
            patch("rayforge.shared.util.http.asyncio.sleep"),
        ):
            result = await resilient_async_get(
                "http://example.com",
                max_attempts=3,
                backoff=0,
            )
        assert result is None


class TestResilientAsyncPost:
    @pytest.mark.asyncio
    async def test_default_max_attempts_is_1(self):
        """POST should not retry by default to avoid duplicates."""
        session = _DummyAsyncSession(
            [_DummyAsyncResponse(503), _DummyAsyncResponse(200, b"ok")]
        )
        with (
            patch(
                "rayforge.shared.util.http.aiohttp.ClientSession",
                return_value=session,
            ),
            patch("rayforge.shared.util.http.asyncio.sleep"),
        ):
            result = await resilient_async_post(
                "http://example.com", data=b"payload"
            )
        assert result is None  # first 503, no retry

    @pytest.mark.asyncio
    async def test_post_2xx_returns_body(self):
        for status in (200, 201, 204):
            session = _DummyAsyncSession([_DummyAsyncResponse(status, b"x")])
            with patch(
                "rayforge.shared.util.http.aiohttp.ClientSession",
                return_value=session,
            ):
                result = await resilient_async_post(
                    "http://example.com",
                    data=b"payload",
                    max_attempts=1,
                    backoff=0,
                )
            assert result == b"x"

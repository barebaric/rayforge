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

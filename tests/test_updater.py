"""
Tests for AppUpdateChecker.

The retry/network behavior of ``_fetch_latest_release`` is delegated to
``rayforge.shared.util.http.resilient_async_get`` and is covered in
``tests/shared/util/test_http.py``.  These tests focus on:
  * the high-level check_on_startup / _check_worker orchestration
  * JSON payload parsing inside _fetch_latest_release
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from rayforge.updater import AppUpdateChecker


@pytest.fixture
def checker():
    mock_task_mgr = MagicMock()
    mock_context = MagicMock()
    mock_context.config.check_for_app_updates = True
    return AppUpdateChecker(mock_task_mgr, mock_context)


class TestCheckOnStartup:
    def test_disabled_does_not_schedule(self, checker):
        checker._context.config.check_for_app_updates = False
        checker.check_on_startup()
        checker._task_mgr.add_coroutine.assert_not_called()

    def test_enabled_schedules_check(self, checker):
        checker.check_on_startup()
        checker._task_mgr.add_coroutine.assert_called_once()


@pytest.mark.asyncio
class TestCheckWorker:
    async def test_no_notification_when_up_to_date(self, checker):
        ctx = MagicMock()
        with (
            patch.object(
                checker,
                "_fetch_latest_release",
                return_value={"tag_name": "0.0.1"},
            ),
            patch("rayforge.updater.__version__", "0.0.1"),
        ):
            await checker._check_worker(ctx)

        checker._task_mgr.schedule_on_main_thread.assert_not_called()
        ctx.set_message.assert_called()

    async def test_notification_when_new_version_available(self, checker):
        ctx = MagicMock()
        with (
            patch.object(
                checker,
                "_fetch_latest_release",
                return_value={"tag_name": "99.0.0"},
            ),
            patch("rayforge.updater.__version__", "1.0.0"),
        ):
            await checker._check_worker(ctx)

        checker._task_mgr.schedule_on_main_thread.assert_called_once()
        kwargs = checker._task_mgr.schedule_on_main_thread.call_args[1]
        assert "99.0.0" in kwargs["message"]

    async def test_no_notification_on_fetch_failure(self, checker):
        ctx = MagicMock()
        with patch.object(checker, "_fetch_latest_release", return_value=None):
            await checker._check_worker(ctx)

        checker._task_mgr.schedule_on_main_thread.assert_not_called()
        ctx.set_message.assert_called()

    async def test_no_notification_on_exception(self, checker):
        ctx = MagicMock()
        with patch.object(
            checker,
            "_fetch_latest_release",
            side_effect=Exception("network error"),
        ):
            await checker._check_worker(ctx)

        checker._task_mgr.schedule_on_main_thread.assert_not_called()
        ctx.set_message.assert_called()

    async def test_git_describe_version_treated_as_equal(self, checker):
        ctx = MagicMock()
        with (
            patch.object(
                checker,
                "_fetch_latest_release",
                return_value={"tag_name": "1.5.2"},
            ),
            patch("rayforge.updater.__version__", "1.5.2-3-gabcdef1"),
        ):
            await checker._check_worker(ctx)

        checker._task_mgr.schedule_on_main_thread.assert_not_called()

    async def test_prerelease_notified_when_release_exists(self, checker):
        ctx = MagicMock()
        with (
            patch.object(
                checker,
                "_fetch_latest_release",
                return_value={"tag_name": "1.0.0"},
            ),
            patch("rayforge.updater.__version__", "1.0.0-beta1"),
        ):
            await checker._check_worker(ctx)

        checker._task_mgr.schedule_on_main_thread.assert_called_once()

    async def test_pep440_post_git_not_newer_than_older_release(self, checker):
        ctx = MagicMock()
        with (
            patch.object(
                checker,
                "_fetch_latest_release",
                return_value={"tag_name": "1.5.2"},
            ),
            patch(
                "rayforge.updater.__version__",
                "1.6.0b2.post6+git.7f927a18",
            ),
        ):
            await checker._check_worker(ctx)

        checker._task_mgr.schedule_on_main_thread.assert_not_called()


@pytest.mark.asyncio
class TestFetchLatestRelease:
    async def test_returns_parsed_dict_on_success(self, checker):
        payload = {"tag_name": "1.2.3"}
        with patch(
            "rayforge.updater.resilient_async_get",
            return_value=json.dumps(payload).encode(),
        ):
            result = await checker._fetch_latest_release()
        assert result == payload

    async def test_returns_none_when_util_returns_none(self, checker):
        with patch(
            "rayforge.updater.resilient_async_get", return_value=None
        ):
            result = await checker._fetch_latest_release()
        assert result is None

    async def test_returns_none_on_invalid_json(self, checker):
        with patch(
            "rayforge.updater.resilient_async_get",
            return_value=b"not json {{{",
        ):
            result = await checker._fetch_latest_release()
        assert result is None

    async def test_returns_none_on_non_object_payload(self, checker):
        # JSON list, not a dict — should be rejected
        with patch(
            "rayforge.updater.resilient_async_get",
            return_value=b'["not", "an", "object"]',
        ):
            result = await checker._fetch_latest_release()
        assert result is None

    async def test_passes_correct_headers(self, checker):
        with patch(
            "rayforge.updater.resilient_async_get",
            return_value=b'{"tag_name": "1.0.0"}',
        ) as mock_get:
            await checker._fetch_latest_release()
        # The util is called with the GitHub Accept header
        kwargs = mock_get.call_args.kwargs
        assert kwargs["headers"]["Accept"] == "application/vnd.github+json"

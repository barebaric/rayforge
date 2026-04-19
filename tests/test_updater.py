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

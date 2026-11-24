from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rayforge.const import DOWNLOAD_URL
from rayforge.updater import AppUpdateChecker

INSTALLER_URL = (
    "https://github.com/barebaric/rayforge/releases/download/"
    "99.0.0/rayforge-v99.0.0-installer.exe"
)


def _release(tag="99.0.0", with_installer=True):
    release = {"tag_name": tag, "assets": []}
    if with_installer:
        version = tag.lstrip("v")
        release["assets"].append(
            {
                "name": f"rayforge-v{version}-installer.exe",
                "browser_download_url": INSTALLER_URL,
            }
        )
    return release


def _make_checker(on_update_requested=None, auto_download=True):
    mock_task_mgr = MagicMock()
    mock_context = MagicMock()
    mock_context.config.check_for_app_updates = True
    mock_context.config.auto_download_app_updates = auto_download
    return AppUpdateChecker(
        mock_task_mgr, mock_context, on_update_requested=on_update_requested
    )


@pytest.fixture
def checker():
    return _make_checker()


def _notify(checker):
    return cast(MagicMock, checker._task_mgr.schedule_on_main_thread)


def _add_coroutine(checker):
    return cast(MagicMock, checker._task_mgr.add_coroutine)


def _last_notification(checker):
    return _notify(checker).call_args[1]


class TestCheckOnStartup:
    def test_disabled_does_not_schedule(self, checker):
        checker._context.config.check_for_app_updates = False
        checker.check_on_startup()
        assert _add_coroutine(checker).call_count == 0

    def test_enabled_schedules_check(self, checker):
        checker.check_on_startup()
        assert _add_coroutine(checker).call_count == 1


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

        assert _notify(checker).call_count == 0
        ctx.set_message.assert_called()

    async def test_notification_when_new_version_available(self, checker):
        ctx = MagicMock()
        with (
            patch.object(
                checker, "_fetch_latest_release", return_value=_release()
            ),
            patch("rayforge.updater.__version__", "1.0.0"),
        ):
            await checker._check_worker(ctx)

        assert _notify(checker).call_count == 1
        kwargs = _last_notification(checker)
        assert "99.0.0" in kwargs["message"]
        assert kwargs["action_label"] == "Download"

    async def test_no_notification_on_fetch_failure(self, checker):
        ctx = MagicMock()
        with patch.object(checker, "_fetch_latest_release", return_value=None):
            await checker._check_worker(ctx)

        assert _notify(checker).call_count == 0
        ctx.set_message.assert_called()

    async def test_no_notification_on_exception(self, checker):
        ctx = MagicMock()
        with patch.object(
            checker,
            "_fetch_latest_release",
            side_effect=Exception("network error"),
        ):
            await checker._check_worker(ctx)

        assert _notify(checker).call_count == 0
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

        assert _notify(checker).call_count == 0

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

        assert _notify(checker).call_count == 1

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

        assert _notify(checker).call_count == 0

    async def test_downloads_installer_when_supported(self):
        checker = _make_checker(on_update_requested=MagicMock())
        ctx = MagicMock()
        with (
            patch.object(
                checker, "_fetch_latest_release", return_value=_release()
            ),
            patch("rayforge.updater.__version__", "1.0.0"),
            patch("rayforge.updater.supports_self_update", return_value=True),
        ):
            await checker._check_worker(ctx)

        assert _add_coroutine(checker).call_count == 1
        args = _add_coroutine(checker).call_args
        assert args[0][1] == "99.0.0"
        assert args[0][2] == INSTALLER_URL
        assert args[1]["key"] == "app-update-download"

    async def test_no_download_when_setting_disabled(self):
        checker = _make_checker(
            on_update_requested=MagicMock(), auto_download=False
        )
        ctx = MagicMock()
        with (
            patch.object(
                checker, "_fetch_latest_release", return_value=_release()
            ),
            patch("rayforge.updater.__version__", "1.0.0"),
            patch("rayforge.updater.supports_self_update", return_value=True),
        ):
            await checker._check_worker(ctx)

        assert _add_coroutine(checker).call_count == 0
        assert _last_notification(checker)["action_label"] == "Download"

    async def test_no_download_without_install_callback(self):
        checker = _make_checker()
        ctx = MagicMock()
        with (
            patch.object(
                checker, "_fetch_latest_release", return_value=_release()
            ),
            patch("rayforge.updater.__version__", "1.0.0"),
            patch("rayforge.updater.supports_self_update", return_value=True),
        ):
            await checker._check_worker(ctx)

        assert _add_coroutine(checker).call_count == 0

    async def test_no_download_when_release_has_no_installer(self):
        checker = _make_checker(on_update_requested=MagicMock())
        ctx = MagicMock()
        with (
            patch.object(
                checker,
                "_fetch_latest_release",
                return_value=_release(with_installer=False),
            ),
            patch("rayforge.updater.__version__", "1.0.0"),
            patch("rayforge.updater.supports_self_update", return_value=True),
        ):
            await checker._check_worker(ctx)

        assert _add_coroutine(checker).call_count == 0
        assert _last_notification(checker)["action_label"] == "Download"


@pytest.mark.asyncio
class TestDownloadWorker:
    async def test_download_offers_install(self, tmp_path):
        on_update = MagicMock()
        checker = _make_checker(on_update_requested=on_update)
        checker._download_dir = tmp_path
        ctx = MagicMock()

        async def fake_download(url, target, ctx):
            target.write_bytes(b"installer")

        with patch.object(checker, "_download_to", side_effect=fake_download):
            await checker._download_worker(ctx, "99.0.0", INSTALLER_URL)

        kwargs = _last_notification(checker)
        assert kwargs["action_label"] == "Update"
        assert "99.0.0" in kwargs["message"]

        kwargs["action_callback"]()
        expected = tmp_path / "rayforge-v99.0.0-installer.exe"
        on_update.assert_called_once_with(expected, "99.0.0")

    async def test_failed_download_falls_back_to_download_page(self, tmp_path):
        on_update = MagicMock()
        checker = _make_checker(on_update_requested=on_update)
        checker._download_dir = tmp_path
        ctx = MagicMock()
        target = tmp_path / "rayforge-v99.0.0-installer.exe"
        target.write_bytes(b"partial")

        with patch.object(
            checker,
            "_download_to",
            side_effect=RuntimeError("connection reset"),
        ):
            await checker._download_worker(ctx, "99.0.0", INSTALLER_URL)

        assert not target.exists()
        kwargs = _last_notification(checker)
        assert kwargs["action_label"] == "Download"
        on_update.assert_not_called()

    async def test_download_to_writes_file(self, tmp_path):
        checker = _make_checker(on_update_requested=MagicMock())
        response = MagicMock()
        response.status = 200
        response.content_length = None

        async def chunks(size):
            for chunk in (b"abc", b"def"):
                yield chunk

        response.content.iter_chunked = chunks
        session = MagicMock()
        session.get.return_value.__aenter__ = AsyncMock(return_value=response)
        session.get.return_value.__aexit__ = AsyncMock(return_value=False)
        client_session = MagicMock()
        client_session.__aenter__ = AsyncMock(return_value=session)
        client_session.__aexit__ = AsyncMock(return_value=False)
        ctx = MagicMock()
        target = tmp_path / "installer.exe"

        with patch(
            "rayforge.updater.aiohttp.ClientSession",
            return_value=client_session,
        ):
            await checker._download_to(INSTALLER_URL, target, ctx)

        assert target.read_bytes() == b"abcdef"
        assert not target.with_suffix(".part").exists()

    async def test_download_to_rejects_error_status(self, tmp_path):
        checker = _make_checker(on_update_requested=MagicMock())
        response = MagicMock()
        response.status = 404
        session = MagicMock()
        session.get.return_value.__aenter__ = AsyncMock(return_value=response)
        session.get.return_value.__aexit__ = AsyncMock(return_value=False)
        client_session = MagicMock()
        client_session.__aenter__ = AsyncMock(return_value=session)
        client_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "rayforge.updater.aiohttp.ClientSession",
                return_value=client_session,
            ),
            pytest.raises(RuntimeError),
        ):
            await checker._download_to(
                INSTALLER_URL, tmp_path / "installer.exe", MagicMock()
            )


class TestDownloadPageFallback:
    def test_links_to_installer_when_available(self):
        checker = _make_checker()
        with patch("rayforge.updater.webbrowser.open") as open_url:
            checker._notify_download_page("99.0.0", INSTALLER_URL)
            kwargs = _last_notification(checker)
            kwargs["action_callback"]()

        open_url.assert_called_once_with(INSTALLER_URL)

    def test_links_to_docs_page_without_installer(self):
        checker = _make_checker()
        with patch("rayforge.updater.webbrowser.open") as open_url:
            checker._notify_download_page("99.0.0", None)
            kwargs = _last_notification(checker)
            kwargs["action_callback"]()

        open_url.assert_called_once_with(DOWNLOAD_URL)

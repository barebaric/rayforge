import asyncio
import logging
import tempfile
import webbrowser
from collections.abc import Callable
from gettext import gettext as _
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from blinker import Signal

from . import __version__
from .const import DOWNLOAD_URL, GITHUB_RELEASES_API
from .shared.util.platform import supports_self_update
from .shared.util.versioning import is_newer_version

if TYPE_CHECKING:
    from .context import RayforgeContext
    from .shared.tasker import TaskManager

logger = logging.getLogger(__name__)

_WINDOWS_INSTALLER_SUFFIX = "-installer.exe"
_CHUNK_SIZE = 256 * 1024


class AppUpdateChecker:
    """
    Checks for new Rayforge versions via the GitHub Releases API.
    Runs checks in the background and notifies the UI via signals.

    When Rayforge can update itself, the Windows package is downloaded in
    the background and the notification offers to install it. On other
    platforms the notification only links to the download page.
    """

    notification_requested = Signal()

    def __init__(
        self,
        task_mgr: "TaskManager",
        context: "RayforgeContext",
        on_update_requested: Callable[[Path, str], None] | None = None,
    ):
        """
        Args:
            task_mgr: The task manager used to run background work.
            context: The application context.
            on_update_requested: Called with the path of the downloaded
                installer and its version when the user asks to install
                it. Without it, the downloaded package is not offered.
        """
        self._task_mgr = task_mgr
        self._context = context
        self._on_update_requested = on_update_requested
        self._download_dir = Path(tempfile.gettempdir()) / "rayforge-update"

    def check_on_startup(self):
        config = self._context.config
        if not config.check_for_app_updates:
            logger.info("App update check disabled by user.")
            return
        logger.info("Scheduling app version update check.")
        self._task_mgr.add_coroutine(
            self._check_worker, key="app-update-check"
        )

    async def _check_worker(self, ctx):
        ctx.set_message(_("Checking for Rayforge updates..."))
        try:
            release = await self._fetch_latest_release()
        except Exception as e:  # noqa: BLE001 - async task boundary
            logger.error(f"Failed to check for app updates: {e}")
            ctx.set_message(_("Update check failed."))
            return

        if release is None:
            ctx.set_message(_("Update check failed."))
            return

        latest_tag = release.get("tag_name", "")

        if not is_newer_version(latest_tag, __version__ or "0.0.0"):
            logger.info("Rayforge is up to date.")
            ctx.set_message(_("Rayforge is up to date."))
            return

        logger.info(
            f"New version available: {latest_tag} (current: {__version__})"
        )
        ctx.set_message(_("New version available."))
        installer_url = self._find_installer_url(release)

        if self._should_download(installer_url):
            logger.info(f"Downloading update package from {installer_url}")
            self._task_mgr.add_coroutine(
                self._download_worker,
                latest_tag,
                installer_url,
                key="app-update-download",
            )
            return

        self._notify_download_page(latest_tag, installer_url)

    def _should_download(self, installer_url: str | None) -> bool:
        if not installer_url or self._on_update_requested is None:
            return False
        if not supports_self_update():
            return False
        return self._context.config.auto_download_app_updates

    @staticmethod
    def _find_installer_url(release: dict) -> str | None:
        for asset in release.get("assets") or []:
            name = asset.get("name") or ""
            if name.endswith(_WINDOWS_INSTALLER_SUFFIX):
                return asset.get("browser_download_url")
        return None

    def _notify_download_page(self, version: str, installer_url: str | None):
        msg = _("Rayforge {version} is available.").format(version=version)
        url = installer_url or DOWNLOAD_URL

        def _open_download():
            webbrowser.open(url)

        self._task_mgr.schedule_on_main_thread(
            self.notification_requested.send,
            self,
            message=msg,
            persistent=True,
            action_label=_("Download"),
            action_callback=_open_download,
        )

    async def _download_worker(self, ctx, version: str, installer_url: str):
        ctx.set_message(
            _("Downloading Rayforge {version}...").format(version=version)
        )
        target = self._download_dir / f"rayforge-v{version}-installer.exe"
        try:
            await self._download_to(installer_url, target, ctx)
        except Exception as e:  # noqa: BLE001 - async task boundary
            logger.error(f"Failed to download update package: {e}")
            target.unlink(missing_ok=True)
            target.with_suffix(".part").unlink(missing_ok=True)
            ctx.set_message(_("Update download failed."))
            self._notify_download_page(version, installer_url)
            return

        logger.info(f"Update package saved to {target}")
        ctx.set_message(_("Update ready to install."))
        self._notify_install_ready(target, version)

    def _notify_install_ready(self, target: Path, version: str):
        msg = _("Rayforge {version} is ready to install.").format(
            version=version
        )
        self._task_mgr.schedule_on_main_thread(
            self.notification_requested.send,
            self,
            message=msg,
            persistent=True,
            action_label=_("Update"),
            action_callback=lambda: self._request_install(target, version),
        )

    def _request_install(self, target: Path, version: str):
        if self._on_update_requested is None:
            return
        self._on_update_requested(target, version)

    async def _download_to(self, url: str, target: Path, ctx):
        self._download_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(".part")
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=120)
        async with (
            aiohttp.ClientSession() as session,
            session.get(url, timeout=timeout) as response,
        ):
            if response.status != 200:
                raise RuntimeError(f"Download returned HTTP {response.status}")
            total = response.content_length or 0
            if total:
                ctx.set_total(total)
            received = 0
            with tmp_path.open("wb") as f:
                async for chunk in response.content.iter_chunked(_CHUNK_SIZE):
                    f.write(chunk)
                    received += len(chunk)
                    if total:
                        ctx.set_progress(received)
        tmp_path.replace(target)

    async def _fetch_latest_release(self) -> dict | None:
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    GITHUB_RELEASES_API,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as response,
            ):
                if response.status == 200:
                    return await response.json()
                logger.warning(f"GitHub API returned status {response.status}")
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Error fetching release info: {e}")
            return None

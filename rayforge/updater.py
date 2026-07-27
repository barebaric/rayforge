import json
import logging
import webbrowser
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional

from blinker import Signal

from . import __version__
from .const import DOWNLOAD_URL, GITHUB_RELEASES_API
from .shared.util.http import resilient_async_get
from .shared.util.versioning import is_newer_version

if TYPE_CHECKING:
    from .context import RayforgeContext
    from .shared.tasker import TaskManager

logger = logging.getLogger(__name__)


class AppUpdateChecker:
    """
    Checks for new Rayforge versions via the GitHub Releases API.
    Runs checks in the background and notifies the UI via signals.
    """

    notification_requested = Signal()

    def __init__(self, task_mgr: "TaskManager", context: "RayforgeContext"):
        self._task_mgr = task_mgr
        self._context = context

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
        except Exception as e:
            logger.error(f"Failed to check for app updates: {e}")
            ctx.set_message(_("Update check failed."))
            return

        if release is None:
            ctx.set_message(_("Update check failed."))
            return

        latest_tag = release.get("tag_name", "")

        if is_newer_version(latest_tag, __version__ or "0.0.0"):
            logger.info(
                f"New version available: {latest_tag} (current: {__version__})"
            )
            msg = _("Rayforge {version} is available.").format(
                version=latest_tag
            )

            def _open_download():
                webbrowser.open(DOWNLOAD_URL)

            self._task_mgr.schedule_on_main_thread(
                self.notification_requested.send,
                self,
                message=msg,
                persistent=True,
                action_label=_("Download"),
                action_callback=_open_download,
            )
            ctx.set_message(_("New version available."))
        else:
            logger.info("Rayforge is up to date.")
            ctx.set_message(_("Rayforge is up to date."))

    async def _fetch_latest_release(self) -> Optional[dict]:
        """
        Fetch the latest release info from the GitHub Releases API.

        Retries on transient HTTP and network failures via
        ``resilient_async_get`` (3 attempts, 0.75s backoff by default).
        Returns ``None`` on any failure; never raises.
        """
        body = await resilient_async_get(
            GITHUB_RELEASES_API,
            headers={"Accept": "application/vnd.github+json"},
        )
        if body is None:
            return None
        try:
            payload = json.loads(body)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse GitHub release payload: {e}")
            return None
        if not isinstance(payload, dict):
            logger.warning("GitHub API payload is not an object")
            return None
        return payload

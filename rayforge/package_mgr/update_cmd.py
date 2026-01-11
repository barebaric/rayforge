import asyncio
import logging
from gettext import ngettext
from typing import List, Tuple, TYPE_CHECKING
from blinker import Signal
from ..context import RayforgeContext
from .package import Package, PackageMetadata

if TYPE_CHECKING:
    from ..shared.tasker import TaskManager

logger = logging.getLogger(__name__)


class UpdateCommand:
    """
    Handles checking for and installing package updates.
    This class orchestrates the PackageManager, running its blocking
    operations in background threads and communicating results back to the
    UI via signals.
    """

    # Signal for UI feedback. The sender is this instance.
    # Kwargs can include: message, persistent, action_label, action_callback
    notification_requested = Signal()

    def __init__(self, task_mgr: "TaskManager", context: RayforgeContext):
        self._context = context
        self._package_mgr = context.package_mgr
        self._task_mgr = task_mgr

    def check_for_updates_on_startup(self):
        """
        Initiates a background task to check for package updates.
        This method is non-blocking.
        """
        logger.info("Scheduling startup package update check.")
        self._task_mgr.add_coroutine(
            self._check_for_updates_worker, key="package-update-check"
        )

    async def _check_for_updates_worker(self, ctx):
        """
        The async worker that performs the update check in a thread.
        """
        try:
            ctx.set_message(_("Checking for package updates..."))

            # PackageManager.check_for_updates() involves network I/O, so we
            # run it in a separate thread to avoid blocking the asyncio loop.
            updates = await asyncio.to_thread(
                self._package_mgr.check_for_updates
            )

            if updates:
                logger.info(f"Found {len(updates)} available package updates.")
                names = [
                    remote_meta.display_name or remote_meta.name
                    for _, remote_meta in updates
                ]

                if len(names) == 1:
                    msg = _("An update is available for {name}.").format(
                        name=names[0]
                    )
                elif len(names) == 2:
                    msg = _(
                        "Updates are available for {name1} and {name2}."
                    ).format(name1=names[0], name2=names[1])
                else:
                    msg = _(
                        "Updates are available for {name1}, "
                        "{name2}, and {num} others."
                    ).format(
                        name1=names[0], name2=names[1], num=len(names) - 2
                    )

                def _install_callback():
                    self.install_updates(updates)

                self._task_mgr.schedule_on_main_thread(
                    self.notification_requested.send,
                    self,
                    message=msg,
                    persistent=True,
                    action_label=_("Install All"),
                    action_callback=_install_callback,
                )
                ctx.set_message(_("Package updates found."))
            else:
                ctx.set_message(_("Packages are up to date."))

        except Exception as e:
            logger.error(f"Failed to check for package updates: {e}")
            ctx.set_message(_("Update check failed."))

    def install_updates(self, updates: List[Tuple[Package, PackageMetadata]]):
        """
        Initiates a background task to install a list of package updates.
        This method is non-blocking.
        """
        if not updates:
            return

        logger.info(f"Scheduling installation of {len(updates)} package(s).")
        self._task_mgr.add_coroutine(
            self._install_updates_worker, updates, key="package-install"
        )

    async def _install_updates_worker(
        self, ctx, updates: List[Tuple[Package, PackageMetadata]]
    ):
        """
        The async worker that installs multiple packages concurrently.
        """
        install_tasks = []
        for __, remote_meta in updates:
            # PackageManager.install_package is blocking (git clone, file I/O)
            task = asyncio.to_thread(
                self._package_mgr.install_package,
                remote_meta.url,
                remote_meta.name,
            )
            install_tasks.append(task)

        ctx.set_message(_("Installing package updates..."))
        results = await asyncio.gather(*install_tasks, return_exceptions=True)

        successful = []
        failed = []
        for i, result in enumerate(results):
            __, remote_meta = updates[i]
            if isinstance(result, Exception) or result is None:
                logger.error(
                    f"Failed to install update for {remote_meta.name}",
                    exc_info=result if isinstance(result, Exception) else None,
                )
                failed.append(remote_meta)
            else:
                logger.info(
                    f"Successfully installed update for {remote_meta.name}"
                )
                successful.append(remote_meta)

        num_success = len(successful)
        num_failed = len(failed)
        msg = ""

        if num_failed == 0 and num_success > 0:
            msg = ngettext(
                "Package successfully updated.",
                "{num} packages successfully updated.",
                num_success,
            ).format(num=num_success)
        elif num_success > 0 and num_failed > 0:
            msg = _("{num_s} packages updated, {num_f} failed.").format(
                num_s=num_success, num_f=num_failed
            )
        elif num_failed > 0 and num_success == 0:
            msg = ngettext(
                "Failed to update package.",
                "Failed to update {num} packages.",
                num_failed,
            ).format(num=num_failed)

        if msg:
            self._task_mgr.schedule_on_main_thread(
                self.notification_requested.send,
                self,
                message=msg,
            )

        if failed:
            ctx.set_message(
                _("Finished with {num_failed} errors.").format(
                    num_failed=len(failed)
                )
            )
        else:
            ctx.set_message(_("All package updates installed!"))

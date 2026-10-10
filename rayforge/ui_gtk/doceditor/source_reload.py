"""Watches imported files and offers to reload them when they change."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gi.repository import GLib

from ...context import get_context
from ...core.source_asset import SourceAsset
from ...doceditor.source_watcher import SourceWatcher
from . import file_dialogs
from .source_changed_dialog import SourceChangedDialog

if TYPE_CHECKING:
    from gi.repository import Gtk

    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 2


class SourceReloadMonitor:
    """
    Polls the files behind imported sources and asks the user whether to
    reload the ones that changed. Sources marked for automatic reload are
    reloaded without asking.
    """

    def __init__(
        self,
        parent: Gtk.Window,
        editor: DocEditor,
        watcher: SourceWatcher | None = None,
        dialog_factory: Callable[..., Any] = SourceChangedDialog,
        enabled: Callable[[], bool] | None = None,
    ):
        self._parent = parent
        self._editor = editor
        self._watcher = watcher or SourceWatcher(editor)
        self._dialog_factory = dialog_factory
        self._dialog: Any | None = None
        self._queue: list[SourceAsset] = []
        self._timeout_id: int | None = None
        self._enabled = enabled or _watching_enabled

    def start(self) -> None:
        if self._timeout_id is None:
            self._timeout_id = GLib.timeout_add_seconds(
                POLL_INTERVAL_SECONDS, self.tick
            )

    def stop(self) -> None:
        if self._timeout_id is not None:
            GLib.source_remove(self._timeout_id)
            self._timeout_id = None

    def tick(self) -> bool:
        if not self._enabled():
            return True
        try:
            self._handle(self._watcher.poll())
        except Exception:
            logger.exception("Checking imported files for changes failed")
        return True

    def check_now(self) -> None:
        """Checks all imported files right away, e.g. after opening."""
        if not self._enabled():
            return
        self._handle(self._watcher.check_now())

    def reload_assets(self, assets: list[SourceAsset]) -> None:
        """Reloads the given sources from disk on request."""
        for asset in assets:
            self._reload(asset)

    def choose_relink(self, asset: SourceAsset) -> None:
        """Lets the user pick another file for a source."""
        file_dialogs.show_relink_dialog(
            self._parent, asset, self._on_relink_response, asset
        )

    def relink_to(self, asset: SourceAsset, path: Path) -> None:
        self._editor.reload.relink_source(asset, path)
        self._watcher.acknowledge(asset)

    def _on_relink_response(self, dialog, result, asset: SourceAsset):
        try:
            file = dialog.open_finish(result)
        except GLib.Error:
            return
        if file is not None and file.get_path():
            self.relink_to(asset, Path(file.get_path()))

    def _handle(self, changed: list[SourceAsset]) -> None:
        for asset in changed:
            if asset.auto_reload:
                self._reload(asset)
            elif all(asset.uid != queued.uid for queued in self._queue):
                self._queue.append(asset)
        self._show_next()

    def _show_next(self) -> None:
        if self._dialog is not None or not self._queue:
            return
        assets, self._queue = self._queue, []
        dialog = self._dialog_factory(
            assets,
            on_reload=self._on_reload,
            on_ignore=self._on_ignore,
        )
        self._dialog = dialog
        dialog.present(self._parent)

    def _reload(self, asset: SourceAsset) -> None:
        self._editor.reload.reload_source(asset)
        self._watcher.acknowledge(asset)

    def _on_reload(
        self,
        selected: list[SourceAsset],
        always: bool,
        skipped: list[SourceAsset],
    ) -> None:
        self._dialog = None
        for asset in selected:
            if always and not asset.auto_reload:
                asset.auto_reload = True
                self._editor.mark_as_unsaved()
            self._reload(asset)
        for asset in skipped:
            self._watcher.acknowledge(asset)
        self._show_next()

    def _on_ignore(self, assets: list[SourceAsset]) -> None:
        self._dialog = None
        for asset in assets:
            self._watcher.acknowledge(asset)
        self._show_next()


def _watching_enabled() -> bool:
    return get_context().config.watch_source_files

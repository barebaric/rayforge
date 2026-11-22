from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.source_asset import SourceAsset

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)

FileSignature = tuple[int, int]


@dataclass
class _WatchState:
    signature: FileSignature
    pending: bool = True
    seen_hashes: set[str] = field(default_factory=set)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _signature(path: Path) -> FileSignature | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


class SourceWatcher:
    """
    Detects imported files that changed on disk.

    The watcher is driven by calling :meth:`poll` periodically. A file
    only counts as changed once its modification time and size were the
    same in two consecutive polls, so files that are still being written
    (or saved through a temporary file and a rename) are not reported
    half-way. The content is then compared with the data embedded in the
    document, so touching a file or saving it unchanged is ignored.

    Every changed content is reported once; :meth:`acknowledge` marks the
    current content of a file as handled without reporting it.
    """

    def __init__(self, editor: DocEditor):
        self._editor = editor
        self._states: dict[str, _WatchState] = {}

    def watched_assets(self) -> list[SourceAsset]:
        """Returns the sources whose files are worth watching."""
        doc = self._editor.doc
        used = {
            wp.source_segment.source_asset_uid
            for wp in doc.all_workpieces
            if wp.source_segment
        }
        return [
            asset
            for uid, asset in doc.source_assets.items()
            if uid in used
            and asset.source_file.is_absolute()
            and asset.metadata.get("_importer_class")
        ]

    def poll(self) -> list[SourceAsset]:
        """
        Checks all watched files and returns the sources whose file
        content changed and settled since the last report.
        """
        changed = []
        for asset in self.watched_assets():
            signature = _signature(asset.source_file)
            if signature is None:
                self._states.pop(asset.uid, None)
                continue
            state = self._states.get(asset.uid)
            if state is None:
                self._states[asset.uid] = _WatchState(signature)
                continue
            if signature != state.signature:
                state.signature = signature
                state.pending = True
                continue
            if state.pending:
                state.pending = False
                if self._is_new_content(asset, state):
                    changed.append(asset)
        return changed

    def check_now(self) -> list[SourceAsset]:
        """
        Compares all watched files with the document right away, without
        waiting for the files to settle. Meant for opening a project.
        """
        changed = []
        for asset in self.watched_assets():
            signature = _signature(asset.source_file)
            if signature is None:
                continue
            state = self._states.setdefault(asset.uid, _WatchState(signature))
            state.signature = signature
            state.pending = False
            if self._is_new_content(asset, state):
                changed.append(asset)
        return changed

    def acknowledge(self, asset: SourceAsset) -> None:
        """Marks the current file content of a source as handled."""
        signature = _signature(asset.source_file)
        data = _read(asset.source_file)
        if signature is None or data is None:
            return
        state = self._states.setdefault(asset.uid, _WatchState(signature))
        state.seen_hashes.add(_digest(data))

    def _is_new_content(self, asset: SourceAsset, state: _WatchState) -> bool:
        data = _read(asset.source_file)
        if data is None:
            return False
        digest = _digest(data)
        if digest in state.seen_hashes:
            return False
        if digest == _digest(asset.original_data):
            return False
        state.seen_hashes.add(digest)
        logger.info(f"Source file changed on disk: {asset.source_file}")
        return True

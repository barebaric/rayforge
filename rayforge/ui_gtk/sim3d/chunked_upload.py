"""
Chunked artifact upload controller for the 3D canvas.

Owns the chunked-upload state machine: when the compiled scene artifact
becomes GL-dirty, the controller prepares the per-layer upload items and
steps through them one at a time so a frame is never blocked uploading a
whole artifact.  CPU-bound item preparation (vertex decompression and
concatenation) runs in a worker thread; only the actual GL uploads run
on the main thread.  It also tracks the pending idle source so it can be
cancelled on teardown.

Emits ``upload_complete`` once every item has been processed, so the
presenter can build playback after the fresh layer groups exist.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from blinker import Signal
from gi.repository import GLib

from ...shared.tasker import Task, task_mgr

if TYPE_CHECKING:
    from ...simulator.scene3d import CompiledSceneArtifact
    from .renderer.scene_renderer import SceneRenderer, UploadItem

logger = logging.getLogger(__name__)


@dataclass
class _UploadState:
    """Progress of a chunked upload in flight."""

    items: list["UploadItem"]
    index: int


class ChunkedUploadController:
    """
    Steps through per-layer vertex/texture uploads on idle callbacks.

    ``_artifact_gl_dirty`` and ``_upload_state`` track whether the compiled
    artifact needs uploading and how far the chunked upload has progressed.
    ``process_pending`` is called each frame and starts a new chunked upload
    when the artifact is dirty.
    """

    def __init__(
        self,
        scene: "SceneRenderer",
        get_artifact: Callable[[], Optional["CompiledSceneArtifact"]],
        get_show_travel_moves: Callable[[], bool],
        get_gl_initialized: Callable[[], bool],
        make_current: Callable[[], None],
        request_render: Callable[[], None],
        on_luts_required: Callable[[], None],
    ):
        self.upload_complete = Signal()
        self._scene = scene
        self._get_artifact = get_artifact
        self._get_show_travel_moves = get_show_travel_moves
        self._get_gl_initialized = get_gl_initialized
        self._make_current = make_current
        self._request_render = request_render
        self._on_luts_required = on_luts_required

        self._artifact_gl_dirty = False
        self._upload_state: _UploadState | None = None
        self._idle_source_id: int | None = None

    def mark_artifact_dirty(self):
        """Mark the compiled artifact as needing a (re)upload."""
        self._artifact_gl_dirty = True

    @property
    def is_dirty(self) -> bool:
        """True while a compiled artifact upload is still pending."""
        return self._artifact_gl_dirty

    def process_pending(self):
        """Start a chunked upload when the artifact is GL-dirty."""
        if self._artifact_gl_dirty:
            self._artifact_gl_dirty = False
            self.start()

    def cancel(self):
        """Cancel any pending idle callback and reset upload state."""
        if self._idle_source_id is not None:
            GLib.source_remove(self._idle_source_id)
            self._idle_source_id = None
        self._upload_state = None
        self._artifact_gl_dirty = False

    def start(self):
        artifact = self._get_artifact()
        if not artifact:
            self._scene.clear_layers()
            self._request_render()
            return

        if not self._get_gl_initialized():
            return

        self._make_current()

        upload_items = self._scene.prepare_chunked_upload(
            artifact, self._get_show_travel_moves()
        )

        # Upload the power colour LUTs before any vertex data. The chunked
        # upload runs on idle callbacks, which can be pre-empted by a
        # redraw between items; a redraw that renders powered lines against
        # an uninitialised LUT would draw them at full brightness. This must
        # run after prepare_chunked_upload so the fresh renderers get it.
        self._on_luts_required()

        self._upload_state = _UploadState(items=upload_items, index=0)
        self._idle_source_id = GLib.idle_add(self._step, self._upload_state)

    def _step(self, state: _UploadState) -> bool:
        """Dispatches the next not-yet-dispatched item of ``state``.

        The state is passed explicitly (not read from
        ``self._upload_state``) so that a stale idle scheduled by a
        replaced chain can never advance the current one: idles from
        old chains find the mismatch and become no-ops.
        """
        self._idle_source_id = None
        if self._upload_state is not state:
            return False

        if state.index >= len(state.items):
            self._upload_state = None
            self.upload_complete.send(self)
            self._request_render()
            return False

        item = state.items[state.index]
        state.index += 1

        # Decompress/concat the vertex data in a worker thread; the GL
        # upload runs on the main thread via the when_done callback (the
        # GL context is only current there).
        task_mgr.run_thread(
            item.prepare,
            key=(id(self), "prepare-chunk-upload", id(state), state.index),
            when_done=lambda task, st=state, it=item: self._on_item_prepared(
                st, it, task
            ),
        )
        return False

    def _on_item_prepared(
        self, state: _UploadState, item: "UploadItem", task: Task
    ) -> None:
        """Uploads ``item`` after its worker-thread preparation finished.

        The item is passed explicitly rather than looked up through
        ``state.index``: the index may legitimately have advanced past
        this item (the chain always keeps exactly one worker in
        flight), and addressing by index would upload a different,
        not-yet-prepared item while silently discarding this payload.
        """
        if self._upload_state is not state:
            return
        if task.get_status() != "completed":
            self._upload_state = None
            return

        try:
            self._make_current()
            self._scene.upload_chunk(item)
        except Exception:
            logger.exception("Error during chunked upload")
            self._upload_state = None
            return

        self._idle_source_id = GLib.idle_add(self._step, state)

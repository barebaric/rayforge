"""
Chunked artifact upload controller for the 3D canvas.

Owns the chunked-upload idle state machine: when the compiled scene
artifact becomes GL-dirty, the controller prepares the per-layer upload
items and steps through them one per idle callback so a frame is never
blocked uploading a whole artifact.  It also tracks the pending idle
source so it can be cancelled on teardown.

Emits ``upload_complete`` once every item has been processed, so the
presenter can build playback after the fresh layer groups exist.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from blinker import Signal
from gi.repository import GLib

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
        self._upload_state: Optional[_UploadState] = None
        self._idle_source_id: Optional[int] = None

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
        self._idle_source_id = GLib.idle_add(self._step)

    def _step(self) -> bool:
        self._idle_source_id = None
        if self._upload_state is None:
            return False

        state = self._upload_state
        if state.index >= len(state.items):
            self._upload_state = None
            self.upload_complete.send(self)
            self._request_render()
            return False

        item = state.items[state.index]
        state.index += 1

        try:
            self._scene.upload_chunk(item)
        except Exception:
            logger.exception("[CANVAS3D] Error during chunked upload")
            self._upload_state = None
            return False

        self._idle_source_id = GLib.idle_add(self._step)
        return False

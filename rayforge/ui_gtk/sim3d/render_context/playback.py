"""Playback render context section."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ....simulator.op_player import OpPlayer
    from ....simulator.scene3d import CompiledSceneArtifact
    from .base import FrameInputs


class PlaybackContext:
    """Playback state and per-frame execution counters.

    ``executed_vertex_count`` / ``executed_travel_vertex_count`` are
    written each frame by the layer renderer groups; ``reached_count``
    by the texture renderer.  ``alpha_pending`` is the pending-alpha used
    while drawing not-yet-executed toolpaths.

    The plain constructor leaves the section empty; call :meth:`update`
    each frame to refresh the playback state.
    """

    def __init__(
        self,
        *,
        op_player: Optional["OpPlayer"] = None,
        compiled_artifact: Optional["CompiledSceneArtifact"] = None,
    ):
        self.op_player = op_player
        self.compiled_artifact = compiled_artifact
        self.executed_vertex_count = -1
        self.executed_travel_vertex_count = -1
        self.alpha_pending = 0.2
        self.reached_count: int | None = None

    def update(self, frame: "FrameInputs") -> None:
        """Refreshes the playback section from the current frame inputs."""
        self.op_player = frame.op_player
        self.compiled_artifact = frame.compiled_artifact
        self.executed_vertex_count = -1
        self.executed_travel_vertex_count = -1
        self.alpha_pending = 0.2
        self.reached_count = None

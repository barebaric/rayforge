"""Per-frame render context sections.

The RenderContext is a composite of four mutable sections (camera,
viewport, kinematics, playback), each refreshing itself in place from a
shared :class:`FrameInputs` bundle via ``update()``.
"""

from .base import FrameInputs, RenderContext
from .camera import CameraContext
from .kinematics import HeadConfig, KinematicsContext
from .playback import PlaybackContext
from .viewport import ViewportContext
from .visibility import SceneVisibility

__all__ = [
    "CameraContext",
    "FrameInputs",
    "HeadConfig",
    "KinematicsContext",
    "PlaybackContext",
    "RenderContext",
    "SceneVisibility",
    "ViewportContext",
]

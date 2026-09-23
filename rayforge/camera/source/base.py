import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit

import cv2
import numpy as np

from ..models.source_type import CameraSourceType

if TYPE_CHECKING:
    from ..models.camera import Camera

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceDescriptor:
    """User-facing label/value pair for a discoverable camera source."""

    label: str
    value: str


@dataclass
class SourceConfig:
    extra: dict[str, Any] = field(default_factory=dict)
    fields: ClassVar[frozenset[str]] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {**self.extra}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceConfig":
        return cls(extra={key: value for key, value in data.items()})


class CameraSource(ABC):
    """Abstract camera source interface used by CameraController."""

    supports_hardware_controls = False
    reconnect_delay_seconds = 2.0
    warning_log_interval_seconds: float | None = None
    config_class: ClassVar[type[SourceConfig]] = SourceConfig

    def __init__(self, config: "Camera"):
        self.config = config
        # Set by the owning CameraController via bind_cancel_event() so
        # that open-retry loops and reconnect/frame-pacing waits inside
        # this source can abort promptly when the controller is asked to
        # stop, instead of only noticing on the next iteration.
        self._cancel_event: threading.Event | None = None
        self.source_config = SourceConfig.from_dict(config.source_config)

    def to_dict(self) -> dict[str, Any]:
        return self.source_config.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceConfig:
        return cls.config_class.from_dict(data)

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_frame(self) -> np.ndarray | None:
        raise NotImplementedError

    def apply_settings(self) -> None:
        return

    def apply_preview_settings(self) -> None:
        self.apply_settings()

    def open_for_preview(self) -> None:
        self.open()

    def read_current_settings(self) -> dict[str, object]:
        return {}

    def stop(self) -> None:
        self.close()

    def bind_cancel_event(self, event: threading.Event) -> None:
        """Attach the controller's stop event to this source instance."""
        self._cancel_event = event

    def _cancelled(self) -> bool:
        return self._cancel_event is not None and self._cancel_event.is_set()

    def _wait_or_cancelled(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds, or until cancelled.

        Returns True if the wait ended because of cancellation (so the
        caller should abort early), False if the full timeout elapsed.
        """
        if self._cancel_event is not None:
            return self._cancel_event.wait(timeout)
        time.sleep(timeout)
        return False

    @classmethod
    def list_available_sources(cls) -> list[SourceDescriptor]:
        return []


def validate_source_uri(source_type: CameraSourceType, uri: str) -> str | None:
    """Return an error for an invalid network source URI, or None."""
    allowed_schemes = {
        CameraSourceType.HTTP_SNAPSHOT: ("http", "https"),
        CameraSourceType.HTTP_STREAM: ("http", "https"),
        CameraSourceType.RTSP: ("rtsp", "rtsps"),
    }.get(source_type)
    if allowed_schemes is None:
        return None

    try:
        parsed = urlsplit(uri.strip())
    except ValueError:
        return "URL is malformed"
    if parsed.scheme.lower() not in allowed_schemes:
        schemes = ", ".join(f"{scheme}://" for scheme in allowed_schemes)
        return f"URL must start with one of: {schemes}"
    if not parsed.netloc:
        return "URL must include a host"
    return None


def get_backends_for_platform():
    """Return list of (backend_constant, name) tuples for current platform."""
    import sys

    if sys.platform.startswith("linux"):
        return [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "default"),
        ]
    if sys.platform == "win32":
        return [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MediaFoundation"),
            (cv2.CAP_ANY, "default"),
        ]
    return [(cv2.CAP_ANY, "default")]

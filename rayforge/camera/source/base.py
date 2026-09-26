import ipaddress
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from gettext import gettext as _
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

    placeholder = re.search(r"<([^<>]+)>", uri)
    if placeholder:
        return _(
            "URL contains placeholder <{placeholder}>; "
            "replace it with a valid value"
        ).format(placeholder=placeholder.group(1))

    try:
        parsed = urlsplit(uri.strip())
    except ValueError:
        return _("URL is malformed")
    if parsed.scheme.lower() not in allowed_schemes:
        schemes = ", ".join(f"{scheme}://" for scheme in allowed_schemes)
        return _("URL must start with one of: {schemes}").format(
            schemes=schemes
        )
    if not parsed.netloc:
        return _("URL must include a host")
    if parsed.netloc.rsplit("@", 1)[-1].endswith(":"):
        return _("URL contains an invalid host or port")
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return _("URL contains an invalid host or port")
    if (
        not hostname
        or (port is not None and not 0 <= port <= 65535)
        or not _is_valid_uri_hostname(hostname)
    ):
        return _("URL must include a valid hostname or IP address")
    return None


def _is_valid_uri_hostname(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass

    if re.fullmatch(r"[0-9.]+", hostname) and hostname.count(".") == 3:
        return False

    normalized = hostname.removesuffix(".")
    if not normalized or len(normalized) > 253:
        return False

    for label in normalized.split("."):
        if not label:
            return False
        if "_" in label:
            ascii_label = label
        else:
            try:
                ascii_label = label.encode("idna").decode("ascii")
            except UnicodeError:
                return False
        if len(ascii_label) > 63 or not re.fullmatch(
            r"[A-Za-z0-9_](?:[A-Za-z0-9_-]*[A-Za-z0-9_])?",
            ascii_label,
        ):
            return False
    return True


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

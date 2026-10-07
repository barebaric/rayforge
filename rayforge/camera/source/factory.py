from typing import TYPE_CHECKING

from ..models.source_type import CameraSourceType
from .base import CameraSource
from .local import LocalDeviceConfig, LocalDeviceSource
from .network import (
    HttpSnapshotConfig,
    HttpSnapshotSource,
    HttpStreamSource,
    RtspSource,
    UrlSourceConfig,
)

if TYPE_CHECKING:
    from ..models.camera import Camera


def create_camera_source(config: "Camera") -> CameraSource:
    """Create the runtime source implementation for a camera config."""
    source_type = config.source_type
    if source_type is CameraSourceType.LOCAL_DEVICE:
        return LocalDeviceSource(config)
    if source_type is CameraSourceType.HTTP_SNAPSHOT:
        return HttpSnapshotSource(config)
    if source_type is CameraSourceType.HTTP_STREAM:
        return HttpStreamSource(config)
    if source_type is CameraSourceType.RTSP:
        return RtspSource(config)
    raise ValueError(f"Unsupported camera source type: {source_type}")


def list_local_device_ids() -> list[str]:
    """Return only the raw identifiers for discoverable local cameras."""
    return [
        descriptor.value
        for descriptor in LocalDeviceSource.list_available_sources()
    ]


def source_config_to_dict(
    source_type: CameraSourceType, data: dict[str, object]
) -> dict[str, object]:
    config_type = {
        CameraSourceType.LOCAL_DEVICE: LocalDeviceConfig,
        CameraSourceType.HTTP_SNAPSHOT: HttpSnapshotConfig,
        CameraSourceType.HTTP_STREAM: UrlSourceConfig,
        CameraSourceType.RTSP: UrlSourceConfig,
    }.get(source_type)
    if config_type is None:
        return dict(data)
    return config_type.from_dict(data).to_dict()

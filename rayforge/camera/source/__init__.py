import time
import urllib

import cv2

from .base import (
    CameraSource,
    SourceDescriptor,
    _capture_has_initial_frame,
    _open_local_devices,
    _open_local_devices_lock,
    _to_videocapture_arg,
    get_backends_for_platform,
    validate_source_uri,
)
from .factory import (
    create_camera_source,
    list_local_device_ids,
    source_config_to_dict,
)
from .local import LocalDeviceSource
from .network import (
    HttpSnapshotSource,
    HttpStreamSource,
    OpenCvUrlSource,
    RtspSource,
)

__all__ = [
    "CameraSource",
    "HttpSnapshotSource",
    "HttpStreamSource",
    "LocalDeviceSource",
    "OpenCvUrlSource",
    "RtspSource",
    "SourceDescriptor",
    "_capture_has_initial_frame",
    "_open_local_devices",
    "_open_local_devices_lock",
    "_to_videocapture_arg",
    "create_camera_source",
    "cv2",
    "get_backends_for_platform",
    "list_local_device_ids",
    "source_config_to_dict",
    "time",
    "urllib",
    "validate_source_uri",
]

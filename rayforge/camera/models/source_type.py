from enum import StrEnum


class CameraSourceType(StrEnum):
    """Identifies the kind of backend a camera source uses.

    Deliberately kept in its own module, independent of :mod:`.camera`
    and :mod:`rayforge.camera.source`: both of those need this enum at
    import time, and `.camera` also needs `rayforge.camera.source` at
    import time. Defining it here breaks that cycle instead of relying
    on load-order luck or deferred/local imports.
    """

    LOCAL_DEVICE = "local_device"
    HTTP_SNAPSHOT = "http_snapshot"
    HTTP_STREAM = "http_stream"
    RTSP = "rtsp"

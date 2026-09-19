# flake8: noqa: E402
import gi

gi.require_version("GdkPixbuf", "2.0")
from unittest.mock import patch

import cv2
import numpy as np

from rayforge.camera.controller import (
    CameraController,
    VideoCaptureDevice,
    _scan_cameras_fallback,
)
from rayforge.camera.models.camera import Camera


def test_controller_initialization():
    camera_config = Camera("Test Camera", "device123")
    controller = CameraController(camera_config)
    assert controller.config.name == "Test Camera"
    assert controller.config is camera_config
    assert controller.image_data is None


@patch(
    "rayforge.camera.controller.idle_add",
    side_effect=lambda func, *args, **kwargs: func(*args, **kwargs),
)
def test_capture_image(mock_idle_add):
    original_videocapture = cv2.VideoCapture

    try:

        class MockVideoCapture:
            def __init__(self, device, backend=None):
                self.device = device
                self.opened = True

            def isOpened(self):
                return self.opened

            def read(self):
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                return True, dummy_frame

            def set(self, prop_id, value):
                pass

            def release(self):
                self.opened = False

        cv2.VideoCapture = MockVideoCapture

        camera_config = Camera("Mock Camera", "0")
        controller = CameraController(camera_config)
        controller.capture_image()

        assert controller.image_data is not None
        assert controller.image_data.shape == (480, 640, 3)

    finally:
        cv2.VideoCapture = original_videocapture


def test_video_capture_device_falls_back_when_backend_has_no_frames():
    original_videocapture = cv2.VideoCapture

    try:

        class MockVideoCapture:
            def __init__(self, device, backend=None):
                self.device = device
                self.backend = backend
                self.opened = True

            def isOpened(self):
                return self.opened

            def read(self):
                if self.backend == 1:
                    return False, None
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                return True, dummy_frame

            def release(self):
                self.opened = False

        cv2.VideoCapture = MockVideoCapture

        with patch(
            "rayforge.camera.controller.get_backends_for_platform",
            return_value=[(1, "bad"), (2, "good")],
        ):
            device = VideoCaptureDevice("0")
            with device as cap:
                assert cap is not None
                assert device._backend_used == "good"

    finally:
        cv2.VideoCapture = original_videocapture


def test_list_available_devices_uses_subprocess_scan():
    with patch(
        "rayforge.camera.controller._scan_cameras_in_subprocess",
        return_value=["0", "2"],
    ) as scan_mock:
        assert CameraController.list_available_devices() == ["0", "2"]
        scan_mock.assert_called_once_with()


def test_get_work_surface_image_basic():
    camera_config = Camera("Test Camera", "0")
    controller = CameraController(camera_config)

    controller._image_data = np.zeros((480, 640, 3), dtype=np.uint8)
    controller._image_data[100:400, 100:500] = [
        255,
        255,
        255,
    ]

    image_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
    world_points = [(0, 100), (100, 100), (100, 0), (0, 0)]
    camera_config.image_to_world = (image_points, world_points)

    output_size = (200, 200)
    physical_area = ((0, 0), (100, 100))

    aligned_image = controller.get_work_surface_image(
        output_size, physical_area
    )

    assert aligned_image is not None
    assert aligned_image.shape == (output_size[1], output_size[0], 3)

    center_x, center_y = output_size[0] // 2, output_size[1] // 2
    pixel_value = aligned_image[center_y, center_x]
    np.testing.assert_array_equal(pixel_value, [255, 255, 255])


def test_get_work_surface_image_no_corresponding_points():
    camera_config = Camera("Test Camera", "0")
    controller = CameraController(camera_config)
    controller._image_data = np.zeros((480, 640, 3), dtype=np.uint8)
    output_size = (200, 200)
    physical_area = ((0, 0), (100, 100))

    result = controller.get_work_surface_image(output_size, physical_area)
    assert result is not None
    assert result.shape == (output_size[1], output_size[0], 3)


def test_get_work_surface_image_no_image_data():
    camera_config = Camera("Test Camera", "0")
    image_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
    world_points = [(0, 100), (100, 100), (100, 0), (0, 0)]
    camera_config.image_to_world = (image_points, world_points)

    controller = CameraController(camera_config)

    output_size = (200, 200)
    physical_area = ((0, 0), (100, 100))

    aligned_image = controller.get_work_surface_image(
        output_size, physical_area
    )
    assert aligned_image is None


def test_to_videocapture_arg():
    from rayforge.camera.controller import _to_videocapture_arg

    assert _to_videocapture_arg("0") == 0
    assert _to_videocapture_arg("12") == 12
    assert (
        _to_videocapture_arg("/dev/v4l/by-id/usb-Foo_Webcam-video-index0")
        == "/dev/v4l/by-id/usb-Foo_Webcam-video-index0"
    )


def test_list_available_devices_uses_int_for_numeric_ids():
    """Numeric device IDs must be passed as ints to VideoCapture.

    OpenCV 5.0 treats string arguments as filenames, so probing with
    numeric strings would never find a camera (regression test for
    the "no device found" issue).

    Exercises the in-process scan implementation. The default
    ``list_available_devices()`` runs the probe in a ``spawn``
    subprocess so a bad OpenCV backend cannot crash the app; in-process
    mocks cannot cross that process boundary, so the shared probing /
    int-conversion logic is verified directly here.
    """
    calls = []

    class MockVideoCapture:
        def __init__(self, device, backend=None):
            calls.append(device)

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def release(self):
            pass

    targets = ["0", "1", "/dev/v4l/by-id/usb-Foo_Webcam-video-index0"]

    with (
        patch("rayforge.camera.controller.sys.platform", "linux"),
        patch("rayforge.camera.controller.cv2.VideoCapture", MockVideoCapture),
        patch(
            "rayforge.camera.controller.get_backends_for_platform",
            return_value=[(cv2.CAP_V4L2, "V4L2")],
        ),
        patch(
            "rayforge.camera.controller._get_linux_scan_targets",
            return_value=targets,
        ),
    ):
        devices = _scan_cameras_fallback()

    assert devices == targets
    assert calls == [0, 1, "/dev/v4l/by-id/usb-Foo_Webcam-video-index0"]

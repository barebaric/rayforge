# flake8: noqa: E402
import gi

gi.require_version("GdkPixbuf", "2.0")
import cv2
import numpy as np
from unittest.mock import patch
import pytest
from rayforge.camera.models.camera import Camera
from rayforge.camera.controller import CameraController


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

    with pytest.raises(ValueError):
        controller.get_work_surface_image(output_size, physical_area)


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

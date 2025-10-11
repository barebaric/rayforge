# flake8: noqa: E402
import gi

gi.require_version("GdkPixbuf", "2.0")
import unittest
import cv2
import numpy as np
from rayforge.camera.models.camera import Camera
from rayforge.camera.controller import CameraController


class TestCameraController(unittest.TestCase):
    def test_controller_initialization(self):
        camera_config = Camera("Test Camera", "device123")
        controller = CameraController(camera_config)
        self.assertEqual(controller.config.name, "Test Camera")
        self.assertIs(controller.config, camera_config)
        self.assertIsNone(controller.image_data)

    def test_capture_image(self):
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
                    # Mock the set method for camera properties
                    pass

                def release(self):
                    self.opened = False

            cv2.VideoCapture = MockVideoCapture

            camera_config = Camera("Mock Camera", "0")
            controller = CameraController(camera_config)
            controller.capture_image()

            self.assertIsNotNone(controller.image_data)
            assert controller.image_data is not None  # silence linter
            self.assertEqual(controller.image_data.shape, (480, 640, 3))

        finally:
            cv2.VideoCapture = original_videocapture

    def test_get_work_surface_image_basic(self):
        camera_config = Camera("Test Camera", "0")
        controller = CameraController(camera_config)

        # Set a dummy image data for the controller
        controller._image_data = np.zeros((480, 640, 3), dtype=np.uint8)
        controller._image_data[100:400, 100:500] = [
            255,
            255,
            255,
        ]  # White rectangle

        image_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
        world_points = [(0, 100), (100, 100), (100, 0), (0, 0)]
        camera_config.image_to_world = (image_points, world_points)

        output_size = (200, 200)  # width, height
        physical_area = ((0, 0), (100, 100))  # x_min, y_min, x_max, y_max

        aligned_image = controller.get_work_surface_image(
            output_size, physical_area
        )

        self.assertIsNotNone(aligned_image)
        assert aligned_image is not None  # silence linter
        self.assertEqual(
            aligned_image.shape, (output_size[1], output_size[0], 3)
        )

        # Check if the transformed area is occupied (white)
        # The transformation maps the 100x100 to 500x400 image area (white)
        # to the 0x0 to 100x100 world area.
        # The output image is 200x200, covering the 0x0 to 100x100 world area.
        # So the entire 200x200 output image should be white.
        # We need to check a central region to avoid edge effects.
        center_x, center_y = output_size[0] // 2, output_size[1] // 2
        pixel_value = aligned_image[center_y, center_x]
        np.testing.assert_array_equal(pixel_value, [255, 255, 255])

    def test_get_work_surface_image_no_corresponding_points(self):
        camera_config = Camera("Test Camera", "0")
        controller = CameraController(camera_config)
        controller._image_data = np.zeros((480, 640, 3), dtype=np.uint8)
        output_size = (200, 200)
        physical_area = ((0, 0), (100, 100))

        with self.assertRaises(ValueError):
            controller.get_work_surface_image(output_size, physical_area)

    def test_get_work_surface_image_no_image_data(self):
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
        self.assertIsNone(aligned_image)


if __name__ == "__main__":
    unittest.main()

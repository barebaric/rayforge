# flake8: noqa: E402
import gi
gi.require_version("GdkPixbuf", "2.0")
import unittest
import cv2
import numpy as np
from rayforge.camera.models.camera import Camera


class TestCamera(unittest.TestCase):
    def test_camera_initialization(self):
        camera = Camera("Test Camera", "device123")
        self.assertEqual(camera.name, "Test Camera")
        self.assertEqual(camera.device_id, "device123")
        self.assertFalse(camera.enabled)
        self.assertIsNone(camera.image_data)

    def test_camera_setters(self):
        camera = Camera("Initial Name", "initial_device")
        camera.name = "New Name"
        camera.device_id = "new_device"
        camera.enabled = True
        self.assertEqual(camera.name, "New Name")
        self.assertEqual(camera.device_id, "new_device")
        self.assertTrue(camera.enabled)
        self.assertIsNone(camera.image_data)

    def test_to_from_json(self):
        camera = Camera("JSON Camera", "json_device")
        camera.enabled = True
        json_str = camera.to_json()
        reconstructed_camera = Camera.from_json(json_str)

        self.assertEqual(reconstructed_camera.name, camera.name)
        self.assertEqual(reconstructed_camera.device_id, camera.device_id)
        self.assertEqual(reconstructed_camera.enabled, camera.enabled)
        self.assertIsNone(
            reconstructed_camera.image_data
        )  # image_data not serialized

        # Test image_to_world serialization
        image_points = [
            (1.234, 2.345),
            (3.234, 4.345),
            (5.678, 7.890),
            (9.012, 1.234),
        ]
        world_points = [(0.0, 0.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
        camera.image_to_world = (image_points, world_points)
        json_str_with_points = camera.to_json()
        reconstructed_camera_with_points = Camera.from_json(
            json_str_with_points
        )

        self.assertIsNotNone(reconstructed_camera_with_points.image_to_world)
        assert (
            reconstructed_camera_with_points.image_to_world is not None
        )  # silence linter
        reconstructed_image_points, reconstructed_world_points = (
            reconstructed_camera_with_points.image_to_world
        )

        # Check if points are approximately equal due to float precision
        for i in range(len(image_points)):
            self.assertAlmostEqual(
                reconstructed_image_points[i][0], image_points[i][0]
            )
            self.assertAlmostEqual(
                reconstructed_image_points[i][1], image_points[i][1]
            )
            self.assertAlmostEqual(
                reconstructed_world_points[i][0], world_points[i][0]
            )
            self.assertAlmostEqual(
                reconstructed_world_points[i][1], world_points[i][1]
            )

        # Test with None
        camera.image_to_world = None
        json_str_no_points = camera.to_json()
        reconstructed_camera_no_points = Camera.from_json(json_str_no_points)
        self.assertIsNone(reconstructed_camera_no_points.image_to_world)

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

            camera = Camera("Mock Camera", "0")
            camera.capture_image()

            self.assertIsNotNone(camera.image_data)
            assert camera.image_data is not None  # silence linter
            self.assertEqual(camera.image_data.shape, (480, 640, 3))

        finally:
            cv2.VideoCapture = original_videocapture

    def test_set_image_to_world(self):
        camera = Camera("Test Camera", "0")
        image_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
        world_points = [(0, 100), (100, 100), (100, 0), (0, 0)]
        camera.image_to_world = (image_points, world_points)
        self.assertEqual(len(camera.image_to_world[0]), 4)
        self.assertEqual(camera.image_to_world[0][0], (100, 100))
        self.assertEqual(camera.image_to_world[1][0], (0, 100))

        with self.assertRaises(ValueError):
            camera.image_to_world = ([(0, 0)], [(0, 0)])  # Less than 4

    def test_get_work_surface_image_basic(self):
        camera = Camera("Test Camera", "0")
        # Set a dummy image data for the camera
        camera._image_data = np.zeros((480, 640, 3), dtype=np.uint8)
        camera._image_data[100:400, 100:500] = [
            255,
            255,
            255,
        ]  # White rectangle

        image_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
        world_points = [(0, 100), (100, 100), (100, 0), (0, 0)]
        camera.image_to_world = (image_points, world_points)

        output_size = (200, 200)  # width, height
        physical_area = ((0, 0), (100, 100))  # x_min, y_min, x_max, y_max

        aligned_image = camera.get_work_surface_image(
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
        camera = Camera("Test Camera", "0")
        camera._image_data = np.zeros((480, 640, 3), dtype=np.uint8)
        output_size = (200, 200)
        physical_area = ((0, 0), (100, 100))

        with self.assertRaises(ValueError):
            camera.get_work_surface_image(output_size, physical_area)

    def test_get_work_surface_image_no_image_data(self):
        camera = Camera("Test Camera", "0")
        image_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
        world_points = [(0, 100), (100, 100), (100, 0), (0, 0)]
        camera.image_to_world = (image_points, world_points)
        output_size = (200, 200)
        physical_area = ((0, 0), (100, 100))

        aligned_image = camera.get_work_surface_image(
            output_size, physical_area
        )
        self.assertIsNone(aligned_image)


if __name__ == "__main__":
    unittest.main()

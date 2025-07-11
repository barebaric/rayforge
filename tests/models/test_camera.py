import unittest
import cv2
import numpy as np
from rayforge.models.camera import Camera


class TestCamera(unittest.TestCase):
    def test_camera_initialization(self):
        camera = Camera("Test Camera", "device123")
        self.assertEqual(camera.name, "Test Camera")
        self.assertEqual(camera.device_id, "device123")
        self.assertFalse(camera.enabled)
        self.assertIsNone(camera.image_data)
        self.assertEqual(camera.width_mm, 0.0)
        self.assertEqual(camera.height_mm, 0.0)
        self.assertEqual(camera.transform_matrix, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_camera_setters(self):
        camera = Camera("Initial Name", "initial_device")
        camera.name = "New Name"
        camera.device_id = "new_device"
        camera.enabled = True
        camera.width_mm = 100.5
        camera.height_mm = 75.2
        camera.transform_matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        self.assertEqual(camera.name, "New Name")
        self.assertEqual(camera.device_id, "new_device")
        self.assertTrue(camera.enabled)
        self.assertIsNone(camera.image_data)
        self.assertEqual(camera.width_mm, 100.5)
        self.assertEqual(camera.height_mm, 75.2)
        self.assertEqual(camera.transform_matrix, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_transform_matrix_validation(self):
        camera = Camera("Test", "test_id")
        with self.assertRaises(ValueError):
            camera.transform_matrix = [1.0, 2.0, 3.0]  # Too few elements
        with self.assertRaises(ValueError):
            camera.transform_matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # Too many

    def test_to_from_json(self):
        camera = Camera("JSON Camera", "json_device")
        camera.enabled = True
        camera.width_mm = 200.0
        camera.height_mm = 150.0
        camera.transform_matrix = [0.5, 0.1, 10.0, 0.2, 0.8, 20.0]

        json_str = camera.to_json()
        reconstructed_camera = Camera.from_json(json_str)

        self.assertEqual(reconstructed_camera.name, camera.name)
        self.assertEqual(reconstructed_camera.device_id, camera.device_id)
        self.assertEqual(reconstructed_camera.enabled, camera.enabled)
        self.assertIsNone(reconstructed_camera.image_data) # image_data not serialized
        self.assertEqual(reconstructed_camera.width_mm, camera.width_mm)
        self.assertEqual(reconstructed_camera.height_mm, camera.height_mm)
        self.assertEqual(reconstructed_camera.transform_matrix,
                         camera.transform_matrix)

    def test_capture_image(self):
        original_videocapture = cv2.VideoCapture

        try:
            class MockVideoCapture:
                def __init__(self, device):
                    self.device = device
                    self.opened = True

                def isOpened(self):
                    return self.opened

                def read(self):
                    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    return True, dummy_frame

                def release(self):
                    self.opened = False

            cv2.VideoCapture = MockVideoCapture
            
            camera = Camera("Mock Camera", "0")
            camera.capture_image()

            self.assertIsNotNone(camera.image_data)
            self.assertGreater(camera.width_mm, 0)
            self.assertGreater(camera.height_mm, 0)
            self.assertEqual(camera.image_data.shape, (480, 640, 3))

        finally:
            cv2.VideoCapture = original_videocapture

    def test_apply_affine_transform(self):
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        transform_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        output_size = (100, 100)

        transformed_image = Camera.apply_affine_transform(dummy_image,
                                                          transform_matrix,
                                                          output_size)
        self.assertIsNotNone(transformed_image)
        self.assertEqual(transformed_image.shape, (100, 100, 3))

        transform_matrix_translate = [1.0, 0.0, 10.0, 0.0, 1.0, 20.0]
        transformed_image_translate = Camera.apply_affine_transform(
            dummy_image, transform_matrix_translate, output_size)
        self.assertIsNotNone(transformed_image_translate)

        with self.assertRaises(ValueError):
            Camera.apply_affine_transform(dummy_image, [1.0, 2.0], output_size)

if __name__ == '__main__':
    unittest.main()

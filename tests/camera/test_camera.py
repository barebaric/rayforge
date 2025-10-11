import unittest
from rayforge.camera.models.camera import Camera


class TestCamera(unittest.TestCase):
    def test_camera_initialization(self):
        camera = Camera("Test Camera", "device123")
        self.assertEqual(camera.name, "Test Camera")
        self.assertEqual(camera.device_id, "device123")
        self.assertFalse(camera.enabled)
        self.assertIsNone(camera.image_to_world)

    def test_camera_setters(self):
        camera = Camera("Initial Name", "initial_device")
        camera.name = "New Name"
        camera.device_id = "new_device"
        camera.enabled = True
        self.assertEqual(camera.name, "New Name")
        self.assertEqual(camera.device_id, "new_device")
        self.assertTrue(camera.enabled)

    def test_to_from_json(self):
        camera = Camera("JSON Camera", "json_device")
        camera.enabled = True
        json_str = camera.to_json()
        reconstructed_camera = Camera.from_json(json_str)

        self.assertEqual(reconstructed_camera.name, camera.name)
        self.assertEqual(reconstructed_camera.device_id, camera.device_id)
        self.assertEqual(reconstructed_camera.enabled, camera.enabled)

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


if __name__ == "__main__":
    unittest.main()

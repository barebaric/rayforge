import pytest
from rayforge.camera.models.camera import Camera


def test_camera_initialization():
    camera = Camera("Test Camera", "device123")
    assert camera.name == "Test Camera"
    assert camera.device_id == "device123"
    assert not camera.enabled
    assert camera.image_to_world is None


def test_camera_setters():
    camera = Camera("Initial Name", "initial_device")
    camera.name = "New Name"
    camera.device_id = "new_device"
    camera.enabled = True
    assert camera.name == "New Name"
    assert camera.device_id == "new_device"
    assert camera.enabled


def test_to_from_json():
    camera = Camera("JSON Camera", "json_device")
    camera.enabled = True
    json_str = camera.to_json()
    reconstructed_camera = Camera.from_json(json_str)

    assert reconstructed_camera.name == camera.name
    assert reconstructed_camera.device_id == camera.device_id
    assert reconstructed_camera.enabled == camera.enabled

    image_points = [
        (1.234, 2.345),
        (3.234, 4.345),
        (5.678, 7.890),
        (9.012, 1.234),
    ]
    world_points = [(0.0, 0.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    camera.image_to_world = (image_points, world_points)
    json_str_with_points = camera.to_json()
    reconstructed_camera_with_points = Camera.from_json(json_str_with_points)

    assert reconstructed_camera_with_points.image_to_world is not None
    assert reconstructed_camera_with_points.image_to_world is not None
    reconstructed_image_points, reconstructed_world_points = (
        reconstructed_camera_with_points.image_to_world
    )

    for i in range(len(image_points)):
        assert reconstructed_image_points[i][0] == pytest.approx(
            image_points[i][0]
        )
        assert reconstructed_image_points[i][1] == pytest.approx(
            image_points[i][1]
        )
        assert reconstructed_world_points[i][0] == pytest.approx(
            world_points[i][0]
        )
        assert reconstructed_world_points[i][1] == pytest.approx(
            world_points[i][1]
        )

    camera.image_to_world = None
    json_str_no_points = camera.to_json()
    reconstructed_camera_no_points = Camera.from_json(json_str_no_points)
    assert reconstructed_camera_no_points.image_to_world is None


def test_set_image_to_world():
    camera = Camera("Test Camera", "0")
    image_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
    world_points = [(0, 100), (100, 100), (100, 0), (0, 0)]
    camera.image_to_world = (image_points, world_points)
    assert len(camera.image_to_world[0]) == 4
    assert camera.image_to_world[0][0] == (100, 100)
    assert camera.image_to_world[1][0] == (0, 100)

    with pytest.raises(ValueError):
        camera.image_to_world = ([(0, 0)], [(0, 0)])

import pytest
import numpy as np
from datetime import datetime

from rayforge.camera.models.camera import Camera
from rayforge.camera.calibration.result import CalibrationResult


def test_camera_initialization():
    camera = Camera("Test Camera", "device123")
    assert camera.name == "Test Camera"
    assert camera.device_id == "device123"
    assert not camera.enabled
    assert camera.image_to_world is None
    assert camera.distortion_k1 == 0.0
    assert camera.distortion_k2 == 0.0
    assert camera.distortion_k3 == 0.0
    assert camera.distortion_p1 == 0.0
    assert camera.distortion_p2 == 0.0
    assert camera.camera_matrix_fx is None
    assert camera.camera_matrix_fy is None
    assert camera.camera_matrix_cx is None
    assert camera.camera_matrix_cy is None
    assert not camera.has_calibration


def test_camera_setters():
    camera = Camera("Initial Name", "initial_device")
    camera.name = "New Name"
    camera.device_id = "new_device"
    camera.enabled = True
    assert camera.name == "New Name"
    assert camera.device_id == "new_device"
    assert camera.enabled


def test_distortion_coefficients():
    camera = Camera("Test", "0")
    camera.distortion_k1 = -0.1
    camera.distortion_k2 = 0.05
    camera.distortion_k3 = 0.02
    camera.distortion_p1 = 0.001
    camera.distortion_p2 = -0.002

    assert camera.distortion_k1 == pytest.approx(-0.1)
    assert camera.distortion_k2 == pytest.approx(0.05)
    assert camera.distortion_k3 == pytest.approx(0.02)
    assert camera.distortion_p1 == pytest.approx(0.001)
    assert camera.distortion_p2 == pytest.approx(-0.002)


def test_get_distortion_coeffs():
    camera = Camera("Test", "0")
    camera.distortion_k1 = -0.1
    camera.distortion_k2 = 0.05
    camera.distortion_p1 = 0.001
    camera.distortion_p2 = -0.002
    camera.distortion_k3 = 0.02

    coeffs = camera.get_distortion_coeffs()
    assert coeffs.shape == (5,)
    assert coeffs[0] == pytest.approx(-0.1)
    assert coeffs[1] == pytest.approx(0.05)
    assert coeffs[2] == pytest.approx(0.001)
    assert coeffs[3] == pytest.approx(-0.002)
    assert coeffs[4] == pytest.approx(0.02)


def test_camera_matrix_properties():
    camera = Camera("Test", "0")
    camera.camera_matrix_fx = 800.0
    camera.camera_matrix_fy = 810.0
    camera.camera_matrix_cx = 320.0
    camera.camera_matrix_cy = 240.0

    assert camera.camera_matrix_fx == 800.0
    assert camera.camera_matrix_fy == 810.0
    assert camera.camera_matrix_cx == 320.0
    assert camera.camera_matrix_cy == 240.0
    assert camera.has_calibration


def test_has_calibration():
    camera = Camera("Test", "0")
    assert not camera.has_calibration

    camera.camera_matrix_fx = 800.0
    assert not camera.has_calibration

    camera.camera_matrix_fy = 810.0
    assert not camera.has_calibration

    camera.camera_matrix_cx = 320.0
    assert not camera.has_calibration

    camera.camera_matrix_cy = 240.0
    assert camera.has_calibration


def test_get_camera_matrix():
    camera = Camera("Test", "0")
    assert camera.get_camera_matrix() is None

    camera.camera_matrix_fx = 800.0
    camera.camera_matrix_fy = 810.0
    camera.camera_matrix_cx = 320.0
    camera.camera_matrix_cy = 240.0

    matrix = camera.get_camera_matrix()
    assert matrix is not None
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == pytest.approx(800.0)
    assert matrix[1, 1] == pytest.approx(810.0)
    assert matrix[0, 2] == pytest.approx(320.0)
    assert matrix[1, 2] == pytest.approx(240.0)
    assert matrix[2, 2] == pytest.approx(1.0)


def test_set_calibration_result():
    camera = Camera("Test", "0")
    assert not camera.has_calibration

    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion_coeffs = np.array([-0.1, 0.05, 0.001, -0.002, 0.02])
    calibration_date = datetime(2024, 1, 15, 10, 30, 0)

    result = CalibrationResult(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        rms_error=0.35,
        image_size=(640, 480),
        num_frames_used=15,
        reprojection_errors=[0.1, 0.2, 0.3],
        calibration_date=calibration_date,
    )

    camera.set_calibration_result(result)

    assert camera.has_calibration
    assert camera.camera_matrix_fx == pytest.approx(800.0)
    assert camera.camera_matrix_fy == pytest.approx(810.0)
    assert camera.camera_matrix_cx == pytest.approx(320.0)
    assert camera.camera_matrix_cy == pytest.approx(240.0)

    assert camera.distortion_k1 == pytest.approx(-0.1)
    assert camera.distortion_k2 == pytest.approx(0.05)
    assert camera.distortion_p1 == pytest.approx(0.001)
    assert camera.distortion_p2 == pytest.approx(-0.002)
    assert camera.distortion_k3 == pytest.approx(0.02)

    assert camera.calibration_rms == pytest.approx(0.35)
    assert camera.calibration_date == calibration_date
    assert camera.calibration_image_size == (640, 480)
    assert camera.calibration_frames_used == 15


def test_set_calibration_result_type_error():
    camera = Camera("Test", "0")
    with pytest.raises(TypeError):
        camera.set_calibration_result("not a result")


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


def test_to_from_json_with_calibration():
    camera = Camera("Calibrated Camera", "cal_device")

    camera_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion_coeffs = np.array([-0.1, 0.05, 0.001, -0.002, 0.02])
    calibration_date = datetime(2024, 1, 15, 10, 30, 0)

    result = CalibrationResult(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        rms_error=0.35,
        image_size=(640, 480),
        num_frames_used=15,
        calibration_date=calibration_date,
    )

    camera.set_calibration_result(result)

    json_str = camera.to_json()
    reconstructed = Camera.from_json(json_str)

    assert reconstructed.has_calibration
    assert reconstructed.camera_matrix_fx == pytest.approx(800.0)
    assert reconstructed.camera_matrix_fy == pytest.approx(810.0)
    assert reconstructed.camera_matrix_cx == pytest.approx(320.0)
    assert reconstructed.camera_matrix_cy == pytest.approx(240.0)

    assert reconstructed.distortion_k1 == pytest.approx(-0.1)
    assert reconstructed.distortion_k2 == pytest.approx(0.05)
    assert reconstructed.distortion_p1 == pytest.approx(0.001)
    assert reconstructed.distortion_p2 == pytest.approx(-0.002)
    assert reconstructed.distortion_k3 == pytest.approx(0.02)

    assert reconstructed.calibration_rms == pytest.approx(0.35)
    assert reconstructed.calibration_image_size == (640, 480)
    assert reconstructed.calibration_frames_used == 15


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


def test_settings_changed_signal():
    camera = Camera("Test", "0")
    signal_received = []

    def on_settings_changed(sender):
        signal_received.append(sender)

    camera.settings_changed.connect(on_settings_changed)

    camera.distortion_k1 = 0.1
    assert len(signal_received) == 1

    camera.camera_matrix_fx = 800.0
    assert len(signal_received) == 2

    camera.distortion_k1 = 0.2
    assert len(signal_received) == 3

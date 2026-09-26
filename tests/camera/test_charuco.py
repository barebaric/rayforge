from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from rayforge.camera.calibration.charuco import CharucoConfig, CharucoTarget


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((32, 32), dtype=np.uint8)


def make_target(corners, ids) -> CharucoTarget:
    board = CharucoTarget(CharucoConfig())
    detector = Mock()
    detector.detectBoard.return_value = (corners, ids, None, None)
    board._detector = detector
    return board


@pytest.mark.parametrize("corner_shape", [(4, 1, 2), (4, 2)])
@pytest.mark.parametrize("id_shape", [(4, 1), (4,)])
def test_detect_accepts_common_opencv_shapes(image, corner_shape, id_shape):
    corners = np.arange(8, dtype=np.float32).reshape(corner_shape)
    ids = np.arange(4, dtype=np.int32).reshape(id_shape)

    result = make_target(corners, ids).detect(image)

    assert result == (
        [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0)],
        [0, 1, 2, 3],
    )


@pytest.mark.parametrize(
    ("corners", "ids"),
    [
        (None, None),
        (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        ),
        (
            np.arange(9, dtype=np.float32),
            np.arange(4, dtype=np.int32),
        ),
        (
            np.full((4, 2), "invalid"),
            np.arange(4, dtype=np.int32),
        ),
        (
            np.arange(8, dtype=np.float32).reshape(4, 2),
            np.arange(3, dtype=np.int32),
        ),
    ],
)
def test_detect_rejects_empty_or_invalid_detections(image, corners, ids):
    assert make_target(corners, ids).detect(image) is None


def test_detect_requires_four_corners(image):
    corners = np.arange(6, dtype=np.float32).reshape(3, 2)
    ids = np.arange(3, dtype=np.int32)

    assert make_target(corners, ids).detect(image) is None


def test_detect_generated_board_with_installed_opencv():
    board = CharucoTarget(CharucoConfig())
    image = board.generate_image()

    result = board.detect(image)

    assert result is not None
    corners, ids = result
    assert len(corners) >= 4
    assert len(corners) == len(ids)


def degrade_image(image, scale, blur, contrast, gamma):
    h, w = image.shape[:2]
    size = (int(w * scale), int(h * scale))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), blur)
    small = small.astype(np.float32) * contrast + 128 * (1 - contrast)
    lut = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(np.clip(small, 0, 255).astype(np.uint8), lut)


def test_detect_recovers_corners_from_degraded_image():
    board = CharucoTarget(CharucoConfig())
    degraded = degrade_image(
        board.generate_image(),
        scale=1 / 6,
        blur=1.5,
        contrast=0.55,
        gamma=2.0,
    )

    result = board.detect(degraded)

    assert result is not None
    corners, ids = result
    assert len(corners) >= 4
    assert len(corners) == len(ids)


def test_detect_returns_no_corners_for_noise():
    board = CharucoTarget(CharucoConfig())
    noise = np.random.default_rng(42).integers(
        0, 256, size=(480, 640), dtype=np.uint8
    )

    assert board.detect(noise) is None

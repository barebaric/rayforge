from unittest.mock import Mock

import numpy as np
import pytest

from rayforge.camera.calibration.charuco import CharucoBoard, CharucoConfig


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((32, 32), dtype=np.uint8)


def make_board(corners, ids) -> CharucoBoard:
    board = CharucoBoard(CharucoConfig())
    detector = Mock()
    detector.detectBoard.return_value = (corners, ids, None, None)
    board._detector = detector
    return board


@pytest.mark.parametrize("corner_shape", [(4, 1, 2), (4, 2)])
@pytest.mark.parametrize("id_shape", [(4, 1), (4,)])
def test_detect_accepts_common_opencv_shapes(image, corner_shape, id_shape):
    corners = np.arange(8, dtype=np.float32).reshape(corner_shape)
    ids = np.arange(4, dtype=np.int32).reshape(id_shape)

    result = make_board(corners, ids).detect(image)

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
    assert make_board(corners, ids).detect(image) is None


def test_detect_requires_four_corners(image):
    corners = np.arange(6, dtype=np.float32).reshape(3, 2)
    ids = np.arange(3, dtype=np.int32)

    assert make_board(corners, ids).detect(image) is None


def test_detect_generated_board_with_installed_opencv():
    board = CharucoBoard(CharucoConfig())
    image = board.generate_image()

    result = board.detect(image)

    assert result is not None
    corners, ids = result
    assert len(corners) >= 4
    assert len(corners) == len(ids)

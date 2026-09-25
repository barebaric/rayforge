import logging
from dataclasses import dataclass
from typing import ClassVar

import cv2
import numpy as np

from .target import (
    CalibrationTarget,
    CalibrationTargetType,
    ConfigField,
    SummaryRow,
    TargetConfig,
    normalize_detection,
    to_gray,
)

logger = logging.getLogger(__name__)


@dataclass
class CharucoConfig(TargetConfig):
    squares_x: int = 5
    squares_y: int = 7
    square_length_mm: float = 20.0
    marker_length_mm: float = 15.0
    dictionary_id: int = cv2.aruco.DICT_6X6_250

    FIELDS: ClassVar[tuple[ConfigField, ...]] = (
        ConfigField(key="squares_x", lower=3, upper=12, integer=True),
        ConfigField(key="squares_y", lower=3, upper=14, integer=True),
        ConfigField(key="square_length_mm", lower=1.0, upper=300.0, step=0.5),
        ConfigField(key="marker_length_mm", lower=1.0, upper=300.0, step=0.5),
    )


class CharucoTarget(CalibrationTarget):
    """ChArUco board: a chessboard carrying ArUco markers in its squares.

    Detection yields the chessboard corner intersections rather than the
    marker corners, which is what gives a ChArUco board its higher
    angular accuracy compared to a plain marker grid.
    """

    target_type = CalibrationTargetType.CHARUCO
    config_class = CharucoConfig
    MIN_SQUARES_X = 4
    MIN_SQUARES_Y = 5
    MIN_MARKER_PIXELS = 10

    def __init__(self, config: CharucoConfig):
        self.config = config
        self._board = None
        self._detector = None
        self._create_board()

    def _create_board(self):
        dictionary = cv2.aruco.getPredefinedDictionary(
            self.config.dictionary_id
        )
        self._board = cv2.aruco.CharucoBoard(
            size=(self.config.squares_x, self.config.squares_y),
            squareLength=self.config.square_length_mm,
            markerLength=self.config.marker_length_mm,
            dictionary=dictionary,
        )
        charuco_params = cv2.aruco.CharucoParameters()
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector_params.cornerRefinementWinSize = 5
        detector_params.cornerRefinementMaxIterations = 30
        detector_params.cornerRefinementMinAccuracy = 0.1
        refine_params = cv2.aruco.RefineParameters()
        self._detector = cv2.aruco.CharucoDetector(
            self._board, charuco_params, detector_params, refine_params
        )

    @property
    def board(self):
        return self._board

    @property
    def point_count(self) -> int:
        return (self.config.squares_x - 1) * (self.config.squares_y - 1)

    @property
    def card_size_mm(self) -> tuple[float, float]:
        return (
            self.config.squares_x * self.config.square_length_mm,
            self.config.squares_y * self.config.square_length_mm,
        )

    def object_points(self) -> np.ndarray:
        assert self._board is not None
        return self._board.getChessboardCorners()

    @classmethod
    def recommend_config(
        cls,
        card_width_mm: float,
        card_height_mm: float,
        camera_resolution: tuple[int, int] = (640, 480),
        surface_size_mm: tuple[float, float] | None = None,
    ) -> CharucoConfig:
        min_marker_pixels = cls.MIN_MARKER_PIXELS
        min_dim = min(camera_resolution)

        if surface_size_mm:
            surface_min = min(surface_size_mm)
            mm_per_pixel = surface_min / min_dim
        else:
            mm_per_pixel = card_width_mm / min_dim

        target_square_size = 20.0
        target_squares_x = max(
            cls.MIN_SQUARES_X, int(card_width_mm / target_square_size)
        )
        target_squares_y = max(
            cls.MIN_SQUARES_Y, int(card_height_mm / target_square_size)
        )

        square_length = min(
            card_width_mm / target_squares_x,
            card_height_mm / target_squares_y,
        )

        estimated_marker_pixels = (square_length * 0.75) / mm_per_pixel
        if estimated_marker_pixels < min_marker_pixels:
            scale = min_marker_pixels / estimated_marker_pixels
            square_length *= scale

        squares_x = max(cls.MIN_SQUARES_X, int(card_width_mm / square_length))
        squares_y = max(cls.MIN_SQUARES_Y, int(card_height_mm / square_length))

        squares_x = min(squares_x, 12)
        squares_y = min(squares_y, 14)

        marker_length = square_length * 0.75

        return CharucoConfig(
            squares_x=squares_x,
            squares_y=squares_y,
            square_length_mm=round(square_length, 1),
            marker_length_mm=round(marker_length, 1),
        )

    def generate_image(
        self,
        output_size: tuple[int, int] | None = None,
        margin_px: int = 10,
        border_bits: int = 1,
    ) -> np.ndarray:
        if output_size is None:
            px_per_mm = 10
            w = int(
                self.config.squares_x
                * self.config.square_length_mm
                * px_per_mm
            )
            h = int(
                self.config.squares_y
                * self.config.square_length_mm
                * px_per_mm
            )
            output_size = (w + 2 * margin_px, h + 2 * margin_px)

        assert self._board is not None
        image = self._board.generateImage(output_size, marginSize=margin_px)
        return image

    def detect(
        self, image: np.ndarray
    ) -> tuple[list[tuple[float, float]], list[int]] | None:
        gray = to_gray(image)

        assert self._detector is not None
        try:
            charuco_corners, charuco_ids, _, _ = self._detector.detectBoard(
                gray
            )
        except cv2.error as error:
            logger.debug("ChArUco detection error: %s", error)
            return None

        return normalize_detection(charuco_corners, charuco_ids)

    def summary(self) -> list[SummaryRow]:
        return [
            SummaryRow(
                key="grid_size",
                value=(
                    f"{self.config.squares_x} x {self.config.squares_y}"
                    " squares"
                ),
            ),
            SummaryRow(
                key="square_size",
                measurement_mm=self.config.square_length_mm,
            ),
        ]

    def draw_detection(
        self,
        image: np.ndarray,
        corners: list[tuple[float, float]],
        ids: list[int],
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        result = image.copy()

        charuco_corners = np.array(
            [[[c[0], c[1]]] for c in corners], dtype=np.float32
        )
        charuco_ids = np.array([[i] for i in ids], dtype=np.int32)

        cv2.aruco.drawDetectedCornersCharuco(
            result, charuco_corners, charuco_ids, color
        )

        return result


__all__ = ["CharucoConfig", "CharucoTarget"]

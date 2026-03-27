import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CharucoConfig:
    squares_x: int = 5
    squares_y: int = 7
    square_length_mm: float = 20.0
    marker_length_mm: float = 15.0
    dictionary_id: int = cv2.aruco.DICT_6X6_250

    def to_dict(self) -> dict:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_mm": self.square_length_mm,
            "marker_length_mm": self.marker_length_mm,
            "dictionary_id": self.dictionary_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CharucoConfig":
        return cls(
            squares_x=data.get("squares_x", 5),
            squares_y=data.get("squares_y", 7),
            square_length_mm=data.get("square_length_mm", 20.0),
            marker_length_mm=data.get("marker_length_mm", 15.0),
            dictionary_id=data.get("dictionary_id", cv2.aruco.DICT_6X6_250),
        )


class CharucoBoard:
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
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        refine_params = cv2.aruco.RefineParameters()
        self._detector = cv2.aruco.CharucoDetector(
            self._board, charuco_params, detector_params, refine_params
        )

    @property
    def board(self):
        return self._board

    @property
    def chessboard_corners(self) -> int:
        return (self.config.squares_x - 1) * (self.config.squares_y - 1)

    @property
    def card_size_mm(self) -> Tuple[float, float]:
        return (
            self.config.squares_x * self.config.square_length_mm,
            self.config.squares_y * self.config.square_length_mm,
        )

    @classmethod
    def recommend_config(
        cls,
        card_width_mm: float,
        card_height_mm: float,
        camera_resolution: Tuple[int, int] = (640, 480),
        surface_size_mm: Optional[Tuple[float, float]] = None,
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
        output_size: Optional[Tuple[int, int]] = None,
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
    ) -> Optional[Tuple[List[Tuple[float, float]], List[int]]]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        assert self._detector is not None
        try:
            charuco_corners, charuco_ids, _, _ = self._detector.detectBoard(
                gray
            )
        except cv2.error as e:
            logger.debug(f"ChArUco detection error: {e}")
            return None

        if charuco_corners is None or charuco_ids is None:
            return None

        if len(charuco_corners) < 4:
            return None

        corners = [
            (float(pt[0][0]), float(pt[0][1])) for pt in charuco_corners
        ]
        ids = [int(i[0]) for i in charuco_ids]

        return corners, ids

    def draw_detection(
        self,
        image: np.ndarray,
        corners: List[Tuple[float, float]],
        ids: List[int],
        color: Tuple[int, int, int] = (0, 255, 0),
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

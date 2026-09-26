"""Shared interface for printable calibration targets."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import ClassVar

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# A frame is only usable for calibration if at least this many points are
# found in it.
MIN_DETECTED_POINTS = 4


class CalibrationTargetType(StrEnum):
    """Identifies the kind of printed pattern used for calibration."""

    CHARUCO = "charuco"
    # The marker grid covers both ArUco and AprilTag dictionaries; the
    # value stays historic so stored configurations keep loading.
    ARUCO_GRID = "aruco_grid"
    DOT_GRID = "dot_grid"


@dataclass(frozen=True)
class ConfigField:
    """An editable numeric field of a target configuration.

    ``key`` names both the configuration attribute and the lookup key
    the user interface uses to find a display label. Keeping the label
    itself out of this module lets the interface own the translatable
    text, which is what puts it in the message catalogue.
    """

    key: str
    lower: float
    upper: float
    step: float = 1.0
    integer: bool = False


@dataclass(frozen=True)
class SummaryRow:
    """A read-only description line shown next to a target preview.

    As with :class:`ConfigField`, ``key`` is a lookup key rather than
    display text.
    """

    key: str
    value: str | None = None
    measurement_mm: float | None = None


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single channel view of an arbitrary frame."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image


def normalize_detection(
    corners,
    ids,
    min_points: int = MIN_DETECTED_POINTS,
) -> tuple[list[tuple[float, float]], list[int]] | None:
    """Coerce raw detector output into plain Python types.

    OpenCV returns slightly different shapes depending on the detector
    and version, so every target funnels its result through here. Returns
    None when the detection cannot be used for calibration.
    """
    if corners is None or ids is None:
        return None

    try:
        corners_array = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        ids_array = np.asarray(ids).reshape(-1)
    except (TypeError, ValueError) as error:
        logger.debug("Invalid detection result: %s", error)
        return None

    if len(corners_array) < min_points:
        return None
    if len(corners_array) != len(ids_array):
        return None

    try:
        points = [(float(x), float(y)) for x, y in corners_array]
        point_ids = [int(value) for value in ids_array]
    except (TypeError, ValueError, OverflowError) as error:
        logger.debug("Invalid detection result: %s", error)
        return None

    return points, point_ids


@dataclass
class TargetConfig:
    """Base class for the configuration payload of a target."""

    FIELDS: ClassVar[tuple[ConfigField, ...]] = ()

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name) for field in fields(self)
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TargetConfig":
        known = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in data.items() if key in known}
        return cls(**kwargs)

    def to_field_values(self) -> dict[str, float]:
        """Return the editable field values keyed by field name."""
        return {
            field.key: float(getattr(self, field.key)) for field in self.FIELDS
        }

    def apply_field_values(self, values: dict[str, float]) -> None:
        """Set editable fields from raw spin row values."""
        for field in self.FIELDS:
            if field.key not in values:
                continue
            value = values[field.key]
            setattr(
                self, field.key, int(value) if field.integer else float(value)
            )


class CalibrationTarget(ABC):
    """A printed pattern that can be detected and solved for intrinsics.

    Implementations only have to describe their own geometry. Resolving
    detections into a camera model is the job of
    :class:`~.calibrator.CameraCalibrator`, which works entirely through
    this interface.
    """

    target_type: ClassVar[CalibrationTargetType]
    config_class: ClassVar[type[TargetConfig]]

    @classmethod
    @abstractmethod
    def recommend_config(
        cls,
        card_width_mm: float,
        card_height_mm: float,
        camera_resolution: tuple[int, int] = (640, 480),
        surface_size_mm: tuple[float, float] | None = None,
    ) -> TargetConfig:
        """Return a configuration sized to fill a printable card."""

    @abstractmethod
    def detect(
        self, image: np.ndarray
    ) -> tuple[list[tuple[float, float]], list[int]] | None:
        """Return detected points and their object point indices."""

    @abstractmethod
    def object_points(self) -> np.ndarray:
        """Return the (N, 3) object point table indexed by detection id."""

    @property
    @abstractmethod
    def point_count(self) -> int:
        """Number of distinct points this target can yield."""

    @property
    @abstractmethod
    def card_size_mm(self) -> tuple[float, float]:
        """Physical size of the printed pattern, in millimetres."""

    @abstractmethod
    def generate_image(
        self, output_size: tuple[int, int] | None = None
    ) -> np.ndarray:
        """Render the pattern so it can be printed or previewed."""

    def summary(self) -> list[SummaryRow]:
        """Return read-only description rows for the calibration UI."""
        return []

    def draw_detection(
        self,
        image: np.ndarray,
        corners: list[tuple[float, float]],
        ids: list[int],
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Return a copy of the image with detections drawn on it."""
        return image.copy()


__all__ = [
    "MIN_DETECTED_POINTS",
    "CalibrationTarget",
    "CalibrationTargetType",
    "ConfigField",
    "SummaryRow",
    "TargetConfig",
    "normalize_detection",
    "to_gray",
]

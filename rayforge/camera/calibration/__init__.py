from .aruco import ArucoGridConfig, ArucoGridTarget
from .calibrator import CameraCalibrator
from .charuco import CharucoConfig, CharucoTarget
from .dotgrid import DotGridConfig, DotGridTarget
from .result import CalibrationResult
from .target import (
    CalibrationTarget,
    CalibrationTargetType,
    ConfigField,
    SummaryRow,
    TargetConfig,
)

TARGET_CLASSES: dict[CalibrationTargetType, type[CalibrationTarget]] = {
    CalibrationTargetType.CHARUCO: CharucoTarget,
    CalibrationTargetType.ARUCO_GRID: ArucoGridTarget,
    CalibrationTargetType.DOT_GRID: DotGridTarget,
}


def get_target_class(
    target_type: CalibrationTargetType,
) -> type[CalibrationTarget]:
    """Return the target implementation for a target type."""
    try:
        return TARGET_CLASSES[target_type]
    except KeyError:
        raise ValueError(f"Unsupported calibration target: {target_type}")


def create_target(
    target_type: CalibrationTargetType,
    config: dict | None = None,
) -> CalibrationTarget:
    """Build a target from a stored configuration dictionary.

    Missing keys fall back to the target's own defaults, so a partially
    written configuration still yields a usable target.
    """
    target_class = get_target_class(target_type)
    config_class = target_class.config_class
    payload = dict(config) if config else {}
    return target_class(config_class.from_dict(payload))


def recommend_target(
    target_type: CalibrationTargetType,
    card_width_mm: float,
    card_height_mm: float,
) -> CalibrationTarget:
    """Build a target sized to fill a printable card."""
    target_class = get_target_class(target_type)
    config = target_class.recommend_config(
        card_width_mm=card_width_mm,
        card_height_mm=card_height_mm,
    )
    return target_class(config)


def available_target_types() -> list[CalibrationTargetType]:
    """Return every selectable target type, in presentation order."""
    return [
        CalibrationTargetType.CHARUCO,
        CalibrationTargetType.ARUCO_GRID,
        CalibrationTargetType.DOT_GRID,
    ]


__all__ = [
    "TARGET_CLASSES",
    "ArucoGridConfig",
    "ArucoGridTarget",
    "CalibrationResult",
    "CalibrationTarget",
    "CalibrationTargetType",
    "CameraCalibrator",
    "CharucoConfig",
    "CharucoTarget",
    "ConfigField",
    "DotGridConfig",
    "DotGridTarget",
    "SummaryRow",
    "TargetConfig",
    "available_target_types",
    "create_target",
    "get_target_class",
    "recommend_target",
]

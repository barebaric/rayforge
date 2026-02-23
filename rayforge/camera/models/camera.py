import json
from typing import Optional, Tuple, Dict, Sequence, Any
import numpy as np
import logging
from blinker import Signal


logger = logging.getLogger(__name__)
Pos = Tuple[float, float]
PointList = Sequence[Pos]


class Camera:
    """A pure data model representing the configuration of a camera."""

    def __init__(self, name: str, device_id: str):
        self._name: str = name
        self._device_id: str = device_id
        self._enabled: bool = False
        # None indicates auto white balance, float for manual Kelvin
        self._white_balance: Optional[float] = None
        self._contrast: float = 50.0
        self._brightness: float = 0.0  # Default brightness (0 = no change)
        self._transparency: float = 0.2

        # Noise Reduction (0.0 to 1.0)
        # 0.0 = No denoise, 1.0 = Max smoothing (high latency/ghosting)
        self._denoise: float = 0.0

        # Fisheye / wide-angle distortion parameters
        self._distortion_k1: float = 0.0
        self._distortion_k2: float = 0.0
        self._distortion_p1: float = 0.0
        self._distortion_p2: float = 0.0

        # Positioning and stretching offsets
        self._offset_top: float = 0.0
        self._offset_bottom: float = 0.0
        self._offset_left: float = 0.0
        self._offset_right: float = 0.0

        # Properties for camera calibration
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None

        # Properties for camera alignment points
        self._image_to_world: Optional[Tuple[PointList, PointList]] = None

        # Signals
        self.changed = Signal()
        self.settings_changed = Signal()
        self.extra: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        if self._name == value:
            return
        logger.debug(f"Camera name changed from '{self._name}' to '{value}'")
        self._name = value
        self.changed.send(self)

    @property
    def device_id(self) -> str:
        return self._device_id

    @device_id.setter
    def device_id(self, value: str):
        if self._device_id == value:
            return
        logger.debug(
            f"Camera device_id changed from '{self._device_id}' to '{value}'"
        )
        self._device_id = value
        self.changed.send(self)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        if self._enabled == value:
            return
        logger.debug(f"Camera enabled changed from {self._enabled} to {value}")
        self._enabled = value
        self.changed.send(self)

    @property
    def white_balance(self) -> Optional[float]:
        return self._white_balance

    @white_balance.setter
    def white_balance(self, value: Optional[float]):
        if value is not None:
            if not isinstance(value, (int, float)):
                raise ValueError("White balance must be a number or None.")
            if not (2500 <= value <= 10000):
                logger.warning(
                    f"White balance value {value} is outside range "
                    "(2500-10000). Clamping to nearest bound."
                )
                value = max(2500, min(value, 10000))
        if self._white_balance == value:
            return
        logger.debug(
            f"Camera white_balance changed from {self._white_balance} to "
            f"{value}"
        )
        self._white_balance = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def contrast(self) -> float:
        return self._contrast

    @contrast.setter
    def contrast(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("Contrast must be a number.")
        if not (0.0 <= value <= 100.0):
            logger.warning(
                f"Contrast value {value} is outside range (0.0-100.0). "
                "Clamping to nearest bound."
            )
            value = max(0.0, min(value, 100.0))
        if self._contrast == value:
            return
        logger.debug(
            f"Camera contrast changed from {self._contrast} to {value}"
        )
        self._contrast = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def brightness(self) -> float:
        return self._brightness

    @brightness.setter
    def brightness(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("Brightness must be a number.")
        if not (-100.0 <= value <= 100.0):
            logger.warning(
                f"Brightness value {value} is outside range (-100.0-100.0). "
                "Clamping to nearest bound."
            )
            value = max(-100.0, min(value, 100.0))
        if self._brightness == value:
            return
        logger.debug(
            f"Camera brightness changed from {self._brightness} to {value}"
        )
        self._brightness = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def transparency(self) -> float:
        return self._transparency

    @transparency.setter
    def transparency(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("Transparency must be a number.")
        if not (0.0 <= value <= 1.0):
            logger.warning(
                f"Transparency value {value} is outside range (0.0-1.0). "
                "Clamping to nearest bound."
            )
            value = max(0.0, min(value, 1.0))
        if self._transparency == value:
            return
        logger.debug(
            f"Camera transparency changed from {self._transparency} to {value}"
        )
        self._transparency = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def denoise(self) -> float:
        """Temporal noise reduction factor (0.0 - 1.0)."""
        return self._denoise

    @denoise.setter
    def denoise(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("Denoise must be a number.")
        # Clamp between 0.0 (off) and 0.95 (extreme smoothing)
        value = max(0.0, min(value, 0.95))
        if self._denoise == value:
            return
        logger.debug(f"Camera denoise changed from {self._denoise} to {value}")
        self._denoise = value
        self.changed.send(self)
        self.settings_changed.send(self)

    # --- Fisheye/Distortion Properties ---

    @property
    def distortion_k1(self) -> float:
        return self._distortion_k1

    @distortion_k1.setter
    def distortion_k1(self, value: float):
        self._distortion_k1 = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def distortion_k2(self) -> float:
        return self._distortion_k2

    @distortion_k2.setter
    def distortion_k2(self, value: float):
        self._distortion_k2 = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def distortion_p1(self) -> float:
        return self._distortion_p1

    @distortion_p1.setter
    def distortion_p1(self, value: float):
        self._distortion_p1 = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def distortion_p2(self) -> float:
        return self._distortion_p2

    @distortion_p2.setter
    def distortion_p2(self, value: float):
        self._distortion_p2 = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    # --- Positioning & Stretching Properties ---

    @property
    def offset_top(self) -> float:
        return self._offset_top

    @offset_top.setter
    def offset_top(self, value: float):
        self._offset_top = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def offset_bottom(self) -> float:
        return self._offset_bottom

    @offset_bottom.setter
    def offset_bottom(self, value: float):
        self._offset_bottom = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def offset_left(self) -> float:
        return self._offset_left

    @offset_left.setter
    def offset_left(self, value: float):
        self._offset_left = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def offset_right(self) -> float:
        return self._offset_right

    @offset_right.setter
    def offset_right(self, value: float):
        self._offset_right = float(value)
        self.changed.send(self)
        self.settings_changed.send(self)

    # ---------------------------------------------

    @property
    def image_to_world(self) -> Optional[Tuple[PointList, PointList]]:
        return self._image_to_world

    @image_to_world.setter
    def image_to_world(self, value: Optional[Tuple[PointList, PointList]]):
        if value is not None:
            if not (isinstance(value, tuple) and len(value) == 2):
                raise ValueError(
                    "Corresponding points must be a tuple of two point lists."
                )
            image_points, world_points = value
            if not (
                isinstance(image_points, Sequence)
                and isinstance(world_points, Sequence)
            ):
                raise ValueError(
                    "Both elements of corresponding points must be sequences."
                )
            if len(image_points) < 4 or len(world_points) < 4:
                raise ValueError(
                    "At least 4 corresponding points are required."
                )
            if len(image_points) != len(world_points):
                raise ValueError(
                    "Image points and world points must have the same number "
                    "of entries."
                )
            for points in [image_points, world_points]:
                for p in points:
                    if not (
                        isinstance(p, tuple)
                        and len(p) == 2
                        and isinstance(p[0], (int, float))
                        and isinstance(p[1], (int, float))
                    ):
                        raise ValueError(
                            "Each point must be a tuple of two floats "
                            "(e.g., (x, y))."
                        )
        if self._image_to_world == value:
            return
        logger.debug(
            f"Camera image_to_world changed from "
            f"{self._image_to_world} to {value}"
        )
        self._image_to_world = value
        self.changed.send(self)
        self.settings_changed.send(self)

    def set_camera_calibration(
        self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ):
        """
        Set the camera calibration parameters for distortion correction.

        Args:
            camera_matrix: 3x3 camera matrix from calibration
            dist_coeffs: Distortion coefficients from calibration
        """
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        logger.debug("Camera calibration parameters set.")
        self.changed.send(self)
        self.settings_changed.send(self)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "device_id": self.device_id,
            "enabled": self.enabled,
            "white_balance": self.white_balance,
            "contrast": self.contrast,
            "brightness": self.brightness,
            "transparency": self.transparency,
            "denoise": self.denoise,
            
            "distortion_k1": self.distortion_k1,
            "distortion_k2": self.distortion_k2,
            "distortion_p1": self.distortion_p1,
            "distortion_p2": self.distortion_p2,

            "offset_top": self.offset_top,
            "offset_bottom": self.offset_bottom,
            "offset_left": self.offset_left,
            "offset_right": self.offset_right,
        }
        if self.image_to_world is not None:
            image_points, world_points = self.image_to_world
            data["image_to_world"] = [
                {
                    "image": f"{img[0]}, {img[1]}",
                    "world": f"{wld[0]}, {wld[1]}",
                }
                for img, wld in zip(image_points, world_points)
            ]
        else:
            data["image_to_world"] = None
        data.update(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Camera":
        known_keys = {
            "name",
            "device_id",
            "enabled",
            "white_balance",
            "contrast",
            "brightness",
            "transparency",
            "denoise",
            "image_to_world",
            "distortion_k1", "distortion_k2", "distortion_p1", "distortion_p2",
            "offset_top", "offset_bottom", "offset_left", "offset_right"
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        camera = cls(data["name"], data["device_id"])
        camera.enabled = data.get("enabled", camera.enabled)
        camera.white_balance = data.get("white_balance", None)
        camera.contrast = data.get("contrast", camera.contrast)
        camera.brightness = data.get("brightness", camera.brightness)
        camera.transparency = data.get("transparency", camera.transparency)
        camera.denoise = data.get("denoise", 0.0)

        camera.distortion_k1 = data.get("distortion_k1", 0.0)
        camera.distortion_k2 = data.get("distortion_k2", 0.0)
        camera.distortion_p1 = data.get("distortion_p1", 0.0)
        camera.distortion_p2 = data.get("distortion_p2", 0.0)

        camera.offset_top = data.get("offset_top", 0.0)
        camera.offset_bottom = data.get("offset_bottom", 0.0)
        camera.offset_left = data.get("offset_left", 0.0)
        camera.offset_right = data.get("offset_right", 0.0)

        image_to_world_data = data.get("image_to_world")
        if image_to_world_data is not None:
            image_points = []
            world_points = []
            for entry in image_to_world_data:
                image_str = entry["image"].split(",")
                world_str = entry["world"].split(",")
                image_points.append(
                    (float(image_str[0].strip()), float(image_str[1].strip()))
                )
                world_points.append(
                    (float(world_str[0].strip()), float(world_str[1].strip()))
                )
            camera.image_to_world = (image_points, world_points)
        else:
            camera.image_to_world = None

        camera.extra = extra
        return camera

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.from_dict(data)

import json
import threading
from typing import Optional, List, Tuple, Dict, Any
import cv2
import numpy as np
import logging
from blinker import Signal
from gi.repository import GLib, GdkPixbuf
from ..util.glib import idle_add


logger = logging.getLogger(__name__)


class VideoCaptureDevice:
    def __init__(self, device_id):
        self.device_id = device_id
        self.cap = None

    def __enter__(self):
        if isinstance(self.device_id, str) and self.device_id.isdigit():
            device_id_int = int(self.device_id)
        else:
            device_id_int = self.device_id

        self.cap = cv2.VideoCapture(device_id_int)
        if not self.cap.isOpened():
            raise IOError(
                f"Cannot open camera with device ID: {self.device_id}"
            )
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


class Camera:
    def __init__(self, name: str, device_id: str):
        self._name: str = name
        self._device_id: str = device_id
        self._enabled: bool = False
        self._image_data: Optional[np.ndarray] = None
        self._transform_matrix: List[float] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self._white_balance: float = 4000.0  # Default white balance in Kelvin
        self._contrast: float = 50.0
        self._brightness: float = 0.0  # Default brightness (0 = no change)
        self._capture_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self.changed = Signal()
        self.image_captured = Signal()
        self.settings_changed = Signal()  # New signal for settings changes

    @property
    def pixbuf(self) -> Optional[GdkPixbuf.Pixbuf]:
        if self._image_data is None:
            return None
        height, width, channels = self._image_data.shape
        if channels == 3:
            # OpenCV uses BGR, GdkPixbuf expects RGB
            np_array = cv2.cvtColor(self._image_data, cv2.COLOR_BGR2RGB)
            has_alpha = False
        elif channels == 4:
            np_array = self._image_data
            has_alpha = True
        else:
            return None

        # Ensure the array is contiguous
        np_array = np.ascontiguousarray(np_array)

        # Create GBytes from the numpy array
        pixels = GLib.Bytes.new(np_array.tobytes())

        pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
            pixels,
            GdkPixbuf.Colorspace.RGB,
            has_alpha,
            8,  # bits per sample
            width,
            height,
            width * channels,  # rowstride
        )
        return pixbuf

    @staticmethod
    def list_available_devices() -> List[str]:
        """
        Lists available camera device IDs.
        Returns a list of strings, where each string is a device ID.
        """
        available_devices = []
        # Try device IDs from 0 up to a reasonable number (e.g., 10)
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_devices.append(str(i))
                cap.release()
        logger.debug("Found available camera devices: %s", available_devices)
        return available_devices

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        if self._name == value:
            return
        logger.debug(
            "Camera name changed from '%s' to '%s'", self._name, value
        )
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
            "Camera device_id changed from '%s' to '%s'",
            self._device_id, value
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
        logger.debug(
            "Camera enabled changed from %s to %s", self._enabled, value
        )
        self._enabled = value
        if self._enabled:
            self._start_capture_stream()
        else:
            self._stop_capture_stream()
        self.changed.send(self)

    @property
    def image_data(self) -> Optional[np.ndarray]:
        return self._image_data

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._image_data is None:
            return 640, 480
        height, width, _ = self._image_data.shape
        return width, height

    @property
    def aspect(self) -> float:
        resolution = self.resolution
        return resolution[1] / resolution[0]

    @property
    def transform_matrix(self) -> List[float]:
        return self._transform_matrix

    @transform_matrix.setter
    def transform_matrix(self, value: List[float]):
        if not isinstance(value, list) or len(value) != 6:
            raise ValueError("Transform matrix must be a list of 6 floats.")
        if self._transform_matrix == value:
            return
        logger.debug(
            "Camera transform_matrix changed from %s to %s",
            self._transform_matrix, value
        )
        self._transform_matrix = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def white_balance(self) -> float:
        return self._white_balance

    @white_balance.setter
    def white_balance(self, value: float):
        if not isinstance(value, (int, float)):
            raise ValueError("White balance must be a number.")
        if not (2500 <= value <= 10000):
            logger.warning(
                f"White balance value {value} is outside range (2500-10000). "
                "Clamping to nearest bound."
            )
            value = max(2500, min(value, 10000))
        if self._white_balance == value:
            return
        logger.debug(
            "Camera white_balance changed from %f to %f",
            self._white_balance, value
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
            "Camera contrast changed from %f to %f", self._contrast, value
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
            "Camera brightness changed from %f to %f", self._brightness, value
        )
        self._brightness = value
        self.changed.send(self)
        self.settings_changed.send(self)

    def _read_frame_and_update_data(self, cap: cv2.VideoCapture):
        """
        Reads a single frame from the given VideoCapture object,
        updates camera data, and emits the image_captured signal.
        """
        try:
            # Apply white balance and contrast settings
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.white_balance)
            cap.set(cv2.CAP_PROP_CONTRAST, self.contrast)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)

            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame from camera.")
                self._image_data = None
                return

            self._image_data = frame
            # Emit the signal in a GLib-safe way
            idle_add(self.image_captured.send, self)
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            self._image_data = None

    def _capture_loop(self):
        """
        Internal method to continuously capture images from the camera.
        Runs in a separate thread.
        """
        while self._running:
            try:
                with VideoCaptureDevice(self.device_id) as cap:
                    logger.info("Camera %s opened successfully.",
                                self.device_id)
                    while self._running:
                        self._read_frame_and_update_data(cap)
                        GLib.usleep(33000)  # ~30 FPS
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                GLib.usleep(1000000)  # 1 second delay

        logger.debug("Camera capture loop stopped for camera %s.", self.name)

    def _start_capture_stream(self):
        """
        Starts a continuous image capture stream in a separate thread.
        """
        if self._running:
            logger.debug(
                "Capture stream already running for camera %s.", self.name
            )
            return

        logger.debug("Starting capture stream for camera %s.", self.name)
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop)
        self._capture_thread.daemon = True  # Allow the main program to exit
        self._capture_thread.start()

    def _stop_capture_stream(self):
        """
        Stops the continuous image capture stream.
        """
        if not self._running:
            logger.debug(
                "Capture stream not running for camera %s.", self.name
            )
            return

        logger.debug("Stopping capture stream for camera %s.", self.name)
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)  # Wait for thread to finish
            if self._capture_thread.is_alive():
                logger.warning("Capture thread did not terminate gracefully.")
        self._capture_thread = None

    def capture_image(self):
        """
        Captures a single image from this camera device.
        """
        try:
            with VideoCaptureDevice(self.device_id) as cap:
                self._read_frame_and_update_data(cap)
        except IOError as e:
            logger.error(f"Error capturing image: {e}")
            self._image_data = None
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            self._image_data = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "device_id": self.device_id,
            "enabled": self.enabled,
            "transform_matrix": self.transform_matrix,
            "white_balance": self.white_balance,
            "contrast": self.contrast,
            "brightness": self.brightness,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Camera':
        camera = cls(data["name"], data["device_id"])
        camera.enabled = data.get("enabled", camera.enabled)
        camera.transform_matrix = data.get("transform_matrix",
                                           camera.transform_matrix)
        camera.white_balance = data.get("white_balance",
                                        camera.white_balance)
        camera.contrast = data.get("contrast", camera.contrast)
        camera.brightness = data.get("brightness", camera.brightness)
        return camera

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.from_dict(data)

    @staticmethod
    def apply_affine_transform(image_data: np.ndarray,
                               transform_matrix: List[float],
                               output_size: Tuple[int, int]):
        """
        Applies a 2x3 affine transformation to an image.

        Args:
            image_data: The input image as a NumPy array.
            transform_matrix: A list of 6 floats representing the 2x3 affine
                              matrix. [M11, M12, M13, M21, M22, M23]
            output_size: A tuple (width, height) for the output image
                         dimensions.

        Returns:
            The transformed image as a NumPy array.
        """
        if image_data is None:
            return None

        if not isinstance(transform_matrix, list) or \
           len(transform_matrix) != 6:
            raise ValueError("Transform matrix must be a list of 6 floats.")

        M = np.array(transform_matrix).reshape((2, 3))

        transformed_image = cv2.warpAffine(image_data, M, output_size)
        return transformed_image

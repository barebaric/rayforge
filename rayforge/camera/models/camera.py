import json
import time
import threading
from typing import Optional, List, Tuple, Dict, Sequence, Any
import cv2
import numpy as np
import logging
from blinker import Signal
from gi.repository import GLib, GdkPixbuf  # type: ignore
from ...shared.util.glib import idle_add


logger = logging.getLogger(__name__)
Pos = Tuple[float, float]
PointList = Sequence[Pos]


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
        # None indicates auto white balance, float for manual Kelvin
        self._white_balance: Optional[float] = None
        self._active_subscribers: int = 0
        self._contrast: float = 50.0
        self._brightness: float = 0.0  # Default brightness (0 = no change)
        self._transparency: float = 0.2
        self._capture_thread: Optional[threading.Thread] = None
        self._running: bool = False

        # Properties for camera calibration
        # How to determine calibration values:
        # 1. Print a calibration pattern, e.g. a 8x6 grid with 25mm grid size
        # 2. Capture 10 or so calibration images of the grid (camera static,
        #    grid in different positions/rotations)
        # 3. Detect checkerboard corners: cv2.findChessboardCorners()
        # 4. Perform camera calibration: cv2.calibrateCamera
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None

        # Properties for camera calibration and alignment
        # Example usage to map pixel positions (image points) to
        # real world positions (in mm):
        #   image_points:
        #     List[Pos] = [(100, 100), (500, 100), (500, 400), (100, 400)]
        #   world_points:
        #     List[Pos] = [(-1, 120), (130, 120.5), (133, 0.1), (0, -0.1)]
        #   camera.image_to_world = image_points, world_points
        self._image_to_world: Optional[Tuple[PointList, PointList]] = None

        # Signals
        self.changed = Signal()
        self.image_captured = Signal()
        self.settings_changed = Signal()

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
        logger.debug(f"Found available camera devices: {available_devices}")
        return available_devices

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
            f"Camera device_id changed from '{self._device_id}' to "
            f"'{value}'"
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
        if self._enabled and self._active_subscribers > 0:
            # Only start if enabled AND someone is subscribed.
            self._start_capture_stream()
        else:
            # Stop if disabled OR no one is subscribed (handled in
            # unsubscribe).
            self._stop_capture_stream()
        self.changed.send(self)

    @property
    def image_data(self) -> Optional[np.ndarray]:
        return self._image_data

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
            f"Camera transparency changed from {self._transparency} to "
            f"{value}"
        )
        self._transparency = value
        self.changed.send(self)
        self.settings_changed.send(self)

    @property
    def image_to_world(self) -> Optional[Tuple[PointList, PointList]]:
        return self._image_to_world

    @image_to_world.setter
    def image_to_world(
        self, value: Optional[Tuple[PointList, PointList]]
    ):
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

    def subscribe(self):
        """
        Registers a subscriber to the camera's image stream.

        The stream will start if this is the first subscriber and the camera
        is enabled.
        """
        self._active_subscribers += 1
        logger.debug(
            f"Camera {self.name} subscribed. Count: "
            f"{self._active_subscribers}"
        )
        if self._active_subscribers == 1 and self.enabled:
            self._start_capture_stream()

    def unsubscribe(self):
        """
        Unregisters a subscriber.

        The stream will stop if this was the last active subscriber.
        """
        if self._active_subscribers > 0:
            self._active_subscribers -= 1
        logger.debug(
            f"Camera {self.name} unsubscribed. Count: "
            f"{self._active_subscribers}"
        )
        if self._active_subscribers == 0:
            self._stop_capture_stream()

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

    def _compute_homography(self, image_height: int) -> np.ndarray:
        """
        Compute the homography matrix from corresponding points.

        Args:
            image_height: The height of the image in pixels.

        Returns:
            3x3 homography matrix mapping world to image coordinates
        """
        if self._image_to_world is None:
            raise ValueError("Corresponding points are not set")

        image_points_raw, world_points = self._image_to_world

        # Invert y-coordinates of image_points to align with world coordinates
        # (y-up)
        image_points_y_up = [
            (p[0], image_height - p[1]) for p in image_points_raw
        ]

        if self._camera_matrix is not None and self._dist_coeffs is not None:
            # Undistort image points if calibration parameters are available
            image_points_y_up = cv2.undistortPoints(
                np.array(image_points_y_up, dtype=np.float32),
                self._camera_matrix,
                self._dist_coeffs,
            )
            image_points_y_up = image_points_y_up.reshape(-1, 2)

        # Compute homography (world to image_y_up)
        H, _ = cv2.findHomography(
            np.array(world_points, dtype=np.float32),
            np.array(image_points_y_up, dtype=np.float32),
        )
        return H

    def get_work_surface_image(
        self, output_size: Tuple[int, int], physical_area: Tuple[Pos, Pos]
    ) -> Optional[np.ndarray]:
        """
        Get an aligned image of the specified physical area.

        Args:
            output_size: Desired output image size (width, height) in pixels
            physical_area: Physical area ((x_min, y_min), (x_max, y_max))
              to capture in real-world coordinates

        Returns:
            Aligned image as a NumPy array
        """
        if self._image_to_world is None:
            raise ValueError(
                "Corresponding points must be set before getting the"
                " the work surface image"
            )
        if self._image_data is None:
            logger.warning("No image data available.")
            return None

        # Capture raw image
        raw_image = self._image_data

        # Undistort if calibration parameters are set
        if self._camera_matrix is not None and self._dist_coeffs is not None:
            try:
                undistorted_image = cv2.undistort(
                    raw_image, self._camera_matrix, self._dist_coeffs
                )
            except cv2.error as e:
                logger.warning(f"Failed to undistort image: {e}")
                undistorted_image = raw_image
        else:
            undistorted_image = raw_image

        # Compute homography (world to image)
        try:
            H = self._compute_homography(raw_image.shape[0])
        except ValueError as e:
            logger.error(f"Cannot compute homography: {e}")
            return None

        # Define transformation from output pixels to world coordinates
        (x_min, y_min), (x_max, y_max) = physical_area
        width_px, height_px = output_size

        # Calculate the actual physical width and height of the area being
        # viewed
        physical_width = x_max - x_min
        physical_height = y_max - y_min

        # Calculate the scaling factors from output pixels to world coordinates
        scale_x = physical_width / width_px
        scale_y = -physical_height / height_px

        offset_x = x_min
        offset_y = y_max
        T = np.array(
            [
                [scale_x, 0, offset_x],
                [0, scale_y, offset_y],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Overall transformation: output pixels -> world -> image
        M = H @ T

        # Apply perspective warp
        # Log transformed corner points
        world_corners = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            dtype=np.float32,
        )
        world_corners_h = np.hstack((world_corners, np.ones((4, 1)))).T
        transformed_corners = M @ world_corners_h
        transformed_corners = transformed_corners[:2] / transformed_corners[2]
        transformed_corners = transformed_corners.T

        try:
            aligned_image = cv2.warpPerspective(
                undistorted_image, np.linalg.inv(M), output_size
            )
            return aligned_image
        except cv2.error as e:
            logger.error(f"Failed to apply perspective warp: {e}")
            return None

    def _read_frame_and_update_data(self, cap: cv2.VideoCapture):
        """
        Reads a single frame from the given VideoCapture object,
        updates camera data, and emits the image_captured signal.
        """
        try:
            # Apply white balance, contrast, and brightness settings
            if self.white_balance is None:
                cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Enable auto white balance
            else:
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
                # Open the device ONCE
                with VideoCaptureDevice(self.device_id) as cap:
                    logger.info(
                        f"Camera {self.device_id} opened successfully."
                    )
                    # Loop to read frames from the open device
                    while self._running:
                        self._read_frame_and_update_data(cap)
                        # Use time.sleep for portability in a non-GUI thread
                        time.sleep(1 / 30)  # ~30 FPS
            except Exception as e:
                logger.error(
                    f"Error in capture loop for camera '{self.name}': {e}. "
                    "Retrying in 1 second."
                )
                if self._running:
                    time.sleep(1)  # 1 second delay before retrying

        logger.debug(f"Camera capture loop stopped for camera {self.name}.")

    def _start_capture_stream(self):
        """
        Starts a continuous image capture stream in a separate thread.
        """
        if self._running:
            logger.debug(
                f"Capture stream already running for camera {self.name}."
            )
            return

        logger.debug(f"Starting capture stream for camera {self.name}.")
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
                f"Capture stream not running for camera {self.name}."
            )
            return

        logger.debug(f"Stopping capture stream for camera {self.name}.")
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
            "white_balance": self.white_balance,
            "contrast": self.contrast,
            "brightness": self.brightness,
            "transparency": self.transparency,
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
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Camera":
        camera = cls(data["name"], data["device_id"])
        camera.enabled = data.get("enabled", camera.enabled)
        camera.white_balance = data.get("white_balance", None)
        camera.contrast = data.get("contrast", camera.contrast)
        camera.brightness = data.get("brightness", camera.brightness)
        camera.transparency = data.get("transparency", camera.transparency)

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
        return camera

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.from_dict(data)

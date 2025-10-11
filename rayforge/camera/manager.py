import logging
from typing import Dict, List
from blinker import Signal
from .models.camera import Camera
from .controller import CameraController

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages the lifecycle of CameraController instances.

    This class acts as the single source of truth for live camera controllers.
    It listens for changes in the application's configuration (specifically,
    the active machine and its list of cameras) and reconciles its internal
    list of controllers to match. It creates, destroys, and provides access
    to CameraController instances, emitting signals as the list of active
    controllers changes.
    """

    def __init__(self):
        self._controllers: Dict[str, CameraController] = {}

        # Signals
        self.controller_added = Signal()
        self.controller_removed = Signal()

    def initialize(self):
        """
        Performs initial setup after all managers are available.
        This is where signal connections are made to avoid circular imports.
        """
        from ..config import config

        config.changed.connect(self._on_config_changed)
        self._reconcile_controllers()

    def shutdown(self):
        """Shuts down all active camera controllers."""
        logger.info("Shutting down all camera controllers.")
        from ..config import config

        config.changed.disconnect(self._on_config_changed)
        for controller in list(self._controllers.values()):
            self._destroy_controller(controller.config.device_id)
        logger.info("All camera controllers shut down.")

    @property
    def controllers(self) -> List[CameraController]:
        """Returns a list of all active CameraController instances."""
        return list(self._controllers.values())

    def get_controller(self, device_id: str) -> CameraController | None:
        """Gets a specific controller by its device ID."""
        return self._controllers.get(device_id)

    def _on_config_changed(self, sender, **kwargs):
        """Handler for when the global config or active machine changes."""
        logger.debug("Configuration changed, reconciling camera controllers.")
        self._reconcile_controllers()

    def _destroy_controller(self, device_id: str):
        """Safely unsubscribes, stops, and removes a controller."""
        if device_id in self._controllers:
            controller = self._controllers.pop(device_id)
            controller.unsubscribe()  # Stops the thread if it's the last sub
            self.controller_removed.send(self, controller=controller)
            logger.info(f"Destroyed controller for camera {device_id}")

    def _reconcile_controllers(self):
        """
        Synchronizes the set of active CameraControllers with the cameras
        defined in the currently active machine model.
        """
        from ..config import config

        active_machine = config.machine
        camera_configs_in_model: Dict[str, Camera] = {}
        if active_machine:
            camera_configs_in_model = {
                c.device_id: c for c in active_machine.cameras
            }

        model_ids = set(camera_configs_in_model.keys())
        active_controller_ids = set(self._controllers.keys())

        # Destroy controllers for cameras that were removed from the model
        for device_id in active_controller_ids - model_ids:
            self._destroy_controller(device_id)

        # Create controllers for new cameras added to the model
        for device_id in model_ids - active_controller_ids:
            config_model = camera_configs_in_model[device_id]
            controller = CameraController(config_model)
            self._controllers[device_id] = controller
            self.controller_added.send(self, controller=controller)
            logger.info(f"Created controller for camera {device_id}")

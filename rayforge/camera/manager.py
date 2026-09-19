import logging

from blinker import Signal

from ..context import RayforgeContext
from ..machine.models.machine import Machine
from .controller import CameraController
from .models.camera import Camera

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

    def __init__(self, context: RayforgeContext):
        self._context = context
        self._controllers: dict[str, CameraController] = {}
        self._active_machine: Machine | None = None

        # Signals
        self.controller_added = Signal()
        self.controller_removed = Signal()

    def initialize(self):
        """
        Performs initial setup after all managers are available.
        This is where signal connections are made to avoid circular imports.
        """
        config = self._context.config
        if not config:
            logger.error(
                "Cannot initialize CameraManager: Config not found in context."
            )
            return

        config.changed.connect(self._on_config_changed)
        # Manually trigger the first check to set up the initial machine
        self._on_config_changed(config)

    def shutdown(self):
        """Shuts down all active camera controllers."""
        logger.info("Shutting down all camera controllers.")
        config = self._context.config
        if config:
            config.changed.disconnect(self._on_config_changed)

        # Disconnect from the last active machine
        if self._active_machine:
            self._active_machine.changed.disconnect(
                self._on_active_machine_changed
            )
            self._active_machine = None

        for controller in list(self._controllers.values()):
            self._destroy_controller(controller.config.device_id)
        logger.info("All camera controllers shut down.")

    @property
    def controllers(self) -> list[CameraController]:
        """Returns a list of all active CameraController instances."""
        return list(self._controllers.values())

    def get_controller(self, device_id: str) -> CameraController | None:
        """Gets a specific controller by its device ID."""
        return self._controllers.get(device_id)

    def _on_config_changed(self, sender, **kwargs):
        """
        Handler for when the global config changes. This method is now
        responsible for tracking the active machine and connecting/
        disconnecting from its `changed` signal.
        """
        config = self._context.config
        if not config:
            return

        if self._active_machine is not config.machine:
            logger.debug(
                "Active machine changed, updating signal connections."
            )
            # Disconnect from the old machine if it exists
            if self._active_machine:
                self._active_machine.changed.disconnect(
                    self._on_active_machine_changed
                )

            # Connect to the new machine
            self._active_machine = config.machine
            if self._active_machine:
                self._active_machine.changed.connect(
                    self._on_active_machine_changed
                )

        # Always reconcile when the config changes, as this is the top-level
        # trigger for a machine switch.
        self._reconcile_controllers()

    def _on_active_machine_changed(self, sender, **kwargs):
        """
        Handler for when the currently active machine's properties change
        (e.g., a camera is added or removed).
        """
        logger.debug(
            "Active machine's properties changed, reconciling cameras."
        )
        self._reconcile_controllers()

    def _destroy_controller(self, device_id: str):
        """Safely unsubscribes, stops, removes and de-signals a controller."""
        if device_id in self._controllers:
            controller = self._controllers.pop(device_id)
            controller.unsubscribe()  # Stops the thread if it's the last sub
            self.controller_removed.send(self, controller=controller)
            # After the UI subscribers dropped their refs: hard-stop the
            # stream and disconnect from the model so the destroyed
            # controller can never be resurrected by later config changes.
            controller.dispose()
            logger.info(f"Destroyed controller for camera {device_id}")

    def _reconcile_controllers(self):
        """
        Synchronizes the set of active CameraControllers with the cameras
        defined in the currently active machine model.

        A Camera whose device_id changed keeps its controller: it is
        re-keyed and swapped in place via set_device_id(), so subscribers
        (displays, alignment widgets, canvas surfaces) do not churn through
        controller_removed/controller_added for what is merely a device swap.
        """
        config = self._context.config
        if not config:
            return

        active_machine = config.machine
        camera_configs_in_model: dict[str, Camera] = {}
        if active_machine:
            camera_configs_in_model = {
                c.device_id: c for c in active_machine.cameras
            }

        model_ids = set(camera_configs_in_model.keys())
        active_controller_ids = set(self._controllers.keys())

        # Detect device swaps: the same Camera instance is now filed under
        # a different device_id key. Reuse its controller instead of
        # destroying and rebuilding it.
        swapped: dict[str, str] = {}  # old_key -> new_key
        for old_key in active_controller_ids - model_ids:
            controller = self._controllers[old_key]
            new_key = controller.config.device_id
            if (
                camera_configs_in_model.get(new_key) is controller.config
                and new_key not in self._controllers
            ):
                swapped[old_key] = new_key

        for old_key, new_key in swapped.items():
            controller = self._controllers.pop(old_key)
            self._controllers[new_key] = controller
            # Usually a no-op: the controller already reacted to the model
            # change via config.changed and swapped itself.
            controller.set_device_id(new_key)
            logger.info(
                f"Reused controller for camera "
                f"'{controller.config.name}' after device swap "
                f"{old_key} -> {new_key}"
            )

        # Destroy controllers for cameras that were truly removed from the
        # model
        for device_id in (
            active_controller_ids - model_ids - set(swapped.keys())
        ):
            self._destroy_controller(device_id)

        # Create controllers for new cameras added to the model
        for device_id in model_ids - set(self._controllers.keys()):
            config_model = camera_configs_in_model[device_id]
            controller = CameraController(config_model)
            self._controllers[device_id] = controller
            self.controller_added.send(self, controller=controller)
            logger.info(f"Created controller for camera {device_id}")

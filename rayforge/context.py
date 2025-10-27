import logging
import threading
from typing import Optional, TYPE_CHECKING

# Use a TYPE_CHECKING block to import types for static analysis
# without causing a runtime circular import.
if TYPE_CHECKING:
    from .camera.manager import CameraManager
    from .core.config import Config, ConfigManager
    from .core.library_manager import LibraryManager
    from .machine.models.machine import Machine, MachineManager
    from .debug import DebugLogManager


logger = logging.getLogger(__name__)

_context_instance: Optional["RayforgeContext"] = None
_context_lock = threading.Lock()


class RayforgeContext:
    """
    A central, singleton context for managing the lifecycle of major
    application services.
    """

    def __init__(self):
        """
        Initializes the context. This constructor is lightweight and safe
        to call from any process. It only sets up services that are safe
        for subprocess initialization.
        """
        from .pipeline.artifact.store import ArtifactStore
        from .debug import DebugLogManager

        self.artifact_store = ArtifactStore()
        # These managers are initialized to None. The main application thread
        # MUST call initialize_full_context() to create them.
        self._machine_mgr: Optional["MachineManager"] = None
        self._config_mgr: Optional["ConfigManager"] = None
        self._config: Optional["Config"] = None
        self._camera_mgr: Optional["CameraManager"] = None
        self._material_mgr: Optional["LibraryManager"] = None
        self._debug_log_manager = DebugLogManager()

    @property
    def machine(self) -> Optional["Machine"]:
        """
        Returns the active machine from the config, or None if
        the config or machine is not set.
        """
        return self._config.machine if self._config else None

    @property
    def machine_mgr(self) -> "MachineManager":
        """Returns the machine manager. Raises an error if not initialized."""
        if self._machine_mgr is None:
            raise RuntimeError("Machine manager is not initialized.")
        return self._machine_mgr

    @property
    def config_mgr(self) -> "ConfigManager":
        """Returns the config manager. Raises an error if not initialized."""
        if self._config_mgr is None:
            raise RuntimeError("Config manager is not initialized.")
        return self._config_mgr

    @property
    def config(self) -> "Config":
        """Returns the config. Raises an error if not initialized."""
        if self._config is None:
            raise RuntimeError("Config is not initialized.")
        return self._config

    @property
    def camera_mgr(self) -> "CameraManager":
        """Returns the camera manager. Raises an error if not initialized."""
        if self._camera_mgr is None:
            raise RuntimeError("Camera manager is not initialized.")
        return self._camera_mgr

    @property
    def material_mgr(self) -> "LibraryManager":
        """Returns the material manager. Raises an error if not initialized."""
        if self._material_mgr is None:
            raise RuntimeError("Material manager is not initialized.")
        return self._material_mgr

    @property
    def debug_log_manager(self) -> "DebugLogManager":
        """Returns the debug log manager."""
        return self._debug_log_manager

    def initialize_full_context(self):
        """
        Initializes the full application context with managers that should
        only be created in the main process. This is NOT safe to call from
        a subprocess.
        """
        # This function should be idempotent.
        if self._config_mgr is not None:
            return

        # Import high-level managers here, inside the method, to avoid
        # circular dependencies at the module level.
        from .camera.manager import CameraManager
        from .core.library_manager import LibraryManager
        from .machine.models.machine import MachineManager
        from .config import (
            CONFIG_DIR,
            MACHINE_DIR,
            CONFIG_FILE,
            CORE_MATERIALS_DIR,
            USER_MATERIALS_DIR,
        )
        from .core.config import ConfigManager as CoreConfigManager

        logger.info(f"Initializing full application context from {CONFIG_DIR}")

        # Load all machines. If none exist, create a default machine.
        self._machine_mgr = MachineManager(MACHINE_DIR)
        logger.info(f"Loaded {len(self._machine_mgr.machines)} machines")
        if not self._machine_mgr.machines:
            machine = self._machine_mgr.create_default_machine()
            logger.info(f"Created default machine {machine.id}")

        # Load the config file. This must happen before CameraManager init.
        self._config_mgr = CoreConfigManager(CONFIG_FILE, self._machine_mgr)
        self._config = self._config_mgr.config
        if not self._config.machine:
            # Sort by ID for deterministic selection
            machine = list(
                sorted(self._machine_mgr.machines.values(), key=lambda m: m.id)
            )[0]
            self._config.set_machine(machine)
            assert self._config.machine
        logger.info(f"Config loaded. Using machine {self._config.machine.id}")

        # Initialize the camera manager AFTER config is loaded and active
        # machine is set
        self._camera_mgr = CameraManager()
        self._camera_mgr.initialize()
        logger.info(
            f"Camera manager initialized with "
            f"{len(self._camera_mgr.controllers)} controllers."
        )

        # Initialize the material manager
        self._material_mgr = LibraryManager(
            CORE_MATERIALS_DIR, USER_MATERIALS_DIR
        )
        self._material_mgr.load_all_libraries()
        logger.info(
            f"Material manager initialized with "
            f"{len(self._material_mgr)} materials"
        )

        logger.info("Full application context initialized")

    async def shutdown(self):
        """
        Shuts down all managed services in the correct order.
        """
        logger.info("RayforgeContext shutting down...")
        if self._camera_mgr:
            self._camera_mgr.shutdown()
        if self._machine_mgr:
            await self._machine_mgr.shutdown()
        self.artifact_store.shutdown()
        logger.info("RayforgeContext shutdown complete.")


def get_context() -> "RayforgeContext":
    """
    A thread-safe, lazy-initializing accessor for the global RayforgeContext
    singleton.
    """
    global _context_instance
    if _context_instance is None:
        with _context_lock:
            # Double-check lock to prevent race conditions
            if _context_instance is None:
                _context_instance = RayforgeContext()
    return _context_instance

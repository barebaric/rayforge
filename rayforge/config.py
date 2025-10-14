import os
from pathlib import Path
from typing import Optional
from platformdirs import user_config_dir
from .camera.manager import CameraManager
from .core.config import ConfigManager
from .machine.models.machine import MachineManager
from .core.library_manager import LibraryManager
import logging


logger = logging.getLogger(__name__)


CONFIG_DIR = Path(user_config_dir("rayforge"))
MACHINE_DIR = CONFIG_DIR / "machines"
MACHINE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Material directories
CORE_MATERIALS_DIR = Path(__file__).parent / "resources" / "core_materials"
USER_MATERIALS_DIR = CONFIG_DIR / "materials"

logger.info(f"Config dir is {CONFIG_DIR}")


def getflag(name, default=False):
    default = "true" if default else "false"
    return os.environ.get(name, default).lower() in ("true", "1")


# Load all machines. If none exist, create a default machine.
# These are initialized to None to prevent automatic setup in subprocesses.
# The main application must call initialize_managers() to populate them.
machine_mgr = None
config_mgr: Optional[ConfigManager] = None
config = None  # Will be an alias for config_mgr.config after init
camera_mgr: CameraManager
material_mgr = None  # Will be initialized in initialize_managers()


def initialize_managers():
    """
    Initializes the machine, config, and material managers. This function
    is designed to be called once from the main application process.
    It is safe to call multiple times (idempotent).

    This prevents expensive I/O and state setup from running automatically
    when a module is imported into a subprocess.
    """
    global machine_mgr, config_mgr, config, camera_mgr, material_mgr

    # Idempotency check: If already initialized, do nothing.
    if config_mgr is not None:
        return

    logger.info(f"Initializing configuration from {CONFIG_DIR}")
    MACHINE_DIR.mkdir(parents=True, exist_ok=True)

    # Load all machines. If none exist, create a default machine.
    machine_mgr = MachineManager(MACHINE_DIR)
    logger.info(f"Loaded {len(machine_mgr.machines)} machines")
    if not machine_mgr.machines:
        machine = machine_mgr.create_default_machine()
        logger.info(f"Created default machine {machine.id}")

    # Load the config file. This must happen before CameraManager init.
    config_mgr = ConfigManager(CONFIG_FILE, machine_mgr)
    config = config_mgr.config  # Set the global config alias
    if not config.machine:
        # Sort by ID for deterministic selection
        machine = list(
            sorted(machine_mgr.machines.values(), key=lambda m: m.id)
        )[0]
        config.set_machine(machine)
        assert config.machine
    logger.info(f"Config loaded. Using machine {config.machine.id}")

    # Initialize the camera manager AFTER config is loaded and active machine
    # is set
    camera_mgr = CameraManager()
    camera_mgr.initialize()
    logger.info(
        f"Camera manager initialized with {len(camera_mgr.controllers)} "
        "controllers."
    )

    # Initialize the material manager
    material_mgr = LibraryManager(CORE_MATERIALS_DIR, USER_MATERIALS_DIR)
    material_mgr.load_all_libraries()
    logger.info(
        f"Material manager initialized with {len(material_mgr)} materials"
    )

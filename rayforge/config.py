import os
from pathlib import Path
from typing import Optional
from platformdirs import user_config_dir
from .machine.models.machine import MachineManager
from .core.config import ConfigManager
import logging


logger = logging.getLogger(__name__)


CONFIG_DIR = Path(user_config_dir("rayforge"))
MACHINE_DIR = CONFIG_DIR / "machines"
MACHINE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "config.yaml"
logger.info(f"Config dir is {CONFIG_DIR}")


def getflag(name, default=False):
    default = 'true' if default else 'false'
    return os.environ.get(name, default).lower() in ('true', '1')


# Load all machines. If none exist, create a default machine.
# These are initialized to None to prevent automatic setup in subprocesses.
# The main application must call initialize_managers() to populate them.
machine_mgr = None
config_mgr: Optional[ConfigManager] = None
config = None  # Will be an alias for config_mgr.config after init


def initialize_managers():
    """
    Initializes the machine and config managers. This function is designed
    to be called once from the main application process. It is safe to
    call multiple times (idempotent).

    This prevents expensive I/O and state setup from running automatically
    when a module is imported into a subprocess.
    """
    global machine_mgr, config_mgr, config

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

    # Load the config file.
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

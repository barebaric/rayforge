import os
from pathlib import Path
from platformdirs import user_config_dir
from .models.machine import MachineManager
from .models.config import ConfigManager
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
machine_mgr = MachineManager(MACHINE_DIR)
logger.info(f"Loaded {len(machine_mgr.machines)} machines")
if not machine_mgr.machines:
    machine = machine_mgr.create_default_machine()
    logger.info(f"Created default machine {machine.id}")

# Load the config file.
config_mgr = ConfigManager(CONFIG_FILE, machine_mgr)
config = config_mgr.config
if not config.machine:
    machine = list(sorted(machine_mgr.machines.values()))[0]
    config.set_machine(machine)
logger.info(f"Config loaded. Using machine {config.machine.id}")

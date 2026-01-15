import os
from pathlib import Path
from platformdirs import user_config_dir, user_log_dir
import logging


logger = logging.getLogger(__name__)


# Configuration files
CONFIG_DIR = Path(user_config_dir("rayforge"))
logger.info(f"Config dir is {CONFIG_DIR}")

MACHINE_DIR = CONFIG_DIR / "machines"
logger.debug(f"MACHINE_DIR is {MACHINE_DIR}")
MACHINE_DIR.mkdir(parents=True, exist_ok=True)

DIALECT_DIR = CONFIG_DIR / "dialects"
logger.debug(f"DIALECT_DIR is {DIALECT_DIR}")
DIALECT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = CONFIG_DIR / "config.yaml"
PACKAGES_DIR = CONFIG_DIR / "packages"

# State files (like logs)
LOG_DIR = Path(user_log_dir("rayforge"))
logger.info(f"Log dir is {LOG_DIR}")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Material directories
CORE_MATERIALS_DIR = Path(__file__).parent / "resources" / "core_materials"
USER_MATERIALS_DIR = CONFIG_DIR / "materials"

# Material directories
USER_RECIPES_DIR = CONFIG_DIR / "recipes"

# Package registry
PACKAGE_REGISTRY_URL = (
    "https://raw.githubusercontent.com/barebaric/rayforge-registry/"
    "main/registry.yaml"
)


def getflag(name, default=False):
    default = "true" if default else "false"
    return os.environ.get(name, default).lower() in ("true", "1")

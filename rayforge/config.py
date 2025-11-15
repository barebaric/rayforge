import os
from pathlib import Path
from platformdirs import user_config_dir, user_log_dir
import logging


logger = logging.getLogger(__name__)


# Configuration files
CONFIG_DIR = Path(user_config_dir("rayforge"))
MACHINE_DIR = CONFIG_DIR / "machines"
MACHINE_DIR.mkdir(parents=True, exist_ok=True)
DIALECT_DIR = CONFIG_DIR / "dialects"
DIALECT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# State files (like logs)
LOG_DIR = Path(user_log_dir("rayforge"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Material directories
CORE_MATERIALS_DIR = Path(__file__).parent / "resources" / "core_materials"
USER_MATERIALS_DIR = CONFIG_DIR / "materials"

# Material directories
USER_RECIPES_DIR = CONFIG_DIR / "recipes"

logger.info(f"Config dir is {CONFIG_DIR}")
logger.info(f"Log dir is {LOG_DIR}")


def getflag(name, default=False):
    default = "true" if default else "false"
    return os.environ.get(name, default).lower() in ("true", "1")

import os
from pathlib import Path
from platformdirs import user_config_dir, user_log_dir
import logging


logger = logging.getLogger(__name__)


def _get_config_dir() -> Path:
    """Get the config directory, respecting RAYFORGE_CONFIG_DIR env var."""
    env_config = os.environ.get("RAYFORGE_CONFIG_DIR")
    if env_config:
        return Path(env_config)
    return Path(user_config_dir("rayforge"))


CONFIG_DIR = _get_config_dir()
logger.info(f"Config dir is {CONFIG_DIR}")

MACHINE_DIR = CONFIG_DIR / "machines"
logger.debug(f"MACHINE_DIR is {MACHINE_DIR}")
MACHINE_DIR.mkdir(parents=True, exist_ok=True)

DIALECT_DIR = CONFIG_DIR / "dialects"
logger.debug(f"DIALECT_DIR is {DIALECT_DIR}")
DIALECT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = CONFIG_DIR / "config.yaml"
ADDONS_DIR = CONFIG_DIR / "addons"

BUILTIN_ADDONS_DIR = Path(__file__).parent / "builtin_addons"

# State files (like logs)
LOG_DIR = Path(user_log_dir("rayforge"))
logger.info(f"Log dir is {LOG_DIR}")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Material directories
CORE_MATERIALS_DIR = Path(__file__).parent / "resources" / "core_materials"
USER_MATERIALS_DIR = CONFIG_DIR / "materials"
USER_RECIPES_DIR = CONFIG_DIR / "recipes"

ADDON_REGISTRY_URL = (
    "https://raw.githubusercontent.com/barebaric/rayforge-registry/"
    "main/registry.yaml"
)

UMAMI_URL = "https://analytics.barebaric.com/api/send"
UMAMI_WEBSITE_ID = "3b301b16-48d2-4007-977a-ccfb738eab52"


def getflag(name, default=False):
    default = "true" if default else "false"
    return os.environ.get(name, default).lower() in ("true", "1")

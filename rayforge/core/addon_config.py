import logging
from pathlib import Path
from typing import Dict

import yaml

logger = logging.getLogger(__name__)


class AddonState:
    """Represents the enabled/disabled state of an addon."""

    ENABLED = "enabled"
    DISABLED = "disabled"


class AddonConfig:
    """Manages persistent addon enable/disable state."""

    def __init__(self, config_dir: Path):
        self.config_file = config_dir / "addons.yaml"
        self._states: Dict[str, str] = {}

    def load(self):
        """Load addon states from the config file."""
        if not self.config_file.exists():
            logger.debug(
                f"Addon config file not found at {self.config_file}, "
                "using defaults"
            )
            return

        try:
            with open(self.config_file, "r") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                self._states = data
            logger.debug(f"Loaded addon states from {self.config_file}")
        except (yaml.YAMLError, IOError) as e:
            logger.warning(f"Failed to load addon config: {e}")
            self._states = {}

    def save(self):
        """Save addon states to the config file."""
        try:
            if self._states:
                self.config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_file, "w") as f:
                    yaml.dump(self._states, f, default_flow_style=False)
            elif self.config_file.exists():
                self.config_file.unlink()
            logger.debug(f"Saved addon states to {self.config_file}")
        except IOError as e:
            logger.error(f"Failed to save addon config: {e}")

    def get_state(self, addon_name: str) -> str:
        """Get the state of an addon. Defaults to ENABLED."""
        state = self._states.get(addon_name, AddonState.ENABLED)
        if state not in (AddonState.ENABLED, AddonState.DISABLED):
            logger.warning(
                f"Invalid state '{state}' for addon '{addon_name}', "
                "defaulting to ENABLED"
            )
            return AddonState.ENABLED
        return state

    def set_state(self, addon_name: str, state: str):
        """Set the state of an addon."""
        if state not in (AddonState.ENABLED, AddonState.DISABLED):
            raise ValueError(f"Invalid addon state: {state}")
        self._states[addon_name] = state
        self.save()
        logger.info(f"Set addon '{addon_name}' state to '{state}'")

    def remove_state(self, addon_name: str):
        """Remove an addon's state from the config."""
        if addon_name in self._states:
            del self._states[addon_name]
            self.save()
            logger.info(f"Removed addon '{addon_name}' from config")

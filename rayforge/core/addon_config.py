import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import yaml

logger = logging.getLogger(__name__)


class AddonState:
    """Represents the enabled/disabled state of an addon."""

    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclass
class AddonConfigEntry:
    """
    Configuration entry for a single addon.

    Attributes:
        state: The enabled/disabled state of the addon.
        version: The installed version string, or None for builtin/legacy.
    """

    state: str
    version: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to a dictionary for YAML serialization."""
        return {"state": self.state, "version": self.version}

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Optional[str]]
    ) -> "AddonConfigEntry":
        """
        Create an AddonConfigEntry from a dictionary.

        Handles legacy format where state was stored directly as a string.
        """
        if isinstance(data, dict):
            state_value = data.get("state")
            state: str = state_value if state_value else AddonState.ENABLED
            version = data.get("version")
            return cls(state=state, version=version)
        return cls(state=AddonState.ENABLED, version=None)


class AddonConfig:
    """Manages persistent addon configuration including state and version."""

    def __init__(self, config_dir: Path):
        self.config_file = config_dir / "addons.yaml"
        self._entries: Dict[str, AddonConfigEntry] = {}

    def load(self):
        """Load addon configuration from the config file."""
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
                for addon_name, entry_data in data.items():
                    if isinstance(entry_data, dict):
                        self._entries[addon_name] = AddonConfigEntry.from_dict(
                            entry_data
                        )
                    elif isinstance(entry_data, str):
                        self._entries[addon_name] = AddonConfigEntry(
                            state=entry_data, version=None
                        )
            logger.debug(f"Loaded addon config from {self.config_file}")
        except (yaml.YAMLError, IOError) as e:
            logger.warning(f"Failed to load addon config: {e}")
            self._entries = {}

    def save(self):
        """Save addon configuration to the config file."""
        try:
            if self._entries:
                self.config_file.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    name: entry.to_dict()
                    for name, entry in self._entries.items()
                }
                with open(self.config_file, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)
            elif self.config_file.exists():
                self.config_file.unlink()
            logger.debug(f"Saved addon config to {self.config_file}")
        except IOError as e:
            logger.error(f"Failed to save addon config: {e}")

    def get_state(self, addon_name: str) -> str:
        """Get the state of an addon. Defaults to ENABLED."""
        entry = self._entries.get(addon_name)
        if entry is None:
            return AddonState.ENABLED
        if entry.state not in (AddonState.ENABLED, AddonState.DISABLED):
            logger.warning(
                f"Invalid state '{entry.state}' for addon '{addon_name}', "
                "defaulting to ENABLED"
            )
            return AddonState.ENABLED
        return entry.state

    def set_state(self, addon_name: str, state: str):
        """Set the state of an addon."""
        if state not in (AddonState.ENABLED, AddonState.DISABLED):
            raise ValueError(f"Invalid addon state: {state}")
        entry = self._entries.get(addon_name)
        if entry:
            entry.state = state
        else:
            self._entries[addon_name] = AddonConfigEntry(state=state)
        self.save()
        logger.info(f"Set addon '{addon_name}' state to '{state}'")

    def get_version(self, addon_name: str) -> Optional[str]:
        """Get the stored version string of an addon."""
        entry = self._entries.get(addon_name)
        if entry is None:
            return None
        return entry.version

    def set_version(self, addon_name: str, version: str):
        """Set the version of an addon."""
        entry = self._entries.get(addon_name)
        if entry:
            entry.version = version
        else:
            self._entries[addon_name] = AddonConfigEntry(
                state=AddonState.ENABLED, version=version
            )
        self.save()
        logger.debug(f"Set addon '{addon_name}' version to '{version}'")

    def set_entry(
        self, addon_name: str, state: str, version: Optional[str] = None
    ):
        """Set both state and version for an addon in one operation."""
        if state not in (AddonState.ENABLED, AddonState.DISABLED):
            raise ValueError(f"Invalid addon state: {state}")
        self._entries[addon_name] = AddonConfigEntry(
            state=state, version=version
        )
        self.save()
        logger.info(
            f"Set addon '{addon_name}' state='{state}', version='{version}'"
        )

    def remove_state(self, addon_name: str):
        """Remove an addon's configuration entry."""
        if addon_name in self._entries:
            del self._entries[addon_name]
            self.save()
            logger.info(f"Removed addon '{addon_name}' from config")

    def get_entry(self, addon_name: str) -> Optional[AddonConfigEntry]:
        """Get the full configuration entry for an addon."""
        return self._entries.get(addon_name)

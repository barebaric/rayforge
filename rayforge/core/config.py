import yaml
import logging
from typing import Dict, Any, Optional
from blinker import Signal
from pathlib import Path
from ..machine.models.machine import Machine


logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        self.machine: Optional[Machine] = None
        self.theme: str = "system"
        self.changed = Signal()

    def set_machine(self, machine: Machine):
        if self.machine == machine:
            return
        if self.machine:
            self.machine.changed.disconnect(self.changed.send)
        self.machine = machine
        self.changed.send(self)
        self.machine.changed.connect(self.changed.send)

    def set_theme(self, theme: str):
        """Sets the application theme preference."""
        if self.theme == theme:
            return
        self.theme = theme
        self.changed.send(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine": self.machine.id if self.machine else None,
            "theme": self.theme,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], get_machine_by_id) -> "Config":
        config = cls()
        config.theme = data.get("theme", "system")

        # Get the machine by ID. add fallbacks in case the machines
        # no longer exist.
        machine_id = data.get("machine")
        machine = None
        if machine_id is not None:
            machine = get_machine_by_id(machine_id)
            if machine is None:
                msg = f"config references unknown machine {machine_id}"
                logger.error(msg)
        if machine:
            config.set_machine(machine)

        return config


class ConfigManager:
    def __init__(self, filepath: Path, machine_mgr):
        self.filepath = filepath
        self.machine_mgr = machine_mgr
        self.config: Config = Config()

        self.load_config()
        self.config.changed.connect(self._on_config_changed)

    def _on_config_changed(self, sender, **kwargs):
        self.save()

    def save(self):
        if not self.config:
            return
        with open(self.filepath, "w") as f:
            yaml.safe_dump(self.config.to_dict(), f)

    def load_config(self) -> "Config":
        if not self.filepath.exists():
            self.config = Config()
            return self.config

        try:
            with open(self.filepath, "r") as f:
                data = yaml.safe_load(f)
                if not data:
                    self.config = Config()
                else:
                    self.config = Config.from_dict(
                        data, self.machine_mgr.get_machine_by_id
                    )
        except (IOError, yaml.YAMLError) as e:
            logger.error(
                f"Failed to load config file: {e}. Creating a default config."
            )
            self.config = Config()

        return self.config

import yaml
import logging
from typing import Dict, Any
from blinker import Signal
from .machine import Machine
from pathlib import Path


logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        self.machine: Machine = Machine()
        self.paned_position = 60  # in percent
        self.changed = Signal()

    def set_machine(self, machine: Machine):
        if self.machine == machine:
            return
        self.machine.changed.disconnect(self.changed.send)
        self.machine = machine
        self.changed.send(self)
        self.machine.changed.connect(self.changed.send)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine": self.machine.id if self.machine else None,
            "paned_position": self.paned_position
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], get_machine_by_id) -> 'Config':
        config = cls()

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

        config.paned_position = data.get("paned_position",
                                         config.paned_position)
        return config


class ConfigManager:
    def __init__(self, filepath: Path, machine_mgr):
        self.filepath = filepath
        self.machine_mgr = machine_mgr
        self.config: Config = Config()

        self.load_config()

    def save(self):
        if not self.config:
            return
        with open(self.filepath, 'w') as f:
            yaml.safe_dump(self.config.to_dict(), f)

    def load_config(self) -> 'Config':
        if not self.filepath.exists():
            self.config = Config()   # Return a default config
            return self.config

        with open(self.filepath, 'r') as f:
            data = yaml.safe_load(f)
            if not data:
                return Config()
            config = Config.from_dict(data, self.machine_mgr.get_machine_by_id)
            self.config = config
            return config

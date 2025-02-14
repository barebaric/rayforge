from pathlib import Path
from platformdirs import user_config_dir
from .models.machine import MachineManager, Machine
from .models.config import ConfigManager, Config
from .util.resources import get_machine_template_path

CONFIG_DIR = Path(user_config_dir("rayforge"))
MACHINE_DIR = CONFIG_DIR / "machines"
MACHINE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "config.yaml"
print(f"Config dir is {CONFIG_DIR}")

"""
TODO # Load machine templates.
machine_template_path = get_machine_template_path()
machine_template_mgr = MachineManager(machine_template_path)
print(f"Loaded {len(machine_template_mgr.machines)} machine templates")
"""

# Load all machines. If none exist, create a default machine.
machine_mgr = MachineManager(MACHINE_DIR)
print(f"Loaded {len(machine_mgr.machines)} machines")
if not machine_mgr.machines:
    machine = machine_mgr.create_default_machine()
    print(f"Created default machine {machine.id}")

# Load the config file.
config_mgr = ConfigManager(CONFIG_FILE, machine_mgr)
config = config_mgr.config
if not config.machine:
    machine = list(sorted(machine_mgr.machines.values()))[0]
    config.set_machine(machine)
print(f"Config loaded. Using machine {config.machine.id}")

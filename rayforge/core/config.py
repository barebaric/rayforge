import yaml
import logging
from blinker import Signal
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional
from ..machine.models.machine import Machine


logger = logging.getLogger(__name__)


class OpsColorMode(Enum):
    """Enum for ops color source options."""

    LASER = "laser"
    LAYER = "layer"


class StartupBehavior(Enum):
    """Enum for application startup behavior options."""

    NONE = "none"
    LAST_PROJECT = "last_project"
    SPECIFIC_PROJECT = "specific_project"


@dataclass
class CanvasViewState:
    """Persistent view toggle states for the 2D/3D canvases."""

    show_workpieces: bool = True
    show_camera: bool = True
    show_travel_lines: bool = False
    show_nogo_zones: bool = True
    show_grid: bool = True
    show_models: bool = True
    show_tabs: bool = True
    perspective_mode: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanvasViewState":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class Config:
    def __init__(self):
        self.machine: Optional[Machine] = None
        self.theme: str = "system"
        # Default user preferences for units. Key is quantity, value is
        # unit name.
        self.unit_preferences: Dict[str, str] = {
            "length": "mm",
            "speed": "mm/min",
            "acceleration": "mm/s²",
        }
        # Startup behavior: "none", "last_project", or "specific_project"
        self.startup_behavior: str = StartupBehavior.NONE.value
        # Path to the specific project to open on startup (when
        # startup_behavior is SPECIFIC_PROJECT)
        self.startup_project_path: Optional[Path] = None
        # Track the last opened project path
        self.last_opened_project: Optional[Path] = None
        # UI visibility states
        self.bottom_panel: Optional[Dict[str, Any]] = None
        self.right_panel_visible: bool = True
        self.canvas_view: CanvasViewState = CanvasViewState()
        self.auto_pipeline: bool = True
        self.ops_color_mode: OpsColorMode = OpsColorMode.LASER
        self.check_for_app_updates: bool = True
        # Usage tracking consent date: None = not asked, "" = declined,
        # ISO date string = consent given on that date
        self.usage_consent_date: Optional[str] = None
        # Default DPI for unitless SVG imports
        self.import_dpi: float = 96.0
        self.changed = Signal()

    def set_machine(self, machine: Optional[Machine]):
        if self.machine == machine:
            return
        if self.machine:
            self.machine.changed.disconnect(self.changed.send)
        self.machine = machine
        self.changed.send(self)
        if self.machine:
            self.machine.changed.connect(self.changed.send)

    def set_theme(self, theme: str):
        """Sets the application theme preference."""
        if self.theme == theme:
            return
        self.theme = theme
        self.changed.send(self)

    def set_unit_preference(self, quantity: str, unit_name: str):
        """Sets the user's preferred display unit for a quantity."""
        if self.unit_preferences.get(quantity) == unit_name:
            return
        self.unit_preferences[quantity] = unit_name
        self.changed.send(self)

    def set_startup_behavior(self, behavior: StartupBehavior):
        """Sets the startup behavior preference."""
        behavior_value = behavior.value
        if self.startup_behavior == behavior_value:
            return
        self.startup_behavior = behavior_value
        self.changed.send(self)

    def set_startup_project_path(self, path: Optional[Path]):
        """Sets the specific project path to open on startup."""
        if self.startup_project_path == path:
            return
        self.startup_project_path = path
        self.changed.send(self)

    def set_last_opened_project(self, path: Optional[Path]):
        """Sets the last opened project path."""
        if self.last_opened_project == path:
            return
        self.last_opened_project = path
        self.changed.send(self)

    def set_bottom_panel(self, data: Optional[Dict[str, Any]]):
        if self.bottom_panel == data:
            return
        self.bottom_panel = data
        self.changed.send(self)

    def set_right_panel_visible(self, visible: bool):
        """Sets the right panel visibility state."""
        if self.right_panel_visible == visible:
            return
        self.right_panel_visible = visible
        self.changed.send(self)

    def set_import_dpi(self, dpi: float):
        """Sets the default DPI for unitless SVG imports."""
        if self.import_dpi == dpi:
            return
        self.import_dpi = dpi
        self.changed.send(self)

    def set_auto_pipeline(self, enabled: bool):
        """Sets whether the pipeline recalculates automatically."""
        if self.auto_pipeline == enabled:
            return
        self.auto_pipeline = enabled
        self.changed.send(self)

    def set_check_for_app_updates(self, enabled: bool):
        """Sets whether to check for application updates on startup."""
        if self.check_for_app_updates == enabled:
            return
        self.check_for_app_updates = enabled
        self.changed.send(self)

    def set_ops_color_mode(self, mode: OpsColorMode):
        """Sets the ops color mode."""
        if self.ops_color_mode == mode:
            return
        self.ops_color_mode = mode
        self.changed.send(self)

    def set_usage_consent(self, consent: bool):
        """Sets the usage tracking consent preference."""
        new_value = ""
        if consent:
            new_value = datetime.now().isoformat()
        if self.usage_consent_date == new_value:
            return
        self.usage_consent_date = new_value
        self.changed.send(self)

    @property
    def has_consented_tracking(self) -> bool:
        """Returns True if user has consented to usage tracking after
        the current policy date."""
        if not self.usage_consent_date or self.usage_consent_date == "":
            return False
        try:
            consent_date = datetime.fromisoformat(self.usage_consent_date)
            policy_date = datetime(2026, 2, 24)
            return consent_date >= policy_date
        except (ValueError, TypeError):
            return False

    @property
    def has_declined_tracking(self) -> bool:
        """Returns True if user has explicitly declined usage tracking."""
        return self.usage_consent_date == ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine": self.machine.id if self.machine else None,
            "theme": self.theme,
            "unit_preferences": self.unit_preferences,
            "startup_behavior": self.startup_behavior,
            "startup_project_path": (
                str(self.startup_project_path)
                if self.startup_project_path
                else None
            ),
            "last_opened_project": (
                str(self.last_opened_project)
                if self.last_opened_project
                else None
            ),
            "bottom_panel": self.bottom_panel,
            "right_panel_visible": self.right_panel_visible,
            "canvas_view": self.canvas_view.to_dict(),
            "auto_pipeline": self.auto_pipeline,
            "check_for_app_updates": self.check_for_app_updates,
            "ops_color_mode": self.ops_color_mode.value,
            "usage_consent_date": self.usage_consent_date,
            "import_dpi": self.import_dpi,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], get_machine_by_id) -> "Config":
        config = cls()
        config.theme = data.get("theme", "system")

        # Load unit preferences, falling back to defaults for safety
        default_prefs = {
            "length": "mm",
            "speed": "mm/min",
            "acceleration": "mm/s²",
        }
        loaded_prefs = data.get("unit_preferences", default_prefs)
        # Ensure all default keys are present
        default_prefs.update(loaded_prefs)
        config.unit_preferences = default_prefs

        # Load startup behavior
        default_behavior = StartupBehavior.NONE.value
        startup_behavior = data.get("startup_behavior", default_behavior)
        try:
            StartupBehavior(startup_behavior)
            config.startup_behavior = startup_behavior
        except ValueError:
            logger.warning(
                f"Invalid startup behavior in config: {startup_behavior}. "
                f"Using default: {default_behavior}"
            )
            config.startup_behavior = default_behavior

        # Load startup project path
        startup_project_path_str = data.get("startup_project_path")
        if startup_project_path_str:
            config.startup_project_path = Path(startup_project_path_str)

        # Load last opened project path
        last_opened_project_str = data.get("last_opened_project")
        if last_opened_project_str:
            config.last_opened_project = Path(last_opened_project_str)

        # Load UI visibility states
        config.bottom_panel = data.get("bottom_panel", None)
        config.right_panel_visible = data.get("right_panel_visible", True)
        config.canvas_view = CanvasViewState.from_dict(
            data.get("canvas_view", {})
        )
        config.auto_pipeline = data.get("auto_pipeline", True)
        config.check_for_app_updates = data.get("check_for_app_updates", True)

        ops_color_mode_str = data.get(
            "ops_color_mode", OpsColorMode.LASER.value
        )
        try:
            config.ops_color_mode = OpsColorMode(ops_color_mode_str)
        except ValueError:
            config.ops_color_mode = OpsColorMode.LASER

        # Load usage tracking consent date
        config.usage_consent_date = data.get("usage_consent_date", None)

        # Load import DPI
        config.import_dpi = data.get("import_dpi", 96.0)

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

        # Load first, which may trigger 'changed' signals if defaults are set
        self.load()
        # Connect the auto-save handler *after* loading is complete.
        self.config.changed.connect(self._on_config_changed)
        # Listen to machine removal to update config if needed
        self.machine_mgr.machine_removed.connect(self._on_machine_removed)

    def _on_config_changed(self, sender, **kwargs):
        self.save()

    def _on_machine_removed(self, sender, machine_id):
        """Handle machine removal by clearing config reference if needed."""
        if self.config.machine and self.config.machine.id == machine_id:
            msg = f"Current machine {machine_id} removed, clearing config"
            logger.info(msg)
            # Clear the machine reference
            self.config.set_machine(None)
            # If there are other machines available, select the first one
            if self.machine_mgr.machines:
                # Sort by ID for deterministic selection
                first_machine = list(
                    sorted(
                        self.machine_mgr.machines.values(), key=lambda m: m.id
                    )
                )[0]
                self.config.set_machine(first_machine)
                logger.info(f"Selected new machine {first_machine.id}")

    def save(self):
        if not self.config:
            return
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w") as f:
            yaml.safe_dump(self.config.to_dict(), f)

    def load(self) -> "Config":
        if not self.filepath.exists():
            logger.info("Config file does not exist, creating default config.")
            self.config = Config()
            return self.config

        try:
            with open(self.filepath, "r") as f:
                data = yaml.safe_load(f)
                if not data:
                    logger.info(
                        "Config file is empty, creating default config."
                    )
                    self.config = Config()
                else:
                    machine_id = data.get("machine")
                    logger.info(
                        f"Loading config with machine_id: {machine_id}"
                    )
                    self.config = Config.from_dict(
                        data, self.machine_mgr.get_machine_by_id
                    )
                    if self.config.machine:
                        logger.info(
                            f"Config loaded with machine: "
                            f"{self.config.machine.id} "
                            f"({self.config.machine.name})"
                        )
                    else:
                        logger.info("Config loaded but no machine set.")
        except (IOError, yaml.YAMLError) as e:
            logger.error(
                f"Failed to load config file: {e}. Creating a default config."
            )
            self.config = Config()

        return self.config

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .ai_service import AIService

logger = logging.getLogger(__name__)


class AIConfigManager:
    """Manages AI configuration persistence."""

    def __init__(self, filepath: Path, ai_service: "AIService"):
        self.filepath = filepath
        self.ai_service = ai_service
        self._saving = False
        self.ai_service.changed.connect(self._on_service_changed)

    def _on_service_changed(self, sender):
        self.save()

    def save(self):
        """Save AI configuration to file."""
        if self._saving:
            return

        self._saving = True
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            data = self.ai_service.to_dict()
            with open(self.filepath, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
            logger.debug(f"Saved AI config to {self.filepath}")
        except IOError as e:
            logger.error(f"Failed to save AI config: {e}")
        finally:
            self._saving = False

    def load(self):
        """Load AI configuration from file."""
        if not self.filepath.exists():
            logger.debug("AI config file not found, using defaults")
            return

        try:
            with open(self.filepath, "r") as f:
                data = yaml.safe_load(f)
            if data:
                self.ai_service.load_from_config(data)
                logger.info(f"Loaded AI config from {self.filepath}")
        except (yaml.YAMLError, IOError) as e:
            logger.error(f"Failed to load AI config: {e}")

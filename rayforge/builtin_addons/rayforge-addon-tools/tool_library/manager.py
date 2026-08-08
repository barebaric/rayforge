"""
ToolManager: loads, stores, and persists :class:`Tool` objects as YAML.

Modelled on :class:`rayforge.core.recipe_manager.RecipeManager`. One
``*.yaml`` file per tool, named ``<uid>.yaml``. Emits a ``changed``
signal (blinker) after any mutation so dependent UI can refresh.
"""

import logging
from pathlib import Path

import yaml
from blinker import Signal

from .tool import Tool

logger = logging.getLogger(__name__)


class ToolManager:
    """CRUD manager for the user tool library."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.tools: dict[str, Tool] = {}
        self.changed = Signal()
        self.load()

    def _file_for(self, uid: str) -> Path:
        return self.base_dir / f"{uid}.yaml"

    def load(self) -> None:
        """Reload every tool YAML from ``base_dir`` into memory."""
        self.tools.clear()
        for path in self.base_dir.glob("*.yaml"):
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                if not data:
                    continue
                tool = Tool.from_dict(data)
                self.tools[tool.uid] = tool
            except Exception as e:
                logger.error(f"Error loading tool {path.name}: {e}")
        logger.info(f"Loaded {len(self.tools)} tools.")

    def _save(self, tool: Tool) -> None:
        path = self._file_for(tool.uid)
        try:
            with open(path, "w") as f:
                yaml.safe_dump(tool.to_dict(), f, sort_keys=False)
        except Exception as e:
            logger.error(f"Failed to save tool {tool.uid}: {e}")

    def save(self, tool: Tool) -> None:
        """
        Add or update a tool, persist it, and emit ``changed``.
        """
        is_new = tool.uid not in self.tools
        self.tools[tool.uid] = tool
        self._save(tool)
        logger.debug(
            f"{'Added' if is_new else 'Updated'} tool {tool.uid} ({tool.name})"
        )
        self.changed.send(self)

    def delete(self, uid: str) -> bool:
        """Remove a tool by uid, delete its file, and emit ``changed``."""
        if uid not in self.tools:
            return False
        del self.tools[uid]
        path = self._file_for(uid)
        if path.exists():
            try:
                path.unlink()
            except OSError as e:
                logger.error(f"Failed to delete tool file {path}: {e}")
        self.changed.send(self)
        return True

    def get(self, uid: str) -> Tool | None:
        """Return the tool with this uid, or ``None``."""
        return self.tools.get(uid)

    def get_all(self) -> list[Tool]:
        """Return all tools, sorted by name for stable display."""
        return sorted(self.tools.values(), key=lambda t: t.name.lower())

import logging
import yaml
from pathlib import Path
from blinker import Signal
from .dialect import GcodeDialect, _DIALECT_REGISTRY, register_dialect
from .dialect_builtins import BUILTIN_DIALECTS


logger = logging.getLogger(__name__)


class DialectManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.dialects_changed = Signal()
        self.load_all()

    def _load_builtins(self):
        """Loads the hardcoded, built-in dialects into the registry."""
        for dialect in BUILTIN_DIALECTS:
            dialect.is_custom = False  # Enforce built-in status
            try:
                register_dialect(dialect)
            except ValueError as e:
                logger.error(f"Failed to register built-in dialect: {e}")

    def _load_custom_dialects(self):
        """Loads user-defined dialects from individual YAML files."""
        for f in self.base_dir.glob("*.yaml"):
            try:
                with open(f, "r") as stream:
                    data = yaml.safe_load(stream)
                    dialect = GcodeDialect.from_dict(data)
                    dialect.is_custom = True
                    register_dialect(dialect)
            except (yaml.YAMLError, ValueError, TypeError) as e:
                logger.error(f"Failed to load custom dialect from {f}: {e}")

    def load_all(self):
        """Clears the registry and reloads all dialects."""
        _DIALECT_REGISTRY.clear()
        self._load_builtins()
        self._load_custom_dialects()
        self.dialects_changed.send(self)

    def _save_dialect_to_file(self, dialect: GcodeDialect):
        """Saves a single custom dialect to its own YAML file."""
        if not dialect.is_custom:
            return
        file_path = self.base_dir / f"{dialect.uid}.yaml"
        try:
            with open(file_path, "w") as f:
                yaml.safe_dump(dialect.to_dict(), f, sort_keys=False)
        except (IOError, yaml.YAMLError) as e:
            logger.error(f"Failed to save custom dialect to {file_path}: {e}")

    def _delete_dialect_file(self, dialect: GcodeDialect):
        """Deletes the file for a single custom dialect."""
        if not dialect.is_custom:
            return
        file_path = self.base_dir / f"{dialect.uid}.yaml"
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError as e:
            logger.error(f"Error removing dialect file {file_path}: {e}")

    def add_dialect(self, dialect: GcodeDialect):
        """Adds a new custom dialect, saves, and signals."""
        if not dialect.is_custom:
            raise ValueError("Cannot add a non-custom dialect.")
        register_dialect(dialect)
        self._save_dialect_to_file(dialect)
        self.dialects_changed.send(self)

    def update_dialect(self, dialect: GcodeDialect):
        """Updates an existing custom dialect, saves, and signals."""
        if not dialect.is_custom:
            raise ValueError("Cannot update a built-in dialect.")

        uid_key = dialect.uid.lower()
        if uid_key not in _DIALECT_REGISTRY:
            raise ValueError(f"Dialect with UID '{dialect.uid}' not found.")

        _DIALECT_REGISTRY[uid_key] = dialect
        self._save_dialect_to_file(dialect)
        self.dialects_changed.send(self)

    def delete_dialect(self, dialect: GcodeDialect):
        """Deletes a custom dialect, saves, and signals."""
        if not dialect.is_custom:
            raise ValueError("Cannot delete a built-in dialect.")

        uid_key = dialect.uid.lower()
        if uid_key not in _DIALECT_REGISTRY:
            return  # Already gone

        del _DIALECT_REGISTRY[uid_key]
        self._delete_dialect_file(dialect)
        self.dialects_changed.send(self)

import logging
from dataclasses import replace
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import yaml
from pathlib import Path
from blinker import Signal
from gettext import gettext as _
from .dialect import (
    BUILTIN_DIALECTS,
    GcodeDialect,
)

if TYPE_CHECKING:
    from .machine import Machine


logger = logging.getLogger(__name__)


class DialectManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, GcodeDialect] = {}
        self.dialects_changed = Signal()
        self.load_all()

    def get(self, uid: str) -> GcodeDialect:
        """
        Retrieves a GcodeDialect instance from the registry by its
        case-insensitive UID.
        """
        uid_key = uid.lower()
        dialect = self._registry.get(uid_key)
        if not dialect:
            raise ValueError(
                f"Unknown or unsupported G-code dialect UID: '{uid}'"
            )
        return dialect

    def get_all(self) -> List[GcodeDialect]:
        """Returns a list of all registered GcodeDialect instances."""
        return sorted(self._registry.values(), key=lambda d: d.label)

    def register(self, dialect: GcodeDialect):
        """
        Adds a dialect to the registry, keyed by its case-insensitive
        unique `uid`.
        """
        uid_key = dialect.uid.lower()
        if uid_key in self._registry:
            raise ValueError(
                f"Dialect with UID '{dialect.uid}' is already registered."
            )
        self._registry[uid_key] = dialect

    def migrate_builtin_dialect_to_copy(
        self, dialect_uid: Optional[str], machine_name: str
    ) -> Tuple[Optional[str], bool]:
        """
        If dialect_uid references a built-in dialect, creates an isolated
        copy and returns the new UID with migrated=True. Otherwise returns
        the original UID with migrated=False.

        This ensures user configurations are isolated from built-in dialect
        changes during app upgrades.
        """
        if dialect_uid is None:
            return None, False
        try:
            dialect = self.get(dialect_uid)
        except ValueError:
            logger.warning(
                f"Dialect '{dialect_uid}' not found, falling back to 'grbl'."
            )
            dialect = self.get("grbl")

        if not dialect.is_custom:
            new_label = _("{label} (for {machine_name})").format(
                label=dialect.label,
                machine_name=machine_name,
            )
            new_dialect = dialect.copy_as_custom(new_label=new_label)
            new_dialect = replace(new_dialect, parent_uid=None)
            self.add_dialect(new_dialect)
            logger.info(
                f"Migrated built-in dialect '{dialect_uid}' to isolated "
                f"copy '{new_dialect.uid}' for machine '{machine_name}'."
            )
            return new_dialect.uid, True

        return dialect_uid, False

    def _load_builtins(self):
        """Loads the hardcoded, built-in dialects into the registry."""
        for dialect in BUILTIN_DIALECTS:
            dialect.is_custom = False
            try:
                self.register(dialect)
            except ValueError as e:
                logger.error(f"Failed to register built-in dialect: {e}")

    def _load_custom_dialects(self):
        """Loads user-defined dialects from individual YAML files."""
        for f in self.base_dir.glob("*.yaml"):
            try:
                with open(f, "r") as stream:
                    data = yaml.safe_load(stream)
                    dialect = GcodeDialect.from_dict(
                        data, registry=self._registry
                    )
                    dialect.is_custom = True
                    self.register(dialect)
            except (yaml.YAMLError, ValueError, TypeError) as e:
                logger.error(f"Failed to load custom dialect from {f}: {e}")

    def load_all(self):
        """Clears the registry and reloads all dialects."""
        self._registry.clear()
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
        self.register(dialect)
        self._save_dialect_to_file(dialect)
        self.dialects_changed.send(self)

    def update_dialect(self, dialect: GcodeDialect):
        """Updates an existing custom dialect, saves, and signals."""
        if not dialect.is_custom:
            raise ValueError("Cannot update a built-in dialect.")

        uid_key = dialect.uid.lower()
        if uid_key not in self._registry:
            raise ValueError(f"Dialect with UID '{dialect.uid}' not found.")

        self._registry[uid_key] = dialect
        self._save_dialect_to_file(dialect)
        self.dialects_changed.send(self)

    def get_machines_using_dialect(
        self, dialect: GcodeDialect, machines: List["Machine"]
    ) -> List["Machine"]:
        """Returns a list of machines that use the given dialect."""
        return [m for m in machines if m.dialect_uid == dialect.uid]

    def delete_dialect(self, dialect: GcodeDialect, machines: List["Machine"]):
        """Deletes a custom dialect, saves, and signals."""
        if not dialect.is_custom:
            raise ValueError("Cannot delete a built-in dialect.")

        uid_key = dialect.uid.lower()
        if uid_key not in self._registry:
            return

        machines_using = self.get_machines_using_dialect(dialect, machines)
        if machines_using:
            raise ValueError("Dialect is in use by one or more machines.")

        del self._registry[uid_key]
        self._delete_dialect_file(dialect)
        self.dialects_changed.send(self)

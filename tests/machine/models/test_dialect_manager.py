import pytest
from pathlib import Path
import tempfile
import yaml

from rayforge.machine.models.dialect import (
    _DIALECT_REGISTRY,
    get_dialect,
    GRBL_DIALECT,
    BUILTIN_DIALECTS,
)
from rayforge.machine.models.dialect_manager import DialectManager
from rayforge.machine.models.machine import Machine

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture
def temp_dialect_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def clean_registry():
    _DIALECT_REGISTRY.clear()
    for dialect in BUILTIN_DIALECTS:
        dialect.is_custom = False
        _DIALECT_REGISTRY[dialect.uid.lower()] = dialect
    yield
    _DIALECT_REGISTRY.clear()


class TestMigrateBuiltinDialectToCopy:
    """Tests for DialectManager.migrate_builtin_dialect_to_copy()."""

    def test_migrates_builtin_dialect(self, temp_dialect_dir, clean_registry):
        mgr = DialectManager(temp_dialect_dir)
        initial_count = len(_DIALECT_REGISTRY)

        new_uid, migrated = mgr.migrate_builtin_dialect_to_copy(
            "grbl", "Test Machine"
        )

        assert migrated is True
        assert new_uid != "grbl"
        assert len(_DIALECT_REGISTRY) == initial_count + 1

        dialect = get_dialect(new_uid)
        assert dialect.is_custom is True
        assert dialect.parent_uid is None
        assert "grbl" in dialect.label.lower()
        assert "Test Machine" in dialect.label

    def test_migrated_dialect_is_saved(self, temp_dialect_dir, clean_registry):
        mgr = DialectManager(temp_dialect_dir)

        new_uid, _ = mgr.migrate_builtin_dialect_to_copy(
            "grbl", "Test Machine"
        )

        yaml_file = temp_dialect_dir / f"{new_uid}.yaml"
        assert yaml_file.exists()

    def test_does_not_migrate_custom_dialect(
        self, temp_dialect_dir, clean_registry
    ):
        mgr = DialectManager(temp_dialect_dir)
        custom = GRBL_DIALECT.copy_as_custom("My Custom")
        mgr.add_dialect(custom)
        initial_count = len(_DIALECT_REGISTRY)

        new_uid, migrated = mgr.migrate_builtin_dialect_to_copy(
            custom.uid, "Test Machine"
        )

        assert migrated is False
        assert new_uid == custom.uid
        assert len(_DIALECT_REGISTRY) == initial_count

    def test_fallback_to_grbl_for_unknown_dialect(
        self, temp_dialect_dir, clean_registry
    ):
        mgr = DialectManager(temp_dialect_dir)
        initial_count = len(_DIALECT_REGISTRY)

        new_uid, migrated = mgr.migrate_builtin_dialect_to_copy(
            "nonexistent", "Test Machine"
        )

        assert migrated is True
        assert new_uid != "nonexistent"
        assert new_uid != "grbl"
        assert len(_DIALECT_REGISTRY) == initial_count + 1
        dialect = get_dialect(new_uid)
        assert dialect.is_custom is True

    def test_migration_creates_isolated_copy(
        self, temp_dialect_dir, clean_registry
    ):
        mgr = DialectManager(temp_dialect_dir)

        new_uid, _ = mgr.migrate_builtin_dialect_to_copy(
            "grbl", "Test Machine"
        )

        migrated_dialect = get_dialect(new_uid)
        original_dialect = get_dialect("grbl")

        assert migrated_dialect.laser_on == original_dialect.laser_on
        assert migrated_dialect.laser_off == original_dialect.laser_off
        assert migrated_dialect.preamble == original_dialect.preamble
        assert migrated_dialect.postscript == original_dialect.postscript
        assert migrated_dialect.is_custom is True
        assert original_dialect.is_custom is False

    def test_no_parent_uid_in_migrated_dialect(
        self, temp_dialect_dir, clean_registry
    ):
        mgr = DialectManager(temp_dialect_dir)

        new_uid, _ = mgr.migrate_builtin_dialect_to_copy(
            "smoothieware", "Test Machine"
        )

        dialect = get_dialect(new_uid)
        assert dialect.parent_uid is None

    def test_multiple_migrations_create_distinct_copies(
        self, temp_dialect_dir, clean_registry
    ):
        mgr = DialectManager(temp_dialect_dir)
        initial_count = len(_DIALECT_REGISTRY)

        uid1, _ = mgr.migrate_builtin_dialect_to_copy("grbl", "Machine A")
        uid2, _ = mgr.migrate_builtin_dialect_to_copy("grbl", "Machine B")
        uid3, _ = mgr.migrate_builtin_dialect_to_copy(
            "smoothieware", "Machine C"
        )

        assert uid1 != uid2 != uid3
        assert len(_DIALECT_REGISTRY) == initial_count + 3

        d1 = get_dialect(uid1)
        d2 = get_dialect(uid2)
        assert "Machine A" in d1.label
        assert "Machine B" in d2.label


class TestDialectManagerBasics:
    """Basic DialectManager functionality tests."""

    def test_load_all_clears_and_reloads(
        self, temp_dialect_dir, clean_registry
    ):
        mgr = DialectManager(temp_dialect_dir)

        custom = GRBL_DIALECT.copy_as_custom("Persisted Custom")
        mgr.add_dialect(custom)

        mgr.load_all()

        assert custom.uid in [d.uid for d in _DIALECT_REGISTRY.values()]

    def test_add_dialect_saves_to_file(self, temp_dialect_dir, clean_registry):
        mgr = DialectManager(temp_dialect_dir)

        custom = GRBL_DIALECT.copy_as_custom("Test Custom")
        mgr.add_dialect(custom)

        yaml_file = temp_dialect_dir / f"{custom.uid}.yaml"
        assert yaml_file.exists()

    def test_update_dialect_saves_changes(
        self, temp_dialect_dir, clean_registry
    ):
        mgr = DialectManager(temp_dialect_dir)

        custom = GRBL_DIALECT.copy_as_custom("Test Custom")
        mgr.add_dialect(custom)

        custom.laser_on = "M999 S{power}"
        mgr.update_dialect(custom)

        mgr.load_all()
        reloaded = get_dialect(custom.uid)
        assert reloaded.laser_on == "M999 S{power}"

    def test_delete_dialect_removes_file(
        self, temp_dialect_dir, clean_registry
    ):
        mgr = DialectManager(temp_dialect_dir)

        custom = GRBL_DIALECT.copy_as_custom("Test Custom")
        mgr.add_dialect(custom)

        yaml_file = temp_dialect_dir / f"{custom.uid}.yaml"
        assert yaml_file.exists()

        mgr.delete_dialect(custom, [])
        assert not yaml_file.exists()
        assert custom.uid not in _DIALECT_REGISTRY

    def test_cannot_delete_builtin_dialect(
        self, temp_dialect_dir, clean_registry
    ):
        mgr = DialectManager(temp_dialect_dir)

        with pytest.raises(ValueError, match="Cannot delete a built-in"):
            mgr.delete_dialect(GRBL_DIALECT, [])

    def test_cannot_update_builtin_dialect(
        self, temp_dialect_dir, clean_registry
    ):
        mgr = DialectManager(temp_dialect_dir)

        with pytest.raises(ValueError, match="Cannot update a built-in"):
            mgr.update_dialect(GRBL_DIALECT)

    def test_cannot_add_non_custom_dialect(
        self, temp_dialect_dir, clean_registry
    ):
        mgr = DialectManager(temp_dialect_dir)

        with pytest.raises(ValueError, match="Cannot add a non-custom"):
            mgr.add_dialect(GRBL_DIALECT)


class TestLegacyConfigMigration:
    """Tests using actual legacy config files from tests/machine/configs."""

    def test_legacy_dialect_uid_migration(self, lite_context):
        config_path = CONFIGS_DIR / "legacy_dialect_uid.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        machine = Machine.from_dict(data)

        assert machine.dialect_uid != "marlin"
        assert machine.dialect_migrated is True

        dialect = get_dialect(machine.dialect_uid)
        assert dialect.is_custom is True
        assert dialect.parent_uid is None
        assert "marlin" in dialect.label.lower()
        assert "Legacy Dialect UID Machine" in dialect.label

    def test_legacy_hooks_migration(self, lite_context):
        config_path = CONFIGS_DIR / "legacy_hooks.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        machine = Machine.from_dict(data)

        assert machine.dialect_uid != "grbl"
        assert "Legacy Hooks Machine" in get_dialect(machine.dialect_uid).label
        dialect = get_dialect(machine.dialect_uid)
        assert dialect.is_custom is True
        assert "G21 ; Set units to mm" in dialect.preamble
        assert "M5 ; Laser off" in dialect.postscript

    def test_machine_with_builtin_dialect_migration(self, lite_context):
        config_path = CONFIGS_DIR / "machine.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        machine = Machine.from_dict(data)

        assert machine.dialect_uid != "grbl"
        assert machine.dialect_migrated is True

        dialect = get_dialect(machine.dialect_uid)
        assert dialect.is_custom is True
        assert dialect.parent_uid is None
        assert "Default Machine" in dialect.label

    def test_migration_is_idempotent_from_config(self, lite_context):
        config_path = CONFIGS_DIR / "legacy_dialect_uid.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        machine1 = Machine.from_dict(data)
        first_uid = machine1.dialect_uid

        serialized = machine1.to_dict()
        machine2 = Machine.from_dict(serialized)

        assert machine2.dialect_uid == first_uid
        assert machine2.dialect_migrated is False

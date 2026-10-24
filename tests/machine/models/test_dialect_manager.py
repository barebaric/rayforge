import tempfile
from pathlib import Path

import pytest
import yaml

from rayforge.machine.models.dialect import GRBL_DIALECT
from rayforge.machine.models.dialect_manager import DialectManager
from rayforge.machine.models.machine import Machine

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture
def temp_dialect_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mgr(temp_dialect_dir):
    return DialectManager(temp_dialect_dir)


class TestMigrateBuiltinDialectToCopy:
    """Tests for DialectManager.migrate_builtin_dialect_to_copy()."""

    def test_migrates_builtin_dialect(self, mgr):
        initial_count = len(mgr._registry)

        new_uid, migrated = mgr.migrate_builtin_dialect_to_copy(
            "grbl", "Test Machine"
        )

        assert migrated is True
        assert new_uid != "grbl"
        assert len(mgr._registry) == initial_count + 1

        dialect = mgr.get(new_uid)
        assert dialect.is_custom is True
        assert dialect.parent_uid == "grbl"
        assert "grbl" in dialect.label.lower()
        assert "Test Machine" in dialect.label

    def test_migrated_dialect_is_saved(self, mgr, temp_dialect_dir):
        new_uid, _ = mgr.migrate_builtin_dialect_to_copy(
            "grbl", "Test Machine"
        )

        yaml_file = temp_dialect_dir / f"{new_uid}.yaml"
        assert yaml_file.exists()

    def test_does_not_migrate_custom_dialect(self, mgr):
        custom = GRBL_DIALECT.copy_as_custom("My Custom")
        mgr.add_dialect(custom)
        initial_count = len(mgr._registry)

        new_uid, migrated = mgr.migrate_builtin_dialect_to_copy(
            custom.uid, "Test Machine"
        )

        assert migrated is False
        assert new_uid == custom.uid
        assert len(mgr._registry) == initial_count

    def test_fallback_to_grbl_for_unknown_dialect(self, mgr):
        initial_count = len(mgr._registry)

        new_uid, migrated = mgr.migrate_builtin_dialect_to_copy(
            "nonexistent", "Test Machine"
        )

        assert migrated is True
        assert new_uid != "nonexistent"
        assert new_uid != "grbl"
        assert len(mgr._registry) == initial_count + 1
        dialect = mgr.get(new_uid)
        assert dialect.is_custom is True

    def test_migration_creates_isolated_copy(self, mgr):
        new_uid, _ = mgr.migrate_builtin_dialect_to_copy(
            "grbl", "Test Machine"
        )

        migrated_dialect = mgr.get(new_uid)
        original_dialect = mgr.get("grbl")

        assert migrated_dialect.laser_on == original_dialect.laser_on
        assert migrated_dialect.laser_off == original_dialect.laser_off
        assert migrated_dialect.preamble == original_dialect.preamble
        assert migrated_dialect.postscript == original_dialect.postscript
        assert migrated_dialect.is_custom is True
        assert original_dialect.is_custom is False

    def test_parent_uid_kept_in_migrated_dialect(self, mgr):
        new_uid, _ = mgr.migrate_builtin_dialect_to_copy(
            "smoothieware", "Test Machine"
        )

        dialect = mgr.get(new_uid)
        assert dialect.parent_uid == "smoothieware"

    def test_copy_inherits_fields_missing_from_storage(
        self, mgr, temp_dialect_dir
    ):
        """A stored copy predating a new built-in field picks it up
        from the parent on load, while stored fields stay untouched."""
        new_uid, _ = mgr.migrate_builtin_dialect_to_copy(
            "grbl", "Test Machine"
        )
        yaml_file = temp_dialect_dir / f"{new_uid}.yaml"
        data = yaml.safe_load(yaml_file.read_text())
        del data["move_to"]
        yaml.safe_dump(data, yaml_file.open("w"))

        mgr.load_all()

        assert mgr.get(new_uid).move_to == mgr.get("grbl").move_to

    def test_multiple_migrations_create_distinct_copies(self, mgr):
        initial_count = len(mgr._registry)

        uid1, _ = mgr.migrate_builtin_dialect_to_copy("grbl", "Machine A")
        uid2, _ = mgr.migrate_builtin_dialect_to_copy("grbl", "Machine B")
        uid3, _ = mgr.migrate_builtin_dialect_to_copy(
            "smoothieware", "Machine C"
        )

        assert uid1 != uid2 != uid3
        assert len(mgr._registry) == initial_count + 3

        d1 = mgr.get(uid1)
        d2 = mgr.get(uid2)
        assert "Machine A" in d1.label
        assert "Machine B" in d2.label


class TestDialectManagerBasics:
    """Basic DialectManager functionality tests."""

    def test_load_all_clears_and_reloads(self, mgr):
        custom = GRBL_DIALECT.copy_as_custom("Persisted Custom")
        mgr.add_dialect(custom)

        mgr.load_all()

        assert custom.uid in [d.uid for d in mgr._registry.values()]

    def test_add_dialect_saves_to_file(self, mgr, temp_dialect_dir):
        custom = GRBL_DIALECT.copy_as_custom("Test Custom")
        mgr.add_dialect(custom)

        yaml_file = temp_dialect_dir / f"{custom.uid}.yaml"
        assert yaml_file.exists()

    def test_update_dialect_saves_changes(self, mgr):
        custom = GRBL_DIALECT.copy_as_custom("Test Custom")
        mgr.add_dialect(custom)

        custom.laser_on = "M999 S{power}"
        mgr.update_dialect(custom)

        mgr.load_all()
        reloaded = mgr.get(custom.uid)
        assert reloaded.laser_on == "M999 S{power}"

    def test_delete_dialect_removes_file(self, mgr, temp_dialect_dir):
        custom = GRBL_DIALECT.copy_as_custom("Test Custom")
        mgr.add_dialect(custom)

        yaml_file = temp_dialect_dir / f"{custom.uid}.yaml"
        assert yaml_file.exists()

        mgr.delete_dialect(custom, [])
        assert not yaml_file.exists()
        assert custom.uid not in mgr._registry

    def test_cannot_delete_builtin_dialect(self, mgr):
        with pytest.raises(ValueError, match="Cannot delete a built-in"):
            mgr.delete_dialect(GRBL_DIALECT, [])

    def test_cannot_update_builtin_dialect(self, mgr):
        with pytest.raises(ValueError, match="Cannot update a built-in"):
            mgr.update_dialect(GRBL_DIALECT)

    def test_cannot_add_non_custom_dialect(self, mgr):
        with pytest.raises(ValueError, match="Cannot add a non-custom"):
            mgr.add_dialect(GRBL_DIALECT)


class TestUnusedDialectCleanup:
    """Tests for finding and deleting dialects no machine uses."""

    def test_unused_dialects_found(self, mgr, lite_context):
        used = GRBL_DIALECT.copy_as_custom("Used")
        unused = GRBL_DIALECT.copy_as_custom("Unused")
        mgr.add_dialect(used)
        mgr.add_dialect(unused)

        machine = Machine(lite_context)
        machine.dialect_uid = used.uid

        result = mgr.get_unused_dialects([machine])

        assert [d.uid for d in result] == [unused.uid]

    def test_builtin_dialects_never_reported_unused(self, mgr, lite_context):
        machine = Machine(lite_context)
        machine.dialect_uid = "no-machine-uses-this"

        result = mgr.get_unused_dialects([machine])

        assert all(d.is_custom for d in result)

    def test_unused_match_is_case_insensitive(self, mgr, lite_context):
        used = GRBL_DIALECT.copy_as_custom("Used")
        mgr.add_dialect(used)

        machine = Machine(lite_context)
        machine.dialect_uid = used.uid.upper()

        assert mgr.get_unused_dialects([machine]) == []

    def test_delete_unused_dialects_removes_files(
        self, mgr, lite_context, temp_dialect_dir
    ):
        used = GRBL_DIALECT.copy_as_custom("Used")
        unused_one = GRBL_DIALECT.copy_as_custom("Unused One")
        unused_two = GRBL_DIALECT.copy_as_custom("Unused Two")
        mgr.add_dialect(used)
        mgr.add_dialect(unused_one)
        mgr.add_dialect(unused_two)

        machine = Machine(lite_context)
        machine.dialect_uid = used.uid

        deleted = mgr.delete_unused_dialects([machine])

        assert {d.uid for d in deleted} == {unused_one.uid, unused_two.uid}
        assert unused_one.uid not in mgr._registry
        assert unused_two.uid not in mgr._registry
        assert not (temp_dialect_dir / f"{unused_one.uid}.yaml").exists()
        assert not (temp_dialect_dir / f"{unused_two.uid}.yaml").exists()
        assert (temp_dialect_dir / f"{used.uid}.yaml").exists()

    def test_delete_unused_dialects_returns_empty_when_all_used(
        self, mgr, lite_context
    ):
        used = GRBL_DIALECT.copy_as_custom("Used")
        mgr.add_dialect(used)

        machine = Machine(lite_context)
        machine.dialect_uid = used.uid

        assert mgr.delete_unused_dialects([machine]) == []

    def test_delete_unused_dialects_signals_change(self, mgr, lite_context):
        unused = GRBL_DIALECT.copy_as_custom("Unused")
        mgr.add_dialect(unused)
        signal_calls = []

        def signal_handler(sender):
            signal_calls.append(sender)

        mgr.dialects_changed.connect(signal_handler)

        mgr.delete_unused_dialects([])

        assert len(signal_calls) == 1

    def test_delete_unused_dialects_skips_signal_when_nothing_deleted(
        self, mgr, lite_context
    ):
        signal_calls = []

        def signal_handler(sender):
            signal_calls.append(sender)

        mgr.dialects_changed.connect(signal_handler)

        mgr.delete_unused_dialects([])

        assert signal_calls == []


class TestLegacyConfigMigration:
    """Tests using actual legacy config files from tests/machine/configs."""

    def test_legacy_dialect_uid_migration(self, lite_context):
        config_path = CONFIGS_DIR / "legacy_dialect_uid.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        machine = Machine.from_dict(data, context=lite_context)

        assert machine.dialect_uid != "marlin"
        assert machine.dialect_migrated is True

        dialect = lite_context.dialect_mgr.get(machine.dialect_uid)
        assert dialect.is_custom is True
        assert dialect.parent_uid == "marlin"
        assert "marlin" in dialect.label.lower()
        assert "Legacy Dialect UID Machine" in dialect.label

    def test_legacy_hooks_migration(self, lite_context):
        config_path = CONFIGS_DIR / "legacy_hooks.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        machine = Machine.from_dict(data, context=lite_context)

        assert machine.dialect_uid != "grbl"
        dialect = lite_context.dialect_mgr.get(machine.dialect_uid)
        assert "Legacy Hooks Machine" in dialect.label
        assert dialect.is_custom is True
        assert "G21 ; Set units to mm" in dialect.preamble
        assert "M5 ; Laser off" in dialect.postscript

    def test_machine_with_builtin_dialect_migration(self, lite_context):
        config_path = CONFIGS_DIR / "machine.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        machine = Machine.from_dict(data, context=lite_context)

        assert machine.dialect_uid != "grbl"
        assert machine.dialect_migrated is True

        dialect = lite_context.dialect_mgr.get(machine.dialect_uid)
        assert dialect.is_custom is True
        assert dialect.parent_uid == "grbl"
        assert "Default Machine" in dialect.label

    def test_migration_is_idempotent_from_config(self, lite_context):
        config_path = CONFIGS_DIR / "legacy_dialect_uid.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        machine1 = Machine.from_dict(data, context=lite_context)
        first_uid = machine1.dialect_uid

        serialized = machine1.to_dict()
        machine2 = Machine.from_dict(serialized, context=lite_context)

        assert machine2.dialect_uid == first_uid
        assert machine2.dialect_migrated is False

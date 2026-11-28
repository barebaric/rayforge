"""Tests for the versioned config schema migration chain."""

import copy

from rayforge.core.config import Config, OpsColorMode
from rayforge.core.config_schema import (
    CURRENT_CONFIG_SCHEMA_VERSION,
    migrate,
)


def test_migrate_renames_legacy_ops_color_mode():
    data = {"ops_color_mode": "layer"}
    migrated = migrate(copy.deepcopy(data))
    assert migrated["ops_color_mode"] == "workflow"
    assert migrated["schema_version"] == CURRENT_CONFIG_SCHEMA_VERSION


def test_migrate_is_idempotent():
    once = migrate({"ops_color_mode": "layer"})
    twice = migrate(copy.deepcopy(once))
    assert twice == once


def test_migrate_does_not_mutate_input():
    data = {"ops_color_mode": "layer"}
    migrate(data)
    assert data == {"ops_color_mode": "layer"}


def test_migrate_keeps_unknown_keys():
    migrated = migrate({"future_setting": 42})
    assert migrated["future_setting"] == 42


def test_config_loads_legacy_color_mode():
    data = {"ops_color_mode": "layer"}
    config = Config.from_dict(data, get_machine_by_id=lambda mid: None)
    assert config.ops_color_mode == OpsColorMode.WORKFLOW


def test_config_round_trip_stamps_version():
    config = Config()
    config.set_ops_color_mode(OpsColorMode.WORKFLOW)
    data = config.to_dict()
    assert data["schema_version"] == CURRENT_CONFIG_SCHEMA_VERSION

    restored = Config.from_dict(data, get_machine_by_id=lambda mid: None)
    assert restored.ops_color_mode == OpsColorMode.WORKFLOW

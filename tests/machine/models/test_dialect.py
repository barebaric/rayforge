import pytest
from rayforge.machine.models.dialect import (
    GcodeDialect,
    register_dialect,
    _DIALECT_REGISTRY,
)
from rayforge.machine.models.dialect_builtins import GRBL_DIALECT


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensures the registry is clean before each test."""
    _DIALECT_REGISTRY.clear()
    # Register a base dialect to serve as a parent for tests
    register_dialect(GRBL_DIALECT)
    yield
    _DIALECT_REGISTRY.clear()


def test_instantiation():
    """Test basic instantiation of a dialect."""
    dialect = GcodeDialect(
        label="Test Dialect",
        description="A test description",
        laser_on="M3",
        laser_off="M5",
        tool_change="T1",
        set_speed="",
        travel_move="G0",
        linear_move="G1",
        arc_cw="G2",
        arc_ccw="G3",
        air_assist_on="",
        air_assist_off="",
        home_all="$H",
        home_axis="",
        move_to="",
        clear_alarm="",
        set_wcs_offset="",
        probe_cycle="",
    )
    assert dialect.label == "Test Dialect"
    assert dialect.laser_on == "M3"
    assert isinstance(dialect.uid, str)
    assert not dialect.is_custom


def test_serialization():
    """Test to_dict serialization."""
    dialect = GcodeDialect(
        label="Serialization Test",
        description="Testing to_dict",
        laser_on="M4",
        laser_off="M5",
        tool_change="",
        set_speed="",
        travel_move="",
        linear_move="",
        arc_cw="",
        arc_ccw="",
        air_assist_on="",
        air_assist_off="",
        home_all="",
        home_axis="",
        move_to="",
        clear_alarm="",
        set_wcs_offset="G10 L2",
        probe_cycle="G38.2",
        preamble=["G21"],
        postscript=["M5"],
        uid="fixed-uid-123",
        is_custom=True,
    )
    data = dialect.to_dict()

    assert data["label"] == "Serialization Test"
    assert data["uid"] == "fixed-uid-123"
    assert data["is_custom"] is True
    assert data["preamble"] == ["G21"]
    assert data["set_wcs_offset"] == "G10 L2"


def test_deserialization_with_parent():
    """
    Test that deserializing data with missing fields correctly inherits
    defaults from the parent dialect.
    """
    # GRBL_DIALECT is already in the registry thanks to the fixture.

    partial_data = {
        "label": "My Custom GRBL",
        "parent_uid": GRBL_DIALECT.uid,
        "laser_on": "M3 S{power}",  # Override this specific field
        # Missing fields like laser_off, travel_move, etc. should be inherited
    }

    dialect = GcodeDialect.from_dict(partial_data)

    # Check override
    assert dialect.laser_on == "M3 S{power}"

    # Check inheritance from GRBL_DIALECT
    assert dialect.laser_off == GRBL_DIALECT.laser_off
    assert dialect.travel_move == GRBL_DIALECT.travel_move
    assert dialect.set_wcs_offset == GRBL_DIALECT.set_wcs_offset

    # Check identity fields
    assert dialect.parent_uid == GRBL_DIALECT.uid
    assert dialect.label == "My Custom GRBL"


def test_deserialization_fallback_to_grbl():
    """
    Test that if no parent is specified, it defaults to inheriting from GRBL.
    """
    partial_data = {
        "label": "Orphan Dialect",
        "description": "No parent specified",
    }

    dialect = GcodeDialect.from_dict(partial_data)

    # Should inherit default GRBL values
    assert dialect.laser_off == GRBL_DIALECT.laser_off
    assert dialect.home_all == GRBL_DIALECT.home_all
    assert dialect.set_wcs_offset == GRBL_DIALECT.set_wcs_offset


def test_copy_as_custom():
    """Test creating a custom copy of an existing dialect."""
    base = GRBL_DIALECT
    new_label = "My Custom Copy"

    custom = base.copy_as_custom(new_label)

    assert custom.label == new_label
    assert custom.is_custom is True
    assert custom.parent_uid == base.uid
    assert custom.uid != base.uid  # Must have a new UID

    # Content should match
    assert custom.laser_on == base.laser_on
    assert custom.preamble == base.preamble


def test_get_editor_varsets():
    """Test that the editor UI definition is generated correctly."""
    dialect = GRBL_DIALECT
    varsets = dialect.get_editor_varsets()

    assert "info" in varsets
    assert "templates" in varsets
    assert "scripts" in varsets

    # Check specific fields exist in the varsets
    info = varsets["info"]
    assert info["label"].value == dialect.label

    templates = varsets["templates"]
    assert templates["laser_on"].value == dialect.laser_on
    assert templates["set_wcs_offset"].value == dialect.set_wcs_offset

    scripts = varsets["scripts"]
    # TextAreaVar joins lists with newlines
    expected_preamble = "\n".join(dialect.preamble)
    assert scripts["preamble"].value == expected_preamble

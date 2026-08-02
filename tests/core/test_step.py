from unittest.mock import MagicMock

import pytest
from raygeo.geo import Matrix

from rayforge.core.capability import CUT
from rayforge.core.doc import Doc
from rayforge.core.step import Step
from rayforge.machine.models.laser import LaserHead


@pytest.fixture
def step():
    """Provides a basic, standalone Step instance."""
    return Step(typelabel="TestType", name="Test Step")


@pytest.fixture
def step_in_doc():
    """Provides a Step instance properly parented within a Doc hierarchy."""
    doc = Doc()
    layer = doc.active_layer
    workflow = layer.workflow
    step = Step(typelabel="HierarchyTest", name="Nested Step")
    assert workflow is not None
    workflow.add_child(step)
    return doc, layer, workflow, step


def test_step_initialization(step):
    """Tests that a new Step instance has the correct default values."""
    assert step.typelabel == "TestType"
    assert step.name == "Test Step"
    assert step.visible is True
    assert step.selected_head_uid is None
    assert step.generated_workpiece_uid is None
    assert step.applied_recipe_uid is None
    assert step.per_workpiece_transformers_dicts == []
    assert step.per_step_transformers_dicts == []
    assert step.pixels_per_mm == (50, 50)
    assert step.cut_speed == 500
    assert step.travel_speed == 5000


def test_setters_and_signals(step):
    """
    Tests that property setters update the value and fire the 'updated'
    signal.
    """
    handler = MagicMock()
    step.updated.connect(handler)

    step.set_cut_speed(1200)
    assert step.cut_speed == 1200
    handler.assert_called_once_with(step)
    handler.reset_mock()

    step.set_travel_speed(8000)
    assert step.travel_speed == 8000
    handler.assert_called_once_with(step)
    handler.reset_mock()

    step.set_selected_head_uid("laser-123")
    assert step.selected_head_uid == "laser-123"
    handler.assert_called_once_with(step)


def test_set_visible_fires_signals(step):
    """Tests that set_visible fires both visibility_changed and updated."""
    updated_handler = MagicMock()
    visibility_handler = MagicMock()
    step.updated.connect(updated_handler)
    step.visibility_changed.connect(visibility_handler)

    step.set_visible(False)
    assert step.visible is False
    updated_handler.assert_called_once_with(step)
    visibility_handler.assert_called_once_with(step)


def test_hierarchy_properties(step_in_doc):
    """Tests the .workflow and .layer properties."""
    doc, layer, workflow, step = step_in_doc
    assert step.workflow is workflow
    assert step.layer is layer


def test_hierarchy_properties_when_detached(step):
    """
    Tests that hierarchy properties return None when the step is detached.
    """
    assert step.workflow is None
    assert step.layer is None


def test_get_selected_head(step):
    """Tests the logic for retrieving the selected laser from a machine."""
    mock_machine = MagicMock()
    mock_laser1 = MagicMock(spec=LaserHead)
    mock_laser1.uid = "laser-1"
    mock_laser2 = MagicMock(spec=LaserHead)
    mock_laser2.uid = "laser-2"
    mock_machine.heads = [mock_laser1, mock_laser2]

    # Case 1: UID is set and exists
    step.set_selected_head_uid("laser-2")
    assert step.get_selected_head(mock_machine) is mock_laser2

    # Case 2: UID is set but does not exist, should fall back to first
    step.set_selected_head_uid("non-existent-laser")
    assert step.get_selected_head(mock_machine) is mock_laser1

    # Case 3: UID is None, should fall back to first
    step.set_selected_head_uid(None)
    assert step.get_selected_head(mock_machine) is mock_laser1

    # Case 4: Machine has no lasers
    mock_machine.heads = []
    assert step.get_selected_head(mock_machine) is None


def test_serialization_to_dict_all_properties(step):
    """
    Tests that to_dict() correctly serializes all relevant properties with
    non-default values.
    """
    step.name = "My Engrave Step"
    step.matrix = Matrix.translation(1, 2)
    step.visible = False
    step.selected_head_uid = "laser-abc"
    step.generated_workpiece_uid = "wp-xyz"
    step.applied_recipe_uid = "recipe-123"
    step.per_step_transformers_dicts = [{"type": "CoolingPause"}]
    step.pixels_per_mm = (100, 100)
    step.cut_speed = 1500
    step.travel_speed = 9000

    data = step.to_dict()

    assert data["uid"] == step.uid
    assert data["type"] == "step"
    assert data["name"] == "My Engrave Step"
    assert data["matrix"] == Matrix.translation(1, 2).to_list()
    assert data["typelabel"] == "TestType"
    assert data["visible"] is False
    assert data["selected_head_uid"] == "laser-abc"
    assert data["generated_workpiece_uid"] == "wp-xyz"
    assert data["applied_recipe_uid"] == "recipe-123"
    assert data["per_step_transformers_dicts"] == [{"type": "CoolingPause"}]
    assert data["pixels_per_mm"] == (100, 100)
    assert data["cut_speed"] == 1500
    assert data["travel_speed"] == 9000


def test_deserialization_from_dict(step):
    """Tests that from_dict() correctly restores all properties."""
    step_dict = {
        "uid": "step-123",
        "type": "step",
        "name": "Restored Step",
        "matrix": Matrix.rotation(45).to_list(),
        "typelabel": "RestoredType",
        "visible": False,
        "selected_head_uid": "laser-def",
        "generated_workpiece_uid": "wp-123",
        "applied_recipe_uid": "recipe-456",
        "per_workpiece_transformers_dicts": [],
        "per_step_transformers_dicts": [],
        "pixels_per_mm": (20, 20),
        "cut_speed": 2500,
        "max_cut_speed": 10000,
        "travel_speed": 10000,
        "max_travel_speed": 10000,
        "children": [],
    }

    restored = Step.from_dict(step_dict)

    assert restored.uid == "step-123"
    assert restored.name == "Restored Step"
    assert restored.matrix == Matrix.rotation(45)
    assert restored.typelabel == "RestoredType"
    assert restored.visible is False
    assert restored.selected_head_uid == "laser-def"
    assert restored.generated_workpiece_uid == "wp-123"
    assert restored.applied_recipe_uid == "recipe-456"
    assert restored.pixels_per_mm == (20, 20)
    assert restored.cut_speed == 2500
    assert restored.travel_speed == 10000


def test_deserialization_with_missing_keys(step):
    """
    Tests that from_dict() uses sensible defaults for missing optional keys.
    """
    minimal_dict = {
        "uid": "step-min",
        "type": "step",
        "typelabel": "MinimalType",
        "visible": True,
        "matrix": Matrix.identity().to_list(),
        "per_workpiece_transformers_dicts": [],
        "per_step_transformers_dicts": [],
    }

    restored = Step.from_dict(minimal_dict)

    assert restored.uid == "step-min"
    assert restored.name == "MinimalType"  # Falls back to typelabel
    assert restored.selected_head_uid is None
    assert restored.applied_recipe_uid is None
    assert restored.cut_speed == 500
    assert restored.pixels_per_mm == (100, 100)


def test_step_roundtrip_serialization():
    """
    Tests that serializing a Step and then deserializing it results in an
    equivalent object.
    """
    # 1. Create original Step with non-default values
    original = Step(typelabel="Roundtrip", name="My Step")
    original.uid = "roundtrip-uid"
    original.visible = False
    original.set_cut_speed(3000)
    original.selected_head_uid = "the-best-laser"
    original.applied_recipe_uid = "recipe-abc"
    original.matrix = Matrix.translation(50, 50)

    # 2. Serialize
    data = original.to_dict()

    # 3. Deserialize
    restored = Step.from_dict(data)

    # 4. Assert equivalence
    assert restored.uid == original.uid
    assert restored.name == original.name
    assert restored.typelabel == original.typelabel
    assert restored.visible == original.visible
    assert restored.cut_speed == original.cut_speed
    assert restored.selected_head_uid == original.selected_head_uid
    assert restored.applied_recipe_uid == original.applied_recipe_uid
    assert restored.matrix == original.matrix


def test_step_forward_compatibility_with_extra_fields():
    """
    Tests that from_dict() preserves extra fields from newer versions
    and to_dict() re-serializes them.
    """
    step_dict = {
        "uid": "step-forward-123",
        "type": "step",
        "name": "Future Step",
        "matrix": Matrix.identity().to_list(),
        "typelabel": "FutureType",
        "visible": True,
        "selected_head_uid": None,
        "generated_workpiece_uid": None,
        "applied_recipe_uid": None,
        "per_workpiece_transformers_dicts": [],
        "per_step_transformers_dicts": [],
        "pixels_per_mm": (50, 50),
        "cut_speed": 500,
        "max_cut_speed": 10000,
        "travel_speed": 5000,
        "max_travel_speed": 10000,
        "children": [],
        "future_field_string": "some value",
        "future_field_number": 42,
        "future_field_dict": {"nested": "data"},
    }

    step = Step.from_dict(step_dict)

    assert step.extra["future_field_string"] == "some value"
    assert step.extra["future_field_number"] == 42
    assert step.extra["future_field_dict"] == {"nested": "data"}

    data = step.to_dict()
    assert data["future_field_string"] == "some value"
    assert data["future_field_number"] == 42
    assert data["future_field_dict"] == {"nested": "data"}


def test_step_backward_compatibility_with_missing_optional_fields():
    """
    Tests that from_dict() handles missing optional fields gracefully
    (simulating data from an older version).
    """
    minimal_dict = {
        "uid": "step-backward-123",
        "type": "step",
        "name": "Old Step",
        "matrix": Matrix.identity().to_list(),
        "typelabel": "OldType",
        "visible": True,
        "per_workpiece_transformers_dicts": [],
        "per_step_transformers_dicts": [],
        "children": [],
    }

    step = Step.from_dict(minimal_dict)

    assert step.selected_head_uid is None
    assert step.generated_workpiece_uid is None
    assert step.applied_recipe_uid is None
    assert step.pixels_per_mm == (100, 100)
    assert step.cut_speed == 500
    assert step.max_cut_speed == 10000
    assert step.travel_speed == 5000
    assert step.max_travel_speed == 10000


def test_legacy_selected_laser_uid_key_migrates():
    """
    Old project files keyed head selection as "selected_laser_uid".
    It must load into selected_head_uid and not pollute extra.
    """
    step_dict = {
        "uid": "step-legacy-123",
        "type": "step",
        "typelabel": "LegacyType",
        "visible": True,
        "matrix": Matrix.identity().to_list(),
        "per_workpiece_transformers_dicts": [],
        "per_step_transformers_dicts": [],
        "selected_laser_uid": "old-laser-uid",
    }

    step = Step.from_dict(step_dict)

    assert step.selected_head_uid == "old-laser-uid"
    assert "selected_laser_uid" not in step.extra

    data = step.to_dict()
    assert data["selected_head_uid"] == "old-laser-uid"
    assert "selected_laser_uid" not in data


def test_capability_defaults_applied_in_constructor():
    """
    Capability defaults no longer drive attribute creation on core
    ``Step`` — domain bases declare their defaults explicitly. Laser
    defaults are covered by the laser addon tests (see
    ``test_laser_step.py``).
    """
    step = Step(typelabel="Test")

    assert step.cut_speed == 500
    assert step.travel_speed == 5000
    assert not hasattr(step, "power")


def test_deserialization_with_missing_step_class():
    """
    Tests that from_dict() handles a missing step class gracefully
    by falling back to the base Step class.
    """
    step_dict = {
        "uid": "step-missing-class-123",
        "type": "step",
        "step_type": "NonExistentStepClass",
        "name": "Missing Step",
        "matrix": Matrix.identity().to_list(),
        "typelabel": "UnknownType",
        "visible": True,
        "per_workpiece_transformers_dicts": [],
        "per_step_transformers_dicts": [],
        "children": [],
    }

    step = Step.from_dict(step_dict)

    assert isinstance(step, Step)
    assert step.uid == "step-missing-class-123"
    assert step.name == "Missing Step"
    assert step.typelabel == "UnknownType"
    assert step.extra == {}


def test_capabilities_property_returns_class_caps():
    """capabilities exposes the class-level CAPABILITIES."""

    class CutStep(Step):
        CAPABILITIES = (CUT,)

    cs = CutStep(typelabel="Test")
    assert cs.capabilities == (CUT,)

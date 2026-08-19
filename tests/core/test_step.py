from unittest.mock import MagicMock

import pytest
from raygeo.geo import Matrix
from raygeo.ops.state import CoolantMode

from rayforge.core.doc import Doc
from rayforge.core.step import Step
from rayforge.machine.models.laser import LaserHead
from rayforge.machine.models.spindle import SpindleHead


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


def test_set_name_fires_updated(step):
    """Tests that set_name fires the 'updated' signal (issue #343)."""
    handler = MagicMock()
    step.updated.connect(handler)

    step.set_name("Renamed Step")
    assert step.name == "Renamed Step"
    handler.assert_called_with(step)


def test_set_name_no_signal_when_unchanged(step):
    """set_name must not fire when the name is identical."""
    handler = MagicMock()
    step.updated.connect(handler)

    step.set_name(step.name)
    handler.assert_not_called()


def test_set_name_propagates_to_workflow(step_in_doc):
    """Renaming a step bubbles 'descendant_updated' to the workflow."""
    _doc, _layer, workflow, step = step_in_doc
    handler = MagicMock()
    workflow.descendant_updated.connect(handler)

    step.set_name("New Workflow Name")
    assert step.name == "New Workflow Name"
    handler.assert_called()


def test_hierarchy_properties(step_in_doc):
    """Tests the .workflow and .layer properties."""
    _doc, layer, workflow, step = step_in_doc
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
    by falling back to the base Step class and preserving the
    original step type name for reporting.
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
    assert step.original_step_type == "NonExistentStepClass"
    assert step.extra == {}

    data = step.to_dict()
    assert data["step_type"] == "NonExistentStepClass"


def test_common_recipe_varset_groups_intersects_keys():
    """common_recipe_varset_groups keeps only keys shared by all types."""
    from rayforge.core.varset import BoolVar, SpeedVar, VarSet

    class StepA(Step):
        @classmethod
        def recipe_varset(cls):
            return VarSet(
                vars=[
                    SpeedVar(
                        key="cut_speed",
                        label="Cut",
                        default=1,
                        min_val=1,
                        role="cut",
                    ),
                    SpeedVar(
                        key="travel_speed",
                        label="Travel",
                        default=1,
                        min_val=1,
                        role="travel",
                    ),
                    BoolVar(key="only_a", label="A", default=False),
                ]
            )

    class StepB(Step):
        @classmethod
        def recipe_varset(cls):
            return VarSet(
                vars=[
                    SpeedVar(
                        key="cut_speed",
                        label="Cut",
                        default=1,
                        min_val=1,
                        role="cut",
                    ),
                    SpeedVar(
                        key="travel_speed",
                        label="Travel",
                        default=1,
                        min_val=1,
                        role="travel",
                    ),
                    BoolVar(key="only_b", label="B", default=False),
                ]
            )

    common = Step.common_recipe_varset_groups([StepA, StepB])
    keys = {var.key for _, varset in common for var in varset}
    assert keys == {"cut_speed", "travel_speed"}

    # A single class returns its own groups unchanged.
    single = Step.common_recipe_varset_groups([StepA])
    single_keys = {var.key for _, varset in single for var in varset}
    assert "only_a" in single_keys

    # No classes -> base Step groups.
    empty = Step.common_recipe_varset_groups([])
    base = Step.recipe_varset_groups()
    assert [title for title, _ in empty] == [title for title, _ in base]
    assert {var.key for _, varset in empty for var in varset} == {
        var.key for _, varset in base for var in varset
    }


def test_common_transformer_dicts_intersects_names():
    """common_transformer_dicts keeps only transformers shared by all types."""

    class StepA(Step):
        @classmethod
        def get_default_transformers_dicts(cls):
            return (
                [
                    {
                        "name": "Optimize",
                        "enabled": True,
                        "amount": 5,
                    },
                    {"name": "OnlyA", "enabled": True, "param": 1},
                ],
                [{"name": "PerStepOnly", "enabled": True}],
            )

    class StepB(Step):
        @classmethod
        def get_default_transformers_dicts(cls):
            return (
                [
                    {"name": "Optimize", "enabled": True, "amount": 9},
                    {"name": "OnlyB", "enabled": True},
                ],
                [{"name": "PerStepOnly", "enabled": True}],
            )

    common = Step.common_transformer_dicts([StepA, StepB])
    names = [d["name"] for d in common]
    assert names == ["Optimize", "PerStepOnly"]

    # Structural reference is the first type's dicts (copied).
    optimize = next(d for d in common if d["name"] == "Optimize")
    assert optimize["amount"] == 5

    # Copies are independent of the source dicts.
    optimize["amount"] = 42
    step_a_defaults, _ = StepA.get_default_transformers_dicts()
    assert step_a_defaults[0]["amount"] == 5

    # A single class returns its own transformer dicts.
    single = Step.common_transformer_dicts([StepA])
    single_names = {d["name"] for d in single}
    assert single_names == {"Optimize", "OnlyA", "PerStepOnly"}

    # No classes -> empty list.
    assert Step.common_transformer_dicts([]) == []

    # No shared transformers -> empty list.
    class StepC(Step):
        @classmethod
        def get_default_transformers_dicts(cls):
            return ([{"name": "OnlyC", "enabled": True}], [])

    assert Step.common_transformer_dicts([StepC, StepB]) == []


def test_dedupe_transformer_dicts_by_name_keeps_first():
    """_dedupe_transformer_dicts_by_name keeps the first dict per name."""
    first = {"name": "Optimize", "enabled": True, "amount": 1}
    second = {"name": "Optimize", "enabled": False, "amount": 2}
    other = {"name": "Smooth", "enabled": True}
    result = Step._dedupe_transformer_dicts_by_name([first, second, other])
    assert result["Optimize"] is first
    assert result["Smooth"] is other


def test_base_step_has_no_operation_color(step):
    """The generic step reports no operation color."""
    assert step.get_operation_color(None) is None


def test_coolant_method_defaults_to_off(step):
    """A new step uses no coolant by default."""
    assert step.coolant_method is CoolantMode.OFF


def test_set_coolant_method_fires_updated(step):
    """set_coolant_method updates the value and fires the updated signal."""
    handler = MagicMock()
    step.updated.connect(handler)
    step.set_coolant_method(CoolantMode.FLOOD)
    assert step.coolant_method is CoolantMode.FLOOD
    handler.assert_called_once_with(step)


def test_set_coolant_method_accepts_name_string(step):
    """set_coolant_method resolves varset/recipe name strings."""
    step.set_coolant_method("MIST")
    assert step.coolant_method is CoolantMode.MIST

    step.set_coolant_method("FLOOD")
    assert step.coolant_method is CoolantMode.FLOOD

    # Unknown names fall back to OFF.
    step.set_coolant_method("BOGUS")
    assert step.coolant_method is CoolantMode.OFF


def test_coolant_method_serialization_round_trip(step):
    """coolant_method survives a to_dict/from_dict round trip."""
    step.set_coolant_method(CoolantMode.MIST)
    restored = Step.from_dict(step.to_dict())
    assert restored.coolant_method is CoolantMode.MIST


def test_coolant_method_missing_key_loads_as_off():
    """Old project files without a coolant_method key load as OFF."""
    step_dict = {
        "uid": "step-no-coolant",
        "type": "step",
        "typelabel": "OldType",
        "visible": True,
        "matrix": Matrix.identity().to_list(),
        "per_workpiece_transformers_dicts": [],
        "per_step_transformers_dicts": [],
        "children": [],
    }
    restored = Step.from_dict(step_dict)
    assert restored.coolant_method is CoolantMode.OFF
    assert "coolant_method" not in restored.extra


def test_coolant_method_unknown_value_loads_as_off():
    """Unknown coolant method names from newer versions load as OFF."""
    step_dict = {
        "uid": "step-bad-coolant",
        "type": "step",
        "typelabel": "FutureType",
        "visible": True,
        "matrix": Matrix.identity().to_list(),
        "per_workpiece_transformers_dicts": [],
        "per_step_transformers_dicts": [],
        "children": [],
        "coolant_method": "VAPOR",
    }
    restored = Step.from_dict(step_dict)
    assert restored.coolant_method is CoolantMode.OFF
    assert "VAPOR" not in restored.extra


def _spindle_machine(cooling_methods):
    """A machine with a single spindle head supporting the given methods."""
    machine = MagicMock()
    head = SpindleHead()
    head.cooling_methods = tuple(cooling_methods)
    machine.heads = [head]
    return machine


def test_unsupported_coolant_off_never_reported(step):
    """OFF is always supported and therefore never reported."""
    machine = _spindle_machine([CoolantMode.FLOOD])
    assert step.get_unsupported_coolant_methods(machine) == ()


def test_unsupported_coolant_supported_method(step):
    """A method supported by the head is not reported."""
    machine = _spindle_machine([CoolantMode.FLOOD])
    step.set_coolant_method(CoolantMode.FLOOD)
    assert step.get_unsupported_coolant_methods(machine) == ()


def test_unsupported_coolant_reports_missing_method(step):
    """A method the head does not support is reported."""
    machine = _spindle_machine([CoolantMode.FLOOD])
    step.set_coolant_method(CoolantMode.MIST)
    assert step.get_unsupported_coolant_methods(machine) == (CoolantMode.MIST,)


def test_unsupported_coolant_with_laser_head(step):
    """A laser head supports no coolant methods; nothing is reported."""
    machine = MagicMock(heads=[LaserHead()])
    step.set_coolant_method(CoolantMode.FLOOD)
    assert step.get_unsupported_coolant_methods(machine) == ()


def test_unsupported_coolant_with_no_heads(step):
    """With no heads the step cannot run; nothing is reported."""
    step.set_coolant_method(CoolantMode.FLOOD)
    assert step.get_unsupported_coolant_methods(MagicMock(heads=[])) == ()


def test_create_initial_ops_stamps_coolant(step):
    """create_initial_ops sets the coolant state on the ops."""
    step.set_coolant_method(CoolantMode.MIST)
    ops = step.create_initial_ops()
    assert len(ops) == 1
    assert ops.coolant(0) == CoolantMode.MIST


def test_create_initial_ops_omits_off_coolant(step):
    """OFF coolant produces no coolant op."""
    ops = step.create_initial_ops()
    assert len(ops) == 0


def test_check_skips_travel_speed_without_support(sync_machine):
    """No travel speed warning when the machine's dialect cannot emit it."""
    sync_machine.max_cut_speed = 1000
    sync_machine.max_travel_speed = 5000
    step = Step(typelabel="Test")
    step.cut_speed = 500
    step.travel_speed = 99999

    assert not sync_machine.supports_travel_speed()
    assert step.check(sync_machine) == []


def test_check_warns_on_travel_speed_with_support(sync_machine):
    """Travel speed above the machine's max warns when supported."""
    sync_machine.set_dialect_uid("smoothieware")
    sync_machine.max_cut_speed = 1000
    sync_machine.max_travel_speed = 5000
    step = Step(typelabel="Test")
    step.cut_speed = 500
    step.travel_speed = 99999

    assert sync_machine.supports_travel_speed()
    warnings = step.check(sync_machine)
    assert any("Travel speed" in w for w in warnings)

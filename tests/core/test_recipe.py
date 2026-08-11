"""Tests for the Recipe class."""

from unittest.mock import Mock

import pytest

from rayforge.core.doc import Doc
from rayforge.core.recipe import Recipe
from rayforge.core.step import Step
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset


@pytest.fixture
def mock_machine_a() -> Mock:
    """Provides a mock machine with ID 'machine-a'."""
    machine = Mock()
    machine.id = "machine-a"
    head1 = Mock()
    head1.uid = "laser-1"
    head2 = Mock()
    head2.uid = "laser-2"
    machine.heads = [head1, head2]
    return machine


@pytest.fixture
def mock_machine_b() -> Mock:
    """Provides a mock machine with ID 'machine-b'."""
    machine = Mock()
    machine.id = "machine-b"
    head3 = Mock()
    head3.uid = "laser-3"
    machine.heads = [head3]
    return machine


@pytest.fixture
def stock_item_factory():
    """A factory to create real, correctly structured StockItem instances."""

    def _create(
        material_uid: str | None, thickness: float | None
    ) -> StockItem:
        doc = Doc()
        asset = StockAsset()
        asset.material_uid = material_uid
        asset.thickness = thickness
        doc.add_asset(asset)
        item = StockItem(stock_asset_uid=asset.uid)
        doc.add_child(item)
        return item

    return _create


class TestRecipe:
    """Test cases for the Recipe data class."""

    @pytest.fixture
    def sample_recipe(self) -> Recipe:
        """Provides a sample recipe for testing."""
        return Recipe(
            uid="recipe-001",
            name="Cut 6mm Plywood",
            description="A recipe for cutting 6mm plywood",
            target_step_types=["ContourStep"],
            target_machine_id="machine-a",
            material_uid="plywood-6mm",
            min_thickness_mm=5.5,
            max_thickness_mm=6.5,
            settings={
                "power": 0.9,
                "cut_speed": 500,
                "selected_head_uid": "laser-1",
            },
        )

    @pytest.fixture
    def generic_recipe(self) -> Recipe:
        """Provides a generic recipe with no specific criteria."""
        return Recipe(
            uid="recipe-generic",
            name="Generic Cut",
            settings={"power": 1.0, "cut_speed": 200},
        )

    def test_recipe_creation(self, sample_recipe: Recipe):
        """Test creating a Recipe with basic properties."""
        assert sample_recipe.uid == "recipe-001"
        assert sample_recipe.name == "Cut 6mm Plywood"
        assert sample_recipe.material_uid == "plywood-6mm"
        assert sample_recipe.target_step_types == ["ContourStep"]
        assert sample_recipe.target_machine_id == "machine-a"
        assert sample_recipe.settings["power"] == 0.9
        assert sample_recipe.settings["selected_head_uid"] == "laser-1"

    def test_recipe_to_dict(self, sample_recipe: Recipe):
        """Test serializing a Recipe to a dictionary."""
        data = sample_recipe.to_dict()

        assert data["uid"] == "recipe-001"
        assert data["name"] == "Cut 6mm Plywood"
        assert data["material_uid"] == "plywood-6mm"
        assert data["target_step_types"] == ["ContourStep"]
        assert data["target_machine_id"] == "machine-a"
        assert data["min_thickness_mm"] == 5.5
        assert data["max_thickness_mm"] == 6.5
        assert len(data["settings"]) == 3
        assert data["settings"]["power"] == 0.9

    def test_recipe_from_dict(self, sample_recipe: Recipe):
        """Test deserializing a Recipe from a dictionary."""
        data = sample_recipe.to_dict()
        new_recipe = Recipe.from_dict(data)

        assert new_recipe.uid == sample_recipe.uid
        assert new_recipe.name == sample_recipe.name
        assert new_recipe.material_uid == sample_recipe.material_uid
        assert new_recipe.target_step_types == ["ContourStep"]
        assert new_recipe.target_machine_id == "machine-a"
        assert new_recipe.settings["power"] == 0.9

    def test_recipe_from_dict_minimal(self):
        """Test deserializing from a minimal dictionary."""
        data = {"name": "Minimal Recipe"}
        recipe = Recipe.from_dict(data)

        assert recipe.name == "Minimal Recipe"
        assert recipe.uid is not None
        assert recipe.material_uid is None
        assert recipe.target_machine_id is None
        assert recipe.target_step_types == []
        assert recipe.settings == {}

    def test_recipe_from_dict_migrates_legacy_head_key(self):
        """Old recipe files keyed head selection as "selected_laser_uid"."""
        data = {
            "name": "Legacy Recipe",
            "settings": {"power": 0.9, "selected_laser_uid": "laser-1"},
        }

        recipe = Recipe.from_dict(data)

        assert recipe.settings["selected_head_uid"] == "laser-1"
        assert "selected_laser_uid" not in recipe.settings

    def test_recipe_from_dict_migrates_legacy_step_type(self):
        """Legacy target_step_type (single) migrates to a one-element list."""
        data = {
            "name": "Legacy Step Type",
            "target_step_type": "ContourStep",
        }
        recipe = Recipe.from_dict(data)
        assert recipe.target_step_types == ["ContourStep"]

    def test_recipe_from_dict_migrates_legacy_capability(self):
        """Legacy target_capability_name expands to its step types."""
        data = {
            "name": "Legacy Capability",
            "target_capability_name": "ENGRAVE",
        }
        recipe = Recipe.from_dict(data)
        assert recipe.target_step_types == ["EngraveStep"]

    def test_recipe_from_dict_unmigratable_capability_warns(self):
        """An unknown legacy capability yields an empty list (no crash)."""
        data = {
            "name": "Unknown Capability",
            "target_capability_name": "BOGUS",
        }
        recipe = Recipe.from_dict(data)
        assert recipe.target_step_types == []

    def test_get_specificity_score(
        self, sample_recipe: Recipe, generic_recipe: Recipe
    ):
        """Test the specificity scoring."""
        # Machine, head, material, thickness all specific; step_type len 1.
        sample_score = sample_recipe.get_specificity_score()
        assert sample_score[:4] == (0, 0, 0, 0)
        assert sample_score[4] == 1  # len(target_step_types)

        # Generic all -> (1, 1, 1, 1, large)
        generic_score = generic_recipe.get_specificity_score()
        assert generic_score[:4] == (1, 1, 1, 1)
        assert generic_score[4] > 1000

        # Specific machine only
        machine_only = Recipe(target_machine_id="test")
        assert machine_only.get_specificity_score()[:4] == (0, 1, 1, 1)

        # Specific head only
        head_only = Recipe(settings={"selected_head_uid": "laser-x"})
        assert head_only.get_specificity_score()[:4] == (1, 0, 1, 1)

    def test_fewer_step_types_more_specific(self):
        """A single-target recipe outranks a multi-target one."""
        single = Recipe(target_step_types=["ContourStep"])
        multi = Recipe(target_step_types=["ContourStep", "FrameStep"])
        assert single.get_specificity_score() < multi.get_specificity_score()

        generic = Recipe()
        assert multi.get_specificity_score() < generic.get_specificity_score()

    # --- MATCHING LOGIC TESTS ---

    def test_matches_perfect(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test a perfect match for a specific recipe."""
        stock = stock_item_factory("plywood-6mm", 6.0)
        assert (
            sample_recipe.matches(
                [stock], mock_machine_a, step_type="ContourStep"
            )
            is True
        )

    def test_matches_generic(
        self, generic_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test that a generic recipe matches any context."""
        stock = stock_item_factory("any-material", 10.0)
        assert generic_recipe.matches([stock], mock_machine_a) is True
        assert generic_recipe.matches([], mock_machine_a) is True
        assert generic_recipe.matches([stock], None) is True

    def test_matches_machine_fail(
        self, sample_recipe: Recipe, mock_machine_b: Mock, stock_item_factory
    ):
        """Test match failure due to incorrect machine."""
        stock = stock_item_factory("plywood-6mm", 6.0)
        assert (
            sample_recipe.matches(
                [stock], mock_machine_b, step_type="ContourStep"
            )
            is False
        )

    def test_matches_head_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test match failure due to head not on machine."""
        sample_recipe.settings["selected_head_uid"] = "non-existent-laser"
        stock = stock_item_factory("plywood-6mm", 6.0)
        assert (
            sample_recipe.matches(
                [stock], mock_machine_a, step_type="ContourStep"
            )
            is False
        )

    def test_matches_no_machine_provided_fail(
        self, sample_recipe: Recipe, stock_item_factory
    ):
        """Test match failure when recipe requires machine but none given."""
        stock = stock_item_factory("plywood-6mm", 6.0)
        assert (
            sample_recipe.matches([stock], None, step_type="ContourStep")
            is False
        )

    def test_matches_step_type_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """A step-type-scoped recipe rejects a different step type."""
        stock = stock_item_factory("plywood-6mm", 6.0)
        assert (
            sample_recipe.matches(
                [stock], mock_machine_a, step_type="EngraveStep"
            )
            is False
        )

    def test_matches_step_type_none_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """A step-type-scoped recipe cannot match without a step type."""
        stock = stock_item_factory("plywood-6mm", 6.0)
        assert sample_recipe.matches([stock], mock_machine_a) is False

    def test_matches_multi_step_types(
        self, mock_machine_a: Mock, stock_item_factory
    ):
        """A recipe targeting several step types matches any of them."""
        stock = stock_item_factory("plywood-6mm", 6.0)
        recipe = Recipe(
            target_step_types=["ContourStep", "FrameStep"],
            settings={"power": 0.9},
        )
        assert (
            recipe.matches([stock], mock_machine_a, step_type="ContourStep")
            is True
        )
        assert (
            recipe.matches([stock], mock_machine_a, step_type="FrameStep")
            is True
        )
        assert (
            recipe.matches([stock], mock_machine_a, step_type="EngraveStep")
            is False
        )

    def test_matches_generic_ignores_step_type(
        self, mock_machine_a: Mock, stock_item_factory
    ):
        """A generic recipe (no step types) ignores the step_type arg."""
        stock = stock_item_factory("plywood-6mm", 6.0)
        recipe = Recipe(settings={"power": 0.9})
        assert (
            recipe.matches([stock], mock_machine_a, step_type="ContourStep")
            is True
        )
        assert (
            recipe.matches([stock], mock_machine_a, step_type="EngraveStep")
            is True
        )

    def test_matches_material_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test match failure due to material mismatch."""
        stock = stock_item_factory("wrong-material", 6.0)
        assert (
            sample_recipe.matches(
                [stock], mock_machine_a, step_type="ContourStep"
            )
            is False
        )

    def test_matches_thickness_fail_too_thin(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test match failure due to thickness being too low."""
        stock = stock_item_factory("plywood-6mm", 3.0)
        assert (
            sample_recipe.matches(
                [stock], mock_machine_a, step_type="ContourStep"
            )
            is False
        )

    def test_matches_thickness_fail_too_thick(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test match failure due to thickness being too high."""
        stock = stock_item_factory("plywood-6mm", 10.0)
        assert (
            sample_recipe.matches(
                [stock], mock_machine_a, step_type="ContourStep"
            )
            is False
        )

    def test_matches_no_stock_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock
    ):
        """Test that a specific recipe fails to match with no stock."""
        assert (
            sample_recipe.matches([], mock_machine_a, step_type="ContourStep")
            is False
        )

    def test_matches_no_thickness_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test that a thickness-specific recipe fails when stock has none."""
        stock = stock_item_factory("plywood-6mm", None)
        assert (
            sample_recipe.matches(
                [stock], mock_machine_a, step_type="ContourStep"
            )
            is False
        )

    def test_matches_material_only_recipe(self, stock_item_factory):
        """Test a recipe that only specifies material."""
        recipe = Recipe(material_uid="mdf-3mm")
        stock_match = stock_item_factory("mdf-3mm", 3.0)
        stock_fail = stock_item_factory("acrylic-3mm", 3.0)
        assert recipe.matches([stock_match]) is True
        assert recipe.matches([stock_fail]) is False
        assert recipe.matches([]) is False

    def test_matches_thickness_only_recipe(self, stock_item_factory):
        """Test a recipe that only specifies thickness."""
        recipe = Recipe(min_thickness_mm=2.8, max_thickness_mm=3.2)
        stock_match = stock_item_factory("any", 3.0)
        stock_fail = stock_item_factory("any", 4.0)
        stock_no_thickness = stock_item_factory("any", None)
        assert recipe.matches([stock_match]) is True
        assert recipe.matches([stock_fail]) is False
        assert recipe.matches([stock_no_thickness]) is False

    def test_matches_step_settings(self):
        """Tests the comparison between a recipe's settings and a Step."""
        recipe = Recipe(
            settings={
                "power": 0.8,
                "cut_speed": 1000,
                "offset_mm": 0.15,
                "air_assist": True,
            }
        )

        # 1. Perfect match (and step has extra properties which are ignored)
        mock_step = Mock()
        mock_step.power = 0.8
        mock_step.cut_speed = 1000
        mock_step.offset_mm = 0.15
        mock_step.air_assist = True
        mock_step.extra_property = "should_be_ignored"
        assert recipe.matches_step_settings(mock_step) is True

        # 2. Float match within tolerance
        mock_step.power = 0.80000001
        assert recipe.matches_step_settings(mock_step) is True

        # 3. Mismatch (integer value)
        mock_step.power = 0.8  # reset
        mock_step.cut_speed = 1001
        assert recipe.matches_step_settings(mock_step) is False

        # 4. Mismatch (float value outside tolerance)
        mock_step.cut_speed = 1000  # reset
        mock_step.power = 0.81
        assert recipe.matches_step_settings(mock_step) is False

        # 5. Mismatch (boolean value)
        mock_step.power = 0.8  # reset
        mock_step.air_assist = False
        assert recipe.matches_step_settings(mock_step) is False

        # 6. Mismatch (step is missing an attribute)
        mock_step_missing = Mock(spec=["power", "offset_mm", "air_assist"])
        mock_step_missing.power = 0.8
        # cut_speed is missing
        assert recipe.matches_step_settings(mock_step_missing) is False

        # 7. Mismatch (type difference)
        mock_step_bad_type = Mock()
        mock_step_bad_type.power = 0.8
        mock_step_bad_type.cut_speed = "1000"  # string vs int
        mock_step_bad_type.offset_mm = 0.15
        mock_step_bad_type.air_assist = True
        assert recipe.matches_step_settings(mock_step_bad_type) is False

    # --- TRANSFORMER SETTINGS TESTS ---

    def _step_with_transformers(
        self,
        per_wp: list[dict] | None = None,
        per_step: list[dict] | None = None,
    ) -> Step:
        """A plain Step carrying the given transformer dicts."""
        step = Step(typelabel="Test", name="Test")
        step.per_workpiece_transformers_dicts = list(per_wp or [])
        step.per_step_transformers_dicts = list(per_step or [])
        return step

    def test_transformer_dicts_round_trip(self):
        """transformer_dicts survives to_dict/from_dict."""
        recipe = Recipe(
            name="With Transformers",
            transformer_dicts=[
                {
                    "name": "CropTransformer",
                    "enabled": True,
                    "recipe_apply": True,
                    "offset": 1.5,
                },
                {
                    "name": "Optimize",
                    "enabled": False,
                    "recipe_apply": False,
                },
            ],
        )
        data = recipe.to_dict()
        assert len(data["transformer_dicts"]) == 2

        restored = Recipe.from_dict(data)
        assert restored.transformer_dicts == recipe.transformer_dicts

    def test_from_dict_missing_transformer_dicts_defaults_empty(self):
        """Old recipe files without transformer_dicts load as empty."""
        recipe = Recipe.from_dict({"name": "Old Recipe"})
        assert recipe.transformer_dicts == []

    def test_from_dict_non_list_transformer_dicts_ignored(self):
        """A malformed transformer_dicts value is treated as empty."""
        recipe = Recipe.from_dict({"name": "Bad", "transformer_dicts": "nope"})
        assert recipe.transformer_dicts == []

    def test_matches_step_transformers(self):
        """matches_step_transformers compares only recipe_apply entries."""
        recipe = Recipe(
            transformer_dicts=[
                {
                    "name": "CropTransformer",
                    "enabled": True,
                    "recipe_apply": True,
                    "offset": 0.5,
                }
            ]
        )
        step = self._step_with_transformers(
            per_wp=[
                {
                    "name": "CropTransformer",
                    "enabled": True,
                    "offset": 0.5,
                }
            ]
        )
        assert recipe.matches_step_transformers(step) is True

    def test_matches_step_transformers_tolerance(self):
        """Float params are compared with math.isclose tolerance."""
        recipe = Recipe(
            transformer_dicts=[
                {
                    "name": "CropTransformer",
                    "recipe_apply": True,
                    "offset": 0.5,
                }
            ]
        )
        step = self._step_with_transformers(
            per_wp=[
                {
                    "name": "CropTransformer",
                    "enabled": True,
                    "offset": 0.50000001,
                }
            ]
        )
        assert recipe.matches_step_transformers(step) is True

        step.per_workpiece_transformers_dicts[0]["offset"] = 0.6
        assert recipe.matches_step_transformers(step) is False

    def test_matches_step_transformers_missing_transformer(self):
        """A recipe_apply entry with no matching step transformer fails."""
        recipe = Recipe(
            transformer_dicts=[
                {
                    "name": "MissingTransformer",
                    "recipe_apply": True,
                    "enabled": True,
                }
            ]
        )
        step = self._step_with_transformers(
            per_wp=[{"name": "CropTransformer", "enabled": True}]
        )
        assert recipe.matches_step_transformers(step) is False

    def test_matches_step_transformers_skips_leave_unchanged(self):
        """recipe_apply=False entries are ignored entirely."""
        recipe = Recipe(
            transformer_dicts=[
                {
                    "name": "MissingTransformer",
                    "recipe_apply": False,
                    "enabled": True,
                }
            ]
        )
        step = self._step_with_transformers(
            per_wp=[{"name": "CropTransformer", "enabled": True}]
        )
        assert recipe.matches_step_transformers(step) is True

    def test_matches_step_transformers_disabled_param(self):
        """A disabled transformer is matched by its enabled flag."""
        recipe = Recipe(
            transformer_dicts=[
                {
                    "name": "Optimize",
                    "recipe_apply": True,
                    "enabled": False,
                }
            ]
        )
        step = self._step_with_transformers(
            per_step=[{"name": "Optimize", "enabled": False}]
        )
        assert recipe.matches_step_transformers(step) is True

        step.per_step_transformers_dicts[0]["enabled"] = True
        assert recipe.matches_step_transformers(step) is False

    def test_matches_step_transformers_per_step_match(self):
        """A transformer found in per_step dicts matches too."""
        recipe = Recipe(
            transformer_dicts=[
                {
                    "name": "Optimize",
                    "recipe_apply": True,
                    "enabled": True,
                }
            ]
        )
        step = self._step_with_transformers(
            per_step=[{"name": "Optimize", "enabled": True}]
        )
        assert recipe.matches_step_transformers(step) is True

    def test_matches_step_transformers_empty_recipe(self):
        """A recipe with no transformer_dicts always matches."""
        recipe = Recipe()
        step = self._step_with_transformers(
            per_wp=[{"name": "CropTransformer", "enabled": True}]
        )
        assert recipe.matches_step_transformers(step) is True

    # --- STEP TYPE TARGETING TESTS ---

    def test_target_step_types_round_trip(self):
        """target_step_types survives to_dict/from_dict."""
        recipe = Recipe(
            name="Contour-only",
            target_step_types=["ContourStep"],
            settings={"power": 0.8},
        )
        data = recipe.to_dict()
        assert data["target_step_types"] == ["ContourStep"]

        restored = Recipe.from_dict(data)
        assert restored.target_step_types == ["ContourStep"]

    def test_step_type_more_specific_than_generic(
        self, mock_machine_a: Mock, stock_item_factory
    ):
        """A step-type-scoped recipe outranks a generic one for sorting."""
        stock = stock_item_factory("plywood-6mm", 6.0)
        scoped = Recipe(
            target_step_types=["ContourStep"],
            settings={"power": 0.9},
        )
        generic = Recipe(settings={"power": 0.9})
        ctx = ([stock], mock_machine_a, "ContourStep")
        assert scoped.matches(*ctx) is True
        assert generic.matches(*ctx) is True
        # The scoped recipe has a lower (more specific) score.
        assert scoped.get_specificity_score() < generic.get_specificity_score()

    def test_recipe_forward_compatibility_with_extra_fields(self):
        """from_dict preserves unknown fields and to_dict re-serializes."""
        recipe_dict = {
            "uid": "recipe-forward-456",
            "name": "Future Recipe",
            "description": "A future recipe",
            "target_step_types": ["ContourStep"],
            "target_machine_id": None,
            "material_uid": None,
            "min_thickness_mm": None,
            "max_thickness_mm": None,
            "settings": {},
            "future_field_string": "some value",
            "future_field_number": 42,
            "future_field_dict": {"nested": "data"},
        }

        recipe = Recipe.from_dict(recipe_dict)

        # Verify extra fields are stored
        assert recipe.extra["future_field_string"] == "some value"
        assert recipe.extra["future_field_number"] == 42
        assert recipe.extra["future_field_dict"] == {"nested": "data"}

        # Verify extra fields are re-serialized
        data = recipe.to_dict()
        assert data["future_field_string"] == "some value"
        assert data["future_field_number"] == 42
        assert data["future_field_dict"] == {"nested": "data"}

    def test_recipe_backward_compatibility_with_missing_optional_fields(self):
        """from_dict handles missing optional fields gracefully."""
        minimal_dict = {"name": "Old Recipe"}

        recipe = Recipe.from_dict(minimal_dict)

        assert recipe.name == "Old Recipe"
        assert recipe.description == ""
        assert recipe.target_step_types == []
        assert recipe.target_machine_id is None
        assert recipe.material_uid is None
        assert recipe.min_thickness_mm is None
        assert recipe.max_thickness_mm is None
        assert recipe.settings == {}
        assert recipe.extra == {}

    # --- MULTIPLE STOCK ITEMS TESTS ---

    def test_matches_multiple_stocks_any_match(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test that recipe matches if it ANY of multiple stock items match."""
        stock1 = stock_item_factory("plywood-6mm", 6.0)
        stock2 = stock_item_factory("other-material", 3.0)
        stock3 = stock_item_factory("plywood-6mm", 5.0)  # Different thickness

        ctx = (mock_machine_a, "ContourStep")
        assert sample_recipe.matches([stock1], *ctx) is True
        assert sample_recipe.matches([stock2, stock1], *ctx) is True
        # Should not match with list containing no matching stock
        assert sample_recipe.matches([stock2], *ctx) is False
        assert sample_recipe.matches([stock2, stock3], *ctx) is False

    def test_matches_multiple_stocks_empty_list(
        self, sample_recipe: Recipe, mock_machine_a: Mock
    ):
        """A constrained recipe doesn't match an empty stock list."""
        assert (
            sample_recipe.matches([], mock_machine_a, step_type="ContourStep")
            is False
        )

    def test_matches_generic_with_multiple_stocks(
        self, generic_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """A generic recipe matches even with an empty stock list."""
        stock = stock_item_factory("any-material", 10.0)
        assert generic_recipe.matches([], mock_machine_a) is True
        assert generic_recipe.matches([stock], mock_machine_a) is True

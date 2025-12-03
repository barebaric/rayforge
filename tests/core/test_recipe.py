"""Tests for the Recipe class."""

from unittest.mock import Mock
from typing import Optional
import pytest
from rayforge.core.doc import Doc
from rayforge.core.recipe import Recipe
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.capability import CUT, ENGRAVE


class TestRecipe:
    """Test cases for the Recipe data class."""

    @pytest.fixture
    def sample_recipe(self) -> Recipe:
        """Provides a sample recipe for testing."""
        return Recipe(
            uid="recipe-001",
            name="Cut 6mm Plywood",
            description="A recipe for cutting 6mm plywood",
            target_capability_name=CUT.name,
            target_machine_id="machine-a",
            material_uid="plywood-6mm",
            min_thickness_mm=5.5,
            max_thickness_mm=6.5,
            settings={
                "power": 0.9,
                "cut_speed": 500,
                "selected_laser_uid": "laser-1",
            },
        )

    @pytest.fixture
    def generic_recipe(self) -> Recipe:
        """Provides a generic recipe with no specific criteria."""
        return Recipe(
            uid="recipe-generic",
            name="Generic Cut",
            target_capability_name=CUT.name,
            settings={"power": 1.0, "cut_speed": 200},
        )

    @pytest.fixture
    def mock_machine_a(self) -> Mock:
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
    def mock_machine_b(self) -> Mock:
        """Provides a mock machine with ID 'machine-b'."""
        machine = Mock()
        machine.id = "machine-b"
        head3 = Mock()
        head3.uid = "laser-3"
        machine.heads = [head3]
        return machine

    @pytest.fixture
    def stock_item_factory(self):
        """
        A factory to create real, correctly structured StockItem instances.
        """

        def _create(
            material_uid: Optional[str], thickness: Optional[float]
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

    def test_recipe_creation(self, sample_recipe: Recipe):
        """Test creating a Recipe with basic properties."""
        assert sample_recipe.uid == "recipe-001"
        assert sample_recipe.name == "Cut 6mm Plywood"
        assert sample_recipe.material_uid == "plywood-6mm"
        assert sample_recipe.target_capability_name == "CUT"
        assert sample_recipe.target_machine_id == "machine-a"
        assert sample_recipe.capability is CUT
        assert sample_recipe.settings["power"] == 0.9
        assert sample_recipe.settings["selected_laser_uid"] == "laser-1"

    def test_recipe_to_dict(self, sample_recipe: Recipe):
        """Test serializing a Recipe to a dictionary."""
        data = sample_recipe.to_dict()

        assert data["uid"] == "recipe-001"
        assert data["name"] == "Cut 6mm Plywood"
        assert data["material_uid"] == "plywood-6mm"
        assert data["target_capability_name"] == "CUT"
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
        assert new_recipe.target_capability_name == CUT.name
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
        assert recipe.target_capability_name == CUT.name  # Default
        assert recipe.settings == {}

    def test_get_specificity_score(
        self, sample_recipe: Recipe, generic_recipe: Recipe
    ):
        """Test the specificity scoring."""
        # Machine, laser, material, thickness -> (0, 0, 0, 0)
        assert sample_recipe.get_specificity_score() == (0, 0, 0, 0)

        # Generic all -> (1, 1, 1, 1)
        assert generic_recipe.get_specificity_score() == (1, 1, 1, 1)

        # Specific machine only
        machine_only = Recipe(target_machine_id="test")
        assert machine_only.get_specificity_score() == (0, 1, 1, 1)

        # Specific laser only
        laser_only = Recipe(settings={"selected_laser_uid": "laser-x"})
        assert laser_only.get_specificity_score() == (1, 0, 1, 1)

    # --- MATCHING LOGIC TESTS ---

    def test_matches_perfect(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test a perfect match for a specific recipe."""
        stock = stock_item_factory("plywood-6mm", 6.0)
        assert sample_recipe.matches(stock, {CUT}, mock_machine_a) is True

    def test_matches_generic(
        self, generic_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test that a generic recipe matches any context."""
        stock = stock_item_factory("any-material", 10.0)
        assert generic_recipe.matches(stock, {CUT}, mock_machine_a) is True
        assert generic_recipe.matches(None, {CUT}, mock_machine_a) is True
        assert generic_recipe.matches(stock, {CUT}, None) is True

    def test_matches_machine_fail(
        self, sample_recipe: Recipe, mock_machine_b: Mock, stock_item_factory
    ):
        """Test match failure due to incorrect machine."""
        stock = stock_item_factory("plywood-6mm", 6.0)
        assert sample_recipe.matches(stock, {CUT}, mock_machine_b) is False

    def test_matches_laser_head_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test match failure due to laser head not on machine."""
        sample_recipe.settings["selected_laser_uid"] = "non-existent-laser"
        stock = stock_item_factory("plywood-6mm", 6.0)
        assert sample_recipe.matches(stock, {CUT}, mock_machine_a) is False

    def test_matches_no_machine_provided_fail(
        self, sample_recipe: Recipe, stock_item_factory
    ):
        """
        Test match failure when recipe requires machine but none is given.
        """
        stock = stock_item_factory("plywood-6mm", 6.0)
        assert sample_recipe.matches(stock, {CUT}, None) is False

    def test_matches_capability_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test match failure due to incorrect capability."""
        stock = stock_item_factory("plywood-6mm", 6.0)
        assert sample_recipe.matches(stock, {ENGRAVE}, mock_machine_a) is False

    def test_matches_material_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test match failure due to material mismatch."""
        stock = stock_item_factory("wrong-material", 6.0)
        assert sample_recipe.matches(stock, {CUT}, mock_machine_a) is False

    def test_matches_thickness_fail_too_thin(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test match failure due to thickness being too low."""
        stock = stock_item_factory("plywood-6mm", 3.0)
        assert sample_recipe.matches(stock, {CUT}, mock_machine_a) is False

    def test_matches_thickness_fail_too_thick(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """Test match failure due to thickness being too high."""
        stock = stock_item_factory("plywood-6mm", 10.0)
        assert sample_recipe.matches(stock, {CUT}, mock_machine_a) is False

    def test_matches_no_stock_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock
    ):
        """
        Test that a specific recipe fails to match when no stock is provided.
        """
        assert sample_recipe.matches(None, {CUT}, mock_machine_a) is False

    def test_matches_no_thickness_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock, stock_item_factory
    ):
        """
        Test that a thickness-specific recipe fails when stock has no
        thickness.
        """
        stock = stock_item_factory("plywood-6mm", None)
        assert sample_recipe.matches(stock, {CUT}, mock_machine_a) is False

    def test_matches_material_only_recipe(self, stock_item_factory):
        """Test a recipe that only specifies material."""
        recipe = Recipe(material_uid="mdf-3mm")
        stock_match = stock_item_factory("mdf-3mm", 3.0)
        stock_fail = stock_item_factory("acrylic-3mm", 3.0)
        assert recipe.matches(stock_match) is True
        assert recipe.matches(stock_fail) is False
        assert recipe.matches(None) is False

    def test_matches_thickness_only_recipe(self, stock_item_factory):
        """Test a recipe that only specifies thickness."""
        recipe = Recipe(min_thickness_mm=2.8, max_thickness_mm=3.2)
        stock_match = stock_item_factory("any", 3.0)
        stock_fail = stock_item_factory("any", 4.0)
        stock_no_thickness = stock_item_factory("any", None)
        assert recipe.matches(stock_match) is True
        assert recipe.matches(stock_fail) is False
        assert recipe.matches(stock_no_thickness) is False

    def test_matches_step_settings(self):
        """Tests the comparison between a recipe's settings and a Step."""
        recipe = Recipe(
            settings={
                "power": 0.8,
                "cut_speed": 1000,
                "kerf_mm": 0.15,
                "air_assist": True,
            }
        )

        # 1. Perfect match (and step has extra properties which are ignored)
        mock_step = Mock()
        mock_step.power = 0.8
        mock_step.cut_speed = 1000
        mock_step.kerf_mm = 0.15
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
        mock_step_missing = Mock(spec=["power", "kerf_mm", "air_assist"])
        mock_step_missing.power = 0.8
        # cut_speed is missing
        assert recipe.matches_step_settings(mock_step_missing) is False

        # 7. Mismatch (type difference)
        mock_step_bad_type = Mock()
        mock_step_bad_type.power = 0.8
        mock_step_bad_type.cut_speed = "1000"  # string vs int
        mock_step_bad_type.kerf_mm = 0.15
        mock_step_bad_type.air_assist = True
        assert recipe.matches_step_settings(mock_step_bad_type) is False

"""Tests for the Recipe class."""

from unittest.mock import Mock
import pytest
from rayforge.core.recipe import Recipe
from rayforge.core.stock import StockItem
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
        self, sample_recipe: Recipe, mock_machine_a: Mock
    ):
        """Test a perfect match for a specific recipe."""
        stock = StockItem()
        stock.material_uid = "plywood-6mm"
        stock.thickness = 6.0
        assert sample_recipe.matches(stock, CUT, mock_machine_a) is True

    def test_matches_generic(
        self, generic_recipe: Recipe, mock_machine_a: Mock
    ):
        """Test that a generic recipe matches any context."""
        stock = StockItem()
        stock.material_uid = "any-material"
        stock.thickness = 10.0
        assert generic_recipe.matches(stock, CUT, mock_machine_a) is True
        assert generic_recipe.matches(None, CUT, mock_machine_a) is True
        assert generic_recipe.matches(stock, CUT, None) is True

    def test_matches_machine_fail(
        self, sample_recipe: Recipe, mock_machine_b: Mock
    ):
        """Test match failure due to incorrect machine."""
        stock = StockItem()
        stock.material_uid = "plywood-6mm"
        stock.thickness = 6.0
        assert sample_recipe.matches(stock, CUT, mock_machine_b) is False

    def test_matches_laser_head_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock
    ):
        """Test match failure due to laser head not on machine."""
        # Modify recipe to require a laser that machine_a does not have
        sample_recipe.settings["selected_laser_uid"] = "non-existent-laser"
        stock = StockItem()
        stock.material_uid = "plywood-6mm"
        stock.thickness = 6.0
        assert sample_recipe.matches(stock, CUT, mock_machine_a) is False

    def test_matches_no_machine_provided_fail(self, sample_recipe: Recipe):
        """
        Test match failure when recipe requires machine but none is given.
        """
        stock = StockItem()
        stock.material_uid = "plywood-6mm"
        stock.thickness = 6.0
        assert sample_recipe.matches(stock, CUT, None) is False

    def test_matches_capability_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock
    ):
        """Test match failure due to incorrect capability."""
        stock = StockItem()
        stock.material_uid = "plywood-6mm"
        stock.thickness = 6.0
        assert sample_recipe.matches(stock, ENGRAVE, mock_machine_a) is False

    def test_matches_material_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock
    ):
        """Test match failure due to material mismatch."""
        stock = StockItem()
        stock.material_uid = "wrong-material"
        stock.thickness = 6.0
        assert sample_recipe.matches(stock, CUT, mock_machine_a) is False

    def test_matches_thickness_fail_too_thin(
        self, sample_recipe: Recipe, mock_machine_a: Mock
    ):
        """Test match failure due to thickness being too low."""
        stock = StockItem()
        stock.material_uid = "plywood-6mm"
        stock.thickness = 3.0
        assert sample_recipe.matches(stock, CUT, mock_machine_a) is False

    def test_matches_thickness_fail_too_thick(
        self, sample_recipe: Recipe, mock_machine_a: Mock
    ):
        """Test match failure due to thickness being too high."""
        stock = StockItem()
        stock.material_uid = "plywood-6mm"
        stock.thickness = 10.0
        assert sample_recipe.matches(stock, CUT, mock_machine_a) is False

    def test_matches_no_stock_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock
    ):
        """
        Test that a specific recipe fails to match when no stock is provided.
        """
        assert sample_recipe.matches(None, CUT, mock_machine_a) is False

    def test_matches_no_thickness_fail(
        self, sample_recipe: Recipe, mock_machine_a: Mock
    ):
        """
        Test that a thickness-specific recipe fails when stock has no
        thickness.
        """
        stock = StockItem()
        stock.material_uid = "plywood-6mm"
        stock.thickness = None
        assert sample_recipe.matches(stock, CUT, mock_machine_a) is False

    def test_matches_material_only_recipe(self):
        """Test a recipe that only specifies material."""
        recipe = Recipe(material_uid="mdf-3mm")
        stock_match = StockItem()
        stock_match.material_uid = "mdf-3mm"
        stock_match.thickness = 3.0
        stock_fail = StockItem()
        stock_fail.material_uid = "acrylic-3mm"
        stock_fail.thickness = 3.0
        assert recipe.matches(stock_match) is True
        assert recipe.matches(stock_fail) is False
        assert recipe.matches(None) is False

    def test_matches_thickness_only_recipe(self):
        """Test a recipe that only specifies thickness."""
        recipe = Recipe(min_thickness_mm=2.8, max_thickness_mm=3.2)
        stock_match = StockItem()
        stock_match.material_uid = "any"
        stock_match.thickness = 3.0
        stock_fail = StockItem()
        stock_fail.material_uid = "any"
        stock_fail.thickness = 4.0
        stock_no_thickness = StockItem()
        stock_no_thickness.material_uid = "any"
        stock_no_thickness.thickness = None
        assert recipe.matches(stock_match) is True
        assert recipe.matches(stock_fail) is False
        assert recipe.matches(stock_no_thickness) is False

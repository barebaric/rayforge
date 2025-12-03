"""Tests for the RecipeManager class."""

from unittest.mock import Mock
from typing import Optional
import pytest
import tempfile
import yaml
from pathlib import Path
from rayforge.core.doc import Doc
from rayforge.core.recipe import Recipe
from rayforge.core.recipe_manager import RecipeManager
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.capability import CUT, ENGRAVE
from rayforge.machine.models.machine import Machine


class TestRecipeManager:
    """Test cases for the RecipeManager class."""

    @pytest.fixture
    def recipes_dir(self):
        """Creates a temporary directory for recipe files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def machine_a(self) -> Mock:
        mock = Mock(spec=Machine)
        mock.id = "machine-a"
        mock.name = "Machine A"
        head1 = Mock()
        head1.uid = "laser-1"
        mock.heads = [head1]
        return mock

    @pytest.fixture
    def machine_b(self) -> Mock:
        mock = Mock(spec=Machine)
        mock.id = "machine-b"
        mock.name = "Machine B"
        return mock

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

    def test_manager_creation_and_load_empty(self, recipes_dir: Path):
        """Test creating a RecipeManager for an empty directory."""
        manager = RecipeManager(recipes_dir)
        assert not manager.recipes
        assert manager.get_all_recipes() == []

    def test_load_recipes(self, recipes_dir: Path):
        """Test loading recipes from files."""
        recipe1_data = {
            "uid": "recipe1",
            "name": "Recipe 1",
            "target_capability_name": "CUT",
            "settings": {"power": 0.8},
        }
        recipe2_data = {"uid": "recipe2", "name": "Recipe 2"}
        with open(recipes_dir / "recipe1.yaml", "w") as f:
            yaml.dump(recipe1_data, f)
        with open(recipes_dir / "recipe2.yaml", "w") as f:
            yaml.dump(recipe2_data, f)

        manager = RecipeManager(recipes_dir)
        assert len(manager.recipes) == 2
        assert "recipe1" in manager.recipes
        recipe2 = manager.get_recipe_by_id("recipe2")
        assert recipe2 is not None
        assert recipe2.name == "Recipe 2"

    def test_load_invalid_recipe_file(self, recipes_dir: Path):
        """Test that the manager skips invalid recipe files."""
        with open(recipes_dir / "invalid.yaml", "w") as f:
            f.write("this is not yaml")

        manager = RecipeManager(recipes_dir)
        assert len(manager.recipes) == 0

    def test_save_recipe(self, recipes_dir: Path):
        """Test saving a new recipe to a file."""
        manager = RecipeManager(recipes_dir)
        recipe = Recipe(uid="new-recipe", name="New Recipe")

        manager.save_recipe(recipe)

        recipe_file = recipes_dir / "new-recipe.yaml"
        assert recipe_file.exists()

        with open(recipe_file, "r") as f:
            data = yaml.safe_load(f)
        assert data["name"] == "New Recipe"

    def test_add_recipe(self, recipes_dir: Path):
        """Test adding a recipe, which should also save it."""
        manager = RecipeManager(recipes_dir)
        recipe = Recipe(uid="added-recipe", name="Added Recipe")

        manager.add_recipe(recipe)

        assert "added-recipe" in manager.recipes
        assert (recipes_dir / "added-recipe.yaml").exists()

    def test_delete_recipe(self, recipes_dir: Path):
        """Test deleting a recipe and its corresponding file."""
        recipe_data = {"uid": "to-delete", "name": "To Delete"}
        recipe_file = recipes_dir / "to-delete.yaml"
        with open(recipe_file, "w") as f:
            yaml.dump(recipe_data, f)

        manager = RecipeManager(recipes_dir)
        assert "to-delete" in manager.recipes
        assert recipe_file.exists()

        manager.delete_recipe("to-delete")

        assert "to-delete" not in manager.recipes
        assert not recipe_file.exists()

    # --- find_recipes LOGIC TESTS ---

    @pytest.fixture
    def manager_with_recipes(
        self, recipes_dir: Path, machine_a: Mock
    ) -> RecipeManager:
        """Provides a manager pre-populated with a variety of recipes."""
        manager = RecipeManager(recipes_dir)

        # 1. Most specific: Machine, Laser, Material, Thickness
        manager.add_recipe(
            Recipe(
                uid="machine-a-laser-1-walnut-3mm",
                name="Cut 3mm Walnut on Machine A with Laser 1",
                target_machine_id=machine_a.id,
                material_uid="walnut",
                min_thickness_mm=2.9,
                max_thickness_mm=3.1,
                target_capability_name=CUT.name,
                settings={"power": 0.8, "selected_laser_uid": "laser-1"},
            )
        )

        # 2. Machine, Material, Thickness
        manager.add_recipe(
            Recipe(
                uid="machine-a-walnut-3mm",
                name="Cut 3mm Walnut on Machine A",
                target_machine_id=machine_a.id,
                material_uid="walnut",
                min_thickness_mm=2.9,
                max_thickness_mm=3.1,
                target_capability_name=CUT.name,
                settings={"power": 0.8},
            )
        )

        # 3. Material, Thickness
        manager.add_recipe(
            Recipe(
                uid="walnut-3mm-cut",
                name="Cut 3mm Walnut",
                material_uid="walnut",
                min_thickness_mm=2.9,
                max_thickness_mm=3.1,
                target_capability_name=CUT.name,
                settings={"power": 0.9},
            )
        )

        # 4. Material, Any thickness
        manager.add_recipe(
            Recipe(
                uid="walnut-any-thickness",
                name="Cut Any Walnut",
                material_uid="walnut",
                target_capability_name=CUT.name,
                settings={"power": 0.85},
            )
        )

        # 5. Generic Cut
        manager.add_recipe(
            Recipe(
                uid="generic-cut",
                name="Generic Cut",
                target_capability_name=CUT.name,
                settings={"power": 1.0},
            )
        )

        # 6. Generic Engrave
        manager.add_recipe(
            Recipe(
                uid="generic-engrave",
                name="Generic Engrave",
                target_capability_name=ENGRAVE.name,
                settings={"power": 0.2},
            )
        )
        return manager

    def test_find_recipes_perfect_match(
        self,
        manager_with_recipes: RecipeManager,
        machine_a: Mock,
        stock_item_factory,
    ):
        """
        Test finding recipes with a perfect stock and machine match,
        checking sort order.
        """
        stock = stock_item_factory("walnut", 3.0)
        results = manager_with_recipes.find_recipes(
            stock, capabilities={CUT}, machine=machine_a
        )

        assert len(results) == 5
        # Specificity order: (machine, laser, material, thickness)
        assert results[0].uid == "machine-a-laser-1-walnut-3mm"  # (0,0,0,0)
        assert results[1].uid == "machine-a-walnut-3mm"  # (0,1,0,0)
        assert results[2].uid == "walnut-3mm-cut"  # (1,1,0,0)
        assert results[3].uid == "walnut-any-thickness"  # (1,1,0,1)
        assert results[4].uid == "generic-cut"  # (1,1,1,1)

    def test_find_recipes_different_machine(
        self,
        manager_with_recipes: RecipeManager,
        machine_b: Mock,
        stock_item_factory,
    ):
        """Test that machine-specific recipes are filtered out."""
        stock = stock_item_factory("walnut", 3.0)
        results = manager_with_recipes.find_recipes(
            stock, capabilities={CUT}, machine=machine_b
        )

        assert len(results) == 3
        assert "machine-a-walnut-3mm" not in [r.uid for r in results]
        assert "machine-a-laser-1-walnut-3mm" not in [r.uid for r in results]
        assert results[0].uid == "walnut-3mm-cut"
        assert results[1].uid == "walnut-any-thickness"
        assert results[2].uid == "generic-cut"

    def test_find_recipes_no_machine_provided(
        self, manager_with_recipes: RecipeManager, stock_item_factory
    ):
        """
        Test that machine-specific recipes are filtered out when no machine
        is provided.
        """
        stock = stock_item_factory("walnut", 3.0)
        results = manager_with_recipes.find_recipes(
            stock, capabilities={CUT}, machine=None
        )

        assert len(results) == 3
        assert "machine-a-walnut-3mm" not in [r.uid for r in results]
        assert results[0].uid == "walnut-3mm-cut"

    def test_find_recipes_material_only_match(
        self,
        manager_with_recipes: RecipeManager,
        machine_a: Mock,
        stock_item_factory,
    ):
        """Test finding recipes when only material matches."""
        stock = stock_item_factory("walnut", 10.0)  # Thickness mismatch
        results = manager_with_recipes.find_recipes(
            stock, capabilities={CUT}, machine=machine_a
        )

        assert len(results) == 2
        assert results[0].uid == "walnut-any-thickness"
        assert results[1].uid == "generic-cut"

    def test_find_recipes_thickness_only_match(
        self,
        manager_with_recipes: RecipeManager,
        machine_a: Mock,
        stock_item_factory,
    ):
        """Test finding recipes when only thickness matches."""
        stock = stock_item_factory("mdf", 3.0)  # Material mismatch
        # This recipe doesn't match because it also has a machine spec
        # so only generic-cut should appear.
        results = manager_with_recipes.find_recipes(
            stock, capabilities={CUT}, machine=machine_a
        )

        assert len(results) == 1
        assert results[0].uid == "generic-cut"

    def test_find_recipes_no_match(
        self,
        manager_with_recipes: RecipeManager,
        machine_a: Mock,
        stock_item_factory,
    ):
        """
        Test finding recipes when nothing matches, only generic should return.
        """
        stock = stock_item_factory("mdf", 10.0)
        results = manager_with_recipes.find_recipes(
            stock, capabilities={CUT}, machine=machine_a
        )

        assert len(results) == 1
        assert results[0].uid == "generic-cut"

    def test_find_recipes_no_stock_provided(
        self, manager_with_recipes: RecipeManager, machine_a: Mock
    ):
        """
        Test that only generic recipes are returned when no stock is provided.
        """
        results = manager_with_recipes.find_recipes(
            None, capabilities={CUT}, machine=machine_a
        )

        assert len(results) == 1
        assert results[0].uid == "generic-cut"

    def test_find_recipes_filters_by_capability(
        self,
        manager_with_recipes: RecipeManager,
        machine_a: Mock,
        stock_item_factory,
    ):
        """Test that find_recipes correctly filters by capability."""
        stock = stock_item_factory("walnut", 3.0)
        results = manager_with_recipes.find_recipes(
            stock, capabilities={ENGRAVE}, machine=machine_a
        )

        assert len(results) == 1
        assert results[0].uid == "generic-engrave"

    def test_find_recipes_multi_capability(
        self,
        manager_with_recipes: RecipeManager,
        machine_a: Mock,
        stock_item_factory,
    ):
        """Test finding recipes that match one of several capabilities."""
        stock = stock_item_factory("walnut", 3.0)
        results = manager_with_recipes.find_recipes(
            stock, capabilities={CUT, ENGRAVE}, machine=machine_a
        )

        assert len(results) == 6
        result_uids = [r.uid for r in results]

        # All the CUT recipes should be present
        assert "machine-a-laser-1-walnut-3mm" in result_uids
        assert "machine-a-walnut-3mm" in result_uids
        assert "walnut-3mm-cut" in result_uids
        assert "walnut-any-thickness" in result_uids
        assert "generic-cut" in result_uids

        # The ENGRAVE recipe should also be present
        assert "generic-engrave" in result_uids

        # Check the sorting order (specificity, then name)
        # The 5 CUT recipes should come first, in their specific order.
        # Then the ENGRAVE recipe, which has the same low specificity as
        # generic-cut and will be sorted after it by name.
        assert results[0].uid == "machine-a-laser-1-walnut-3mm"
        assert results[1].uid == "machine-a-walnut-3mm"
        assert results[2].uid == "walnut-3mm-cut"
        assert results[3].uid == "walnut-any-thickness"
        assert results[4].uid == "generic-cut"
        assert results[5].uid == "generic-engrave"

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

from ..shared.util.atomic import atomic_write_yaml, load_yaml_with_backup
from .recipe import Recipe

if TYPE_CHECKING:
    from ..machine.models.machine import Machine
    from .stock import StockItem

logger = logging.getLogger(__name__)


class RecipeManager:
    """
    Manages loading, saving, and querying Recipe objects from a directory.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.recipes: dict[str, Recipe] = {}
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.load()

    def filename_from_id(self, recipe_id: str) -> Path:
        """Generates a consistent filename for a given recipe UID."""
        return self.base_dir / f"{recipe_id}.yaml"

    def load(self):
        """Loads all recipes from the base directory."""
        self.recipes.clear()
        for file in sorted(self.base_dir.glob("*.yaml")):
            self._load_file(file)
        logger.info(f"Loaded {len(self.recipes)} recipes.")

    def _load_file(self, file: Path):
        """Loads a single recipe file, keeping a backup-based recovery
        path for files damaged by an interrupted write."""
        try:
            data, _ = load_yaml_with_backup(file)
        except Exception as e:  # noqa: BLE001 - arbitrary user YAML file
            logger.error(f"Error loading recipe file {file.name}: {e}")
            return
        if not data:
            logger.warning(f"Skipping empty or invalid recipe {file.name}")
            return
        try:
            recipe = Recipe.from_dict(data)
            # Ensure UID from file content is used, but fallback
            # to filename
            recipe.uid = data.get("uid", file.stem)
            self.recipes[recipe.uid] = recipe
        except Exception as e:  # noqa: BLE001 - arbitrary user YAML file
            logger.error(f"Error loading recipe file {file.name}: {e}")

    def save_recipe(self, recipe: Recipe):
        """Saves a single recipe to a YAML file."""
        logger.debug(f"Saving recipe {recipe.name} ({recipe.uid})")
        recipe_file = self.filename_from_id(recipe.uid)
        try:
            data = recipe.to_dict()
        except Exception as e:  # noqa: BLE001 - recipe may hold user data
            logger.error(f"Failed to serialize recipe {recipe.uid}: {e}")
            return
        try:
            atomic_write_yaml(recipe_file, data)
        except (OSError, yaml.YAMLError) as e:
            logger.error(f"Failed to save recipe {recipe.uid}: {e}")

    def add_recipe(self, recipe: Recipe):
        """Adds a recipe to the manager and saves it."""
        if recipe.uid in self.recipes:
            logger.warning(
                f"Recipe with UID {recipe.uid} already exists. Overwriting."
            )
        self.recipes[recipe.uid] = recipe
        self.save_recipe(recipe)

    def delete_recipe(self, recipe_uid: str):
        """Deletes a recipe from memory and removes its file."""
        if recipe_uid in self.recipes:
            del self.recipes[recipe_uid]
            recipe_file = self.filename_from_id(recipe_uid)
            if recipe_file.exists():
                try:
                    recipe_file.unlink()
                    logger.info(f"Deleted recipe file: {recipe_file}")
                except OSError as e:
                    logger.error(
                        f"Failed to delete recipe file {recipe_file}: {e}"
                    )

    def get_recipe_by_id(self, recipe_id: str) -> Recipe | None:
        """Retrieves a recipe by its unique identifier."""
        return self.recipes.get(recipe_id)

    def get_all_recipes(self) -> list[Recipe]:
        """Returns a list of all loaded recipes."""
        return list(self.recipes.values())

    def find_recipes(
        self,
        stock_items: list["StockItem"],
        machine: Optional["Machine"] = None,
        step_type: str | None = None,
    ) -> list[Recipe]:
        """
        Finds matching recipes, sorted from most specific to least specific.

        Args:
            stock_items: A list of StockItems. If empty, only generic recipes
                         (without material/thickness constraints) are returned.
            machine: An optional machine context to match against. Can be None.
            step_type: An optional step class name (as registered in
                       ``step_registry``) to match
                       :attr:`Recipe.target_step_types` against.

        Returns:
            A list of Recipe objects, sorted by relevance.
        """
        # 1. Filter the recipes using the `matches` method
        candidates = [
            r
            for r in self.get_all_recipes()
            if r.matches(stock_items, machine, step_type=step_type)
        ]

        # 2. Sort candidates based on their specificity score and name
        candidates.sort(
            key=lambda r: (r.get_specificity_score(), r.name.lower())
        )

        return candidates

    def is_material_in_use(self, material_uid: str) -> bool:
        """
        Checks if any recipe in the library references the given material UID.
        """
        for recipe in self.recipes.values():
            if recipe.material_uid == material_uid:
                return True
        return False

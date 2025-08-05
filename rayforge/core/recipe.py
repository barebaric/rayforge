import yaml
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from blinker import Signal
from .workflow import Workflow


logger = logging.getLogger(__name__)


class Recipe:
    """
    A saved, portable entity that contains a Workflow and its
    associated metadata (e.g., material, thickness). It lives in a
    user-level library, outside any specific document.
    """
    def __init__(self, name: str):
        self.uid: str = str(uuid.uuid4())
        self.name: str = name
        self.workflow: Optional[Workflow] = None
        self.metadata: Dict[str, Any] = {
            "material": "",
            "thickness_mm": 0.0,
            "description": "",
            "author": "",
        }
        self.changed = Signal()

    def set_name(self, name: str):
        """Sets the recipe's name and triggers a save."""
        if self.name == name:
            return
        self.name = name
        self.changed.send(self)

    def set_workflow(self, workflow: Optional[Workflow]):
        """
        Sets the recipe's workflow and triggers a save.
        Note: The workflow must be serializable.
        """
        # A proper comparison would be complex; for now, any set is a change.
        self.workflow = workflow
        self.changed.send(self)

    def set_metadata(self, metadata: Dict[str, Any]):
        """Sets the recipe's metadata and triggers a save."""
        if self.metadata == metadata:
            return
        self.metadata = metadata
        self.changed.send(self)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the Recipe to a dictionary for saving.
        NOTE: Workflow serialization is required for this to be complete.
        """
        return {
            "uid": self.uid,
            "name": self.name,
            # "workflow": self.workflow.to_dict() if self.workflow else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recipe':
        """
        Deserializes a Recipe from a dictionary.
        NOTE: A loaded Recipe's workflow is not instantiated here because it
        lacks a document context. It should be instantiated by the consumer
        when applying the recipe to a document.
        """
        recipe = cls(data["name"])
        recipe.uid = data.get("uid", recipe.uid)
        recipe.metadata = data.get("metadata", {})
        # workflow_data = data.get("workflow")
        # if workflow_data:
        #     # The workflow would be instantiated here if it were document-
        #     # independent, but it requires a `doc` object.
        #     pass
        return recipe


class RecipeManager:
    """
    Manages loading and saving Recipe objects from/to a dedicated folder.
    Automatically saves recipes when they are changed.
    """
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.recipes: Dict[str, Recipe] = {}
        self._recipe_ref_for_pyreverse: Recipe
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.load()

    def filename_from_id(self, recipe_id: str) -> Path:
        return self.base_dir / f"{recipe_id}.yaml"

    def add_recipe(self, recipe: Recipe):
        """Adds a recipe to the manager and connects its changed signal."""
        if recipe.uid in self.recipes:
            return
        self.recipes[recipe.uid] = recipe
        recipe.changed.connect(self.on_recipe_changed)

    def get_recipe_by_id(self, recipe_id: str) -> Optional[Recipe]:
        return self.recipes.get(recipe_id)

    def get_all_recipes(self) -> List[Recipe]:
        return list(self.recipes.values())

    def save_recipe(self, recipe: Recipe):
        """Saves a single recipe to a YAML file."""
        logger.debug(f"Saving recipe {recipe.name} ({recipe.uid})")
        recipe_file = self.filename_from_id(recipe.uid)
        with open(recipe_file, 'w') as f:
            data = recipe.to_dict()
            yaml.safe_dump(data, f)

    def load_recipe(self, recipe_id: str) -> Optional[Recipe]:
        """Loads a single recipe from a file and adds it to the manager."""
        recipe_file = self.filename_from_id(recipe_id)
        if not recipe_file.exists():
            logger.warning(f"Recipe file not found: {recipe_file}")
            return None
        try:
            with open(recipe_file, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading recipe file {recipe_file.name}: {e}")
            return None
        if not data:
            logger.warning(f"Skipping empty or invalid recipe {f.name}")
            return None

        recipe = Recipe.from_dict(data)
        recipe.uid = recipe_id  # Ensure UID matches filename stem
        self.add_recipe(recipe)
        return recipe

    def on_recipe_changed(self, recipe: Recipe, **kwargs):
        """Callback that saves the recipe whenever it changes."""
        self.save_recipe(recipe)

    def load(self):
        """Loads all recipes from the base directory."""
        self.recipes.clear()
        for file in self.base_dir.glob("*.yaml"):
            self.load_recipe(file.stem)

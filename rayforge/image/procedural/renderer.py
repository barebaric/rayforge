import cairo
import importlib
import json
import logging
from typing import Optional, Tuple, Callable

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer

logger = logging.getLogger(__name__)


class ProceduralRenderer(Renderer):
    """
    Renders procedural content by dispatching to a drawing function.

    This renderer is a generic execution engine. It reads a "recipe" from
    the WorkPiece's ImportSource data. The recipe is a JSON object that
    specifies a path to a drawing function and the geometric parameters to
    pass to it. This allows for creating resolution-independent content
    without hardcoding rendering logic for each procedural type.
    """

    def _get_recipe_and_func(
        self, workpiece: "WorkPiece", func_key: str
    ) -> Tuple[Optional[dict], Optional[dict], Optional[Callable]]:
        """Helper to deserialize the recipe and import a function."""
        if not workpiece.source or not workpiece.source.data:
            logger.warning("Procedural workpiece has no source data.")
            return None, None, None

        try:
            recipe = json.loads(workpiece.source.data)
            params = recipe.get("params", {})
            func_path = recipe.get(func_key)

            if not func_path:
                logger.error(f"Recipe missing required key: '{func_key}'")
                return None, None, None

            module_path, func_name = func_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            return recipe, params, func

        except (
            json.JSONDecodeError,
            KeyError,
            ImportError,
            AttributeError,
        ) as e:
            logger.error(
                f"Failed to load procedural function: {e}", exc_info=True
            )
            return None, None, None

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        """
        Calculates the natural size by calling the size function specified
        in the content's recipe.
        """
        _recipe, params, size_func = self._get_recipe_and_func(
            workpiece, "size_function_path"
        )
        if not size_func or params is None:
            return None

        try:
            return size_func(params)
        except Exception as e:
            logger.error(
                f"Error executing procedural size function: {e}", exc_info=True
            )
            return None

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Renders the workpiece by calling the drawing function specified
        in the content's recipe.
        """
        _recipe, params, draw_func = self._get_recipe_and_func(
            workpiece, "drawing_function_path"
        )
        if not draw_func or params is None:
            return None

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        try:
            draw_func(ctx, width, height, params)
            return surface
        except Exception as e:
            logger.error(
                f"Error executing procedural drawing function: {e}",
                exc_info=True,
            )
            return None


PROCEDURAL_RENDERER = ProceduralRenderer()

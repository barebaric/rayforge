import importlib
import json
import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import cairo

from ..base_renderer import Renderer

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from ...image.structures import ImportResult

logger = logging.getLogger(__name__)


class ProceduralRenderer(Renderer):
    """
    Renders procedural content by dispatching to a drawing function.

    This renderer is a generic execution engine. It reads a "recipe" from
    the WorkPiece's SourceAsset data. The recipe is a JSON object that
    specifies a path to a drawing function and the geometric parameters to
    pass to it. This allows for creating resolution-independent content
    without hardcoding rendering logic for each procedural type.
    """

    def _get_recipe_and_func_internal(
        self, source_original_data: bytes | None, func_key: str
    ) -> tuple[dict | None, dict | None, Callable | None]:
        """Helper to deserialize the recipe and import a function."""
        if not source_original_data:
            logger.warning("Procedural source has no original_data.")
            return None, None, None

        try:
            recipe = json.loads(source_original_data)
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
            logger.exception("Failed to load procedural function")
            return None, None, None

    def render_preview_image(
        self,
        import_result: "ImportResult",
        target_width: int,
        target_height: int,
    ) -> pyvips.Image | None:
        """Renders the procedural recipe at the target preview dimensions."""
        if not import_result.payload:
            return None

        return self.render_base_image(
            data=import_result.payload.source.original_data,
            width=target_width,
            height=target_height,
        )

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> pyvips.Image | None:
        _, params, draw_func = self._get_recipe_and_func_internal(
            data, "drawing_function_path"
        )
        if not draw_func or params is None:
            return None

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        try:
            draw_func(ctx, width, height, params)
        except Exception as e:
            logger.exception("Error executing procedural drawing function")
            return None

        h, w = surface.get_height(), surface.get_width()
        vips_image = pyvips.Image.new_from_memory(
            surface.get_data(), w, h, 4, "uchar"
        )
        b, g, r, a = (
            vips_image[0],
            vips_image[1],
            vips_image[2],
            vips_image[3],
        )
        return r.bandjoin([g, b, a])


PROCEDURAL_RENDERER = ProceduralRenderer()

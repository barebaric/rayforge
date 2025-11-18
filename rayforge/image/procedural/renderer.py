import cairo
import json
import logging
import importlib
from typing import Optional, Tuple, Dict, Any, Callable, TYPE_CHECKING
from ..base_renderer import Renderer
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from ...core.geo import Geometry
    from ...core.source_asset_segment import SourceAssetSegment

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
        self, source_original_data: Optional[bytes], func_key: str
    ) -> Tuple[Optional[dict], Optional[dict], Optional[Callable]]:
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
            logger.error(
                f"Failed to load procedural function: {e}", exc_info=True
            )
            return None, None, None

    def get_natural_size_from_data(
        self,
        *,
        render_data: Optional[bytes],
        source_segment: Optional["SourceAssetSegment"],
        source_metadata: Optional[Dict[str, Any]],
        boundaries: Optional["Geometry"] = None,
        current_size: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        # For procedural content, the recipe is passed as render_data (which
        # is the same as original_data).
        _recipe, params, size_func = self._get_recipe_and_func_internal(
            render_data, "size_function_path"
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

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
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
            logger.error(
                f"Error executing procedural drawing function: {e}",
                exc_info=True,
            )
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

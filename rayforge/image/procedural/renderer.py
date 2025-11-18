import cairo
import importlib
import json
import logging
from typing import Optional, Tuple, Callable, Dict, Any, TYPE_CHECKING
import pyvips

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer

if TYPE_CHECKING:
    from ...core.source_asset_segment import SourceAssetSegment
    from ...core.geo import Geometry
    from ...core.matrix import Matrix

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

    def _get_recipe_and_func(
        self, workpiece: "WorkPiece", func_key: str
    ) -> Tuple[Optional[dict], Optional[dict], Optional[Callable]]:
        """Helper to deserialize the recipe and import a function."""
        source = workpiece.source
        if not source:
            logger.warning("Procedural workpiece has no source.")
            return None, None, None
        return self._get_recipe_and_func_internal(
            source.original_data, func_key
        )

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        """
        Calculates the natural size by calling the size function specified
        in the content's recipe.
        """
        source = workpiece.source
        if not source:
            return None

        _recipe, params, size_func = self._get_recipe_and_func_internal(
            source.original_data, "size_function_path"
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
        source = workpiece.source
        if not source:
            return None

        _recipe, params, draw_func = self._get_recipe_and_func_internal(
            source.original_data, "drawing_function_path"
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

    def render_from_data(
        self,
        *,
        render_data: Optional[bytes],
        original_data: Optional[bytes] = None,
        source_segment: Optional["SourceAssetSegment"] = None,
        source_px_dims: Optional[Tuple[int, int]] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
        boundaries: Optional["Geometry"] = None,
        workpiece_matrix: Optional["Matrix"] = None,
        width: int,
        height: int,
    ) -> Optional[pyvips.Image]:
        _recipe, params, draw_func = self._get_recipe_and_func_internal(
            render_data, "drawing_function_path"
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
        b, g, r, a = vips_image[0], vips_image[1], vips_image[2], vips_image[3]
        return r.bandjoin([g, b, a])


PROCEDURAL_RENDERER = ProceduralRenderer()

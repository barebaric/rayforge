import importlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict

from ...core.geo import Geometry
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import VectorizationSpec, ProceduralSpec
from ...core.workpiece import WorkPiece
from ...core.source_asset_segment import SourceAssetSegment
from ..base_importer import Importer, ImportPayload
from .renderer import PROCEDURAL_RENDERER

logger = logging.getLogger(__name__)


class ProceduralImporter(Importer):
    """
    A factory for creating procedural WorkPieces.

    Unlike file-based importers that parse existing data, this importer is
    instantiated programmatically with the "recipe" for creating content.
    It generates the SourceAsset and WorkPiece on the fly.
    """

    def __init__(
        self,
        *,
        drawing_function_path: str,
        size_function_path: str,
        params: Dict,
        name: str,
    ):
        """
        Initializes the importer with the recipe for procedural content.

        Args:
            drawing_function_path: Fully-qualified path to the drawing
              function.
            size_function_path: Fully-qualified path to the size calculation
              function.
            params: Dictionary of geometric parameters for the functions.
            name: The name for the generated WorkPiece and source file.
        """
        self.drawing_function_path = drawing_function_path
        self.size_function_path = size_function_path
        self.params = params
        self.name = name

        # Create the recipe data that will be stored in the SourceAsset.
        recipe_dict = {
            "drawing_function_path": self.drawing_function_path,
            "size_function_path": self.size_function_path,
            "params": self.params,
        }
        recipe_data = json.dumps(recipe_dict).encode("utf-8")

        # Initialize the base class. The recipe data serves as the "raw_data".
        super().__init__(data=recipe_data, source_file=Path(f"[{self.name}]"))

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional[ImportPayload]:
        """
        Generates the ImportPayload containing the procedural WorkPiece and
        its corresponding SourceAsset.
        """
        # Step 1: Calculate the initial size by dynamically calling the size
        # function.
        try:
            module_path, func_name = self.size_function_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            size_func = getattr(module, func_name)
            width_mm, height_mm = size_func(self.params)
        except (ImportError, AttributeError, ValueError) as e:
            logger.error(
                f"Failed to load procedural size function: {e}", exc_info=True
            )
            return None

        # Step 2: Create the SourceAsset using the pre-generated recipe.
        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,  # This is the recipe data
            renderer=PROCEDURAL_RENDERER,
            width_mm=width_mm,
            height_mm=height_mm,
        )

        # Step 3: Create and configure the WorkPiece.
        # Per the architectural contract, we generate a normalized Y-down
        # geometry (a 1x1 frame) for the segment mask.
        frame_geo = Geometry()
        frame_geo.move_to(0, 0)
        frame_geo.line_to(1, 0)
        frame_geo.line_to(1, 1)
        frame_geo.line_to(0, 1)
        frame_geo.close_path()

        procedural_spec = ProceduralSpec()
        gen_config = SourceAssetSegment(
            source_asset_uid=source.uid,
            segment_mask_geometry=frame_geo,
            vectorization_spec=procedural_spec,
        )

        wp = WorkPiece(
            name=self.name,
            source_segment=gen_config,
        )
        wp.natural_width_mm = width_mm
        wp.natural_height_mm = height_mm
        wp.set_size(width_mm, height_mm)

        # Step 4: Return the complete payload.
        return ImportPayload(source=source, items=[wp])

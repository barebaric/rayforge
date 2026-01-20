import importlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict

from ...core.geo import Geometry
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import VectorizationSpec, ProceduralSpec
from ..assembler import ItemAssembler
from ..base_importer import (
    Importer,
    ImportPayload,
    ImporterFeature,
    ImportManifest,
)
from ..engine import NormalizationEngine
from ..structures import ParsingResult, LayerGeometry, VectorizationResult
from .renderer import PROCEDURAL_RENDERER

logger = logging.getLogger(__name__)


class ProceduralImporter(Importer):
    """
    A factory for creating procedural WorkPieces.

    Unlike file-based importers that parse existing data, this importer is
    instantiated programmatically with the "recipe" for creating content.
    It generates the SourceAsset and WorkPiece on the fly.
    """

    features = {ImporterFeature.PROCEDURAL_GENERATION}

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

    def scan(self) -> ImportManifest:
        """
        Calculates the size of the procedural item from its recipe.
        """
        try:
            module_path, func_name = self.size_function_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            size_func = getattr(module, func_name)
            size_mm = size_func(self.params)
            return ImportManifest(title=self.name, natural_size_mm=size_mm)
        except (ImportError, AttributeError, ValueError) as e:
            logger.error(
                f"Failed to calculate procedural size: {e}", exc_info=True
            )
            return ImportManifest(
                title=self.name,
                warnings=["Could not calculate size from procedural recipe."],
            )

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional[ImportPayload]:
        """
        Generates the ImportPayload containing the procedural WorkPiece and
        its corresponding SourceAsset.
        """
        # Ensure we use a ProceduralSpec if none provided
        spec = vectorization_spec or ProceduralSpec()

        # Phase 2: Parse (Calculate dimensions)
        parse_result = self.parse()
        if not parse_result:
            return None

        # Create SourceAsset
        _, _, w, h = parse_result.page_bounds
        # For procedural, native units are 1:1 with mm (scale=1.0)
        width_mm = w
        height_mm = h

        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,  # This is the recipe data
            renderer=PROCEDURAL_RENDERER,
            width_mm=width_mm,
            height_mm=height_mm,
        )

        # Phase 3: Vectorize (Generate placeholder geometry)
        vec_result = self.vectorize(parse_result, spec)

        # Phase 4: Layout
        engine = NormalizationEngine()
        plan = engine.calculate_layout(vec_result, spec)
        if not plan:
            return ImportPayload(source=source, items=[])

        # Phase 5: Assembly
        assembler = ItemAssembler()
        items = assembler.create_items(
            source_asset=source,
            layout_plan=plan,
            spec=spec,
            source_name=self.name,
            geometries=vec_result.geometries_by_layer,
        )

        return ImportPayload(source=source, items=items)

    def parse(self) -> Optional[ParsingResult]:
        """
        Phase 2: "Parse" the procedural parameters to determine geometric
        properties.
        """
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

        # Define the native coordinate system as 1 unit = 1 mm.
        # This preserves the aspect ratio in the parsing result.
        page_bounds = (0.0, 0.0, float(width_mm), float(height_mm))
        layer_id = "__default__"

        return ParsingResult(
            page_bounds=page_bounds,
            native_unit_to_mm=1.0,  # 1 native unit = 1 mm
            is_y_down=True,  # Standardize on Y-down for generated content
            layers=[
                LayerGeometry(layer_id=layer_id, content_bounds=page_bounds)
            ],
        )

    def vectorize(
        self, parse_result: ParsingResult, spec: VectorizationSpec
    ) -> VectorizationResult:
        """
        Phase 3: Generate the pristine geometry.
        We create a rectangle matching the calculated dimensions.
        """
        _, _, w, h = parse_result.page_bounds

        frame_geo = Geometry()
        frame_geo.move_to(0, 0)
        frame_geo.line_to(w, 0)
        frame_geo.line_to(w, h)
        frame_geo.line_to(0, h)
        frame_geo.close_path()

        # Retrieve the layer ID (we know there is one)
        layer_id = parse_result.layers[0].layer_id

        return VectorizationResult(
            geometries_by_layer={layer_id: frame_geo},
            source_parse_result=parse_result,
        )

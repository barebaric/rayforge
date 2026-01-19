import logging
from typing import Optional, Tuple, Dict
from ...core.geo import Geometry
from ...core.vectorization_spec import VectorizationSpec
from ..base_importer import (
    Importer,
    ImportPayload,
    ImporterFeature,
    ImportManifest,
)
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import PassthroughSpec
from ..assembler import ItemAssembler
from ..engine import NormalizationEngine
from ..structures import ParsingResult, LayerGeometry, VectorizationResult
from .renderer import RUIDA_RENDERER
from .parser import RuidaParser, RuidaParseError
from .job import RuidaJob

logger = logging.getLogger(__name__)


class RuidaImporter(Importer):
    label = "Ruida files"
    mime_types = ("application/x-rd-file", "application/octet-stream")
    extensions = (".rd",)
    features = {ImporterFeature.DIRECT_VECTOR}

    def scan(self) -> ImportManifest:
        """
        Scans the Ruida file to determine its overall dimensions.
        """
        try:
            job = self._get_job()
            if not job.commands:
                return ImportManifest(
                    title=self.source_file.name,
                    warnings=["File contains no vector data."],
                )

            min_x, min_y, max_x, max_y = job.get_extents()
            width_mm = max_x - min_x
            height_mm = max_y - min_y
            return ImportManifest(
                title=self.source_file.name,
                natural_size_mm=(width_mm, height_mm),
            )
        except RuidaParseError as e:
            logger.warning(
                f"Ruida scan failed for {self.source_file.name}: {e}"
            )
            return ImportManifest(
                title=self.source_file.name,
                warnings=["Could not parse Ruida file. It may be corrupt."],
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during Ruida scan for "
                f"{self.source_file.name}: {e}",
                exc_info=True,
            )
            return ImportManifest(
                title=self.source_file.name,
                warnings=[
                    "An unexpected error occurred while scanning the "
                    "Ruida file."
                ],
            )

    def _get_job(self) -> RuidaJob:
        """Parses the Ruida data into a job object."""
        parser = RuidaParser(self.raw_data)
        return parser.parse()

    def get_doc_items(
        self, vectorization_spec: Optional[VectorizationSpec] = None
    ) -> Optional[ImportPayload]:
        try:
            # Phase 2: Parsing
            parse_result, geometries_by_layer = self._parse_to_result()
        except RuidaParseError as e:
            logger.error("Ruida file parse failed: %s", e)
            return None

        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=RUIDA_RENDERER,
        )

        if not parse_result.layers:
            return ImportPayload(source=source, items=[])

        # Store natural size metadata from the definitive bounds
        _, _, w, h = parse_result.page_bounds
        source.width_mm = w
        source.height_mm = h

        # Phase 3: Vectorize (packaging)
        vec_result = self._vectorize(parse_result, geometries_by_layer)

        # Phase 4: Layout
        engine = NormalizationEngine()
        spec = vectorization_spec or PassthroughSpec()
        plan = engine.calculate_layout(vec_result, spec)

        # Since Ruida files are always a single merged entity, the plan will
        # have one item with layer_id=None. The assembler expects the geometry
        # under the `None` key.
        geometries: Dict[Optional[str], Geometry] = {
            None: list(geometries_by_layer.values())[0]
        }

        # Phase 5: Assembly
        assembler = ItemAssembler()
        items = assembler.create_items(
            source_asset=source,
            layout_plan=plan,
            spec=spec,
            source_name=self.source_file.stem,
            geometries=geometries,
        )
        return ImportPayload(source=source, items=items)

    def _vectorize(
        self,
        parse_result: ParsingResult,
        geometries_by_layer: Dict[Optional[str], Geometry],
    ) -> VectorizationResult:
        """Phase 3: Package parsed data for the layout engine."""
        return VectorizationResult(
            geometries_by_layer=geometries_by_layer,
            source_parse_result=parse_result,
        )

    def _parse_to_result(
        self,
    ) -> Tuple[ParsingResult, Dict[Optional[str], Geometry]]:
        job = self._get_job()
        pristine_geo = self._get_geometry(job)
        pristine_geo.close_gaps()

        if not job.commands or pristine_geo.is_empty():
            # Return empty but valid structures
            empty_result = ParsingResult(
                page_bounds=(0, 0, 0, 0),
                native_unit_to_mm=1.0,
                is_y_down=False,
                layers=[],
            )
            geometries: Dict[Optional[str], Geometry] = {None: pristine_geo}
            return empty_result, geometries

        min_x, min_y, max_x, max_y = job.get_extents()
        width_mm = max_x - min_x
        height_mm = max_y - min_y

        # Use a virtual layer ID for consistency with other importers
        layer_id = "__default__"
        page_bounds = (min_x, min_y, width_mm, height_mm)
        parse_result = ParsingResult(
            page_bounds=page_bounds,
            native_unit_to_mm=1.0,
            is_y_down=False,
            layers=[
                LayerGeometry(layer_id=layer_id, content_bounds=page_bounds)
            ],
        )
        geometries: Dict[Optional[str], Geometry] = {layer_id: pristine_geo}
        return parse_result, geometries

    def _get_geometry(self, job: RuidaJob) -> Geometry:
        """
        Returns the parsed vector geometry. The coordinate system is
        canonical (Y-up, origin at bottom-left of content).
        """
        geo = Geometry()
        if not job.commands:
            return geo

        _min_x, min_y, _max_x, max_y = job.get_extents()
        y_flip_val = max_y + min_y

        for cmd in job.commands:
            # Check the command type first, then safely access params.
            if cmd.command_type in ("Move_Abs", "Cut_Abs"):
                # Ensure params are valid before unpacking.
                if not cmd.params or len(cmd.params) != 2:
                    logger.warning(
                        f"Skipping Ruida command with invalid params: {cmd}"
                    )
                    continue

                x, y = cmd.params
                flipped_y = y_flip_val - y
                if cmd.command_type == "Move_Abs":
                    geo.move_to(x, flipped_y)
                elif cmd.command_type == "Cut_Abs":
                    geo.line_to(x, flipped_y)
        return geo

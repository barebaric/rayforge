import logging
from typing import Optional, Dict
from pathlib import Path

from ...core.geo import Geometry
from ...core.vectorization_spec import VectorizationSpec
from ..base_importer import (
    Importer,
    ImporterFeature,
)
from ...core.source_asset import SourceAsset
from ..structures import (
    ParsingResult,
    LayerGeometry,
    VectorizationResult,
    ImportManifest,
)
from .renderer import RUIDA_RENDERER
from .parser import RuidaParser, RuidaParseError
from .job import RuidaJob

logger = logging.getLogger(__name__)


class RuidaImporter(Importer):
    label = "Ruida files"
    mime_types = ("application/x-rd-file", "application/octet-stream")
    extensions = (".rd",)
    features = {ImporterFeature.DIRECT_VECTOR}

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self._job: Optional[RuidaJob] = None
        self._geometries_by_layer: Dict[Optional[str], Geometry] = {}

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

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        """
        Creates a SourceAsset for Ruida import.
        """
        _, _, w, h = parse_result.page_bounds

        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=RUIDA_RENDERER,
            width_mm=w,
            height_mm=h,
        )
        return source

    def vectorize(
        self,
        parse_result: ParsingResult,
        spec: VectorizationSpec,
    ) -> VectorizationResult:
        """
        Phase 3: Package parsed data for the layout engine.
        Since Ruida files are always a single merged entity, we package the
        geometry under the `None` key for the layout engine and assembler.
        """
        # A Ruida file is conceptually a single "layer" or entity.
        # We merge all geometries into one entry for the layout engine.
        merged_geo = Geometry()
        for geo in self._geometries_by_layer.values():
            merged_geo.extend(geo)

        geometries_for_layout: Dict[Optional[str], Geometry] = {
            None: merged_geo
        }

        return VectorizationResult(
            geometries_by_layer=geometries_for_layout,
            source_parse_result=parse_result,
        )

    def parse(self) -> Optional[ParsingResult]:
        """Phase 2: Parse Ruida file into geometric facts."""
        try:
            job = self._get_job()
            self._job = job
        except RuidaParseError as e:
            logger.error("Ruida file parse failed: %s", e)
            self._job = None
            return None

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
            self._geometries_by_layer: Dict[Optional[str], Geometry] = {
                None: pristine_geo
            }
            return empty_result

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
                LayerGeometry(
                    layer_id=layer_id,
                    name=layer_id,
                    content_bounds=page_bounds,
                )
            ],
        )
        self._geometries_by_layer = {layer_id: pristine_geo}
        return parse_result

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

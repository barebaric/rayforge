import logging
from typing import Optional

from ...core.vectorization_spec import (
    PassthroughSpec,
    TraceSpec,
    VectorizationSpec,
)
from ..base_importer import Importer, ImporterFeature
from ..structures import ImportResult, ParsingResult, VectorizationResult
from .svg_trace import SvgTraceImporter
from .svg_vector import SvgVectorImporter

logger = logging.getLogger(__name__)


class SvgImporter(Importer):
    """
    Facade importer for SVG files.
    """

    label = "SVG files"
    mime_types = ("image/svg+xml",)
    extensions = (".svg",)
    features = {
        ImporterFeature.DIRECT_VECTOR,
        ImporterFeature.BITMAP_TRACING,
        ImporterFeature.LAYER_SELECTION,
    }

    def scan(self):
        return SvgVectorImporter(self.raw_data, self.source_file).scan()

    def parse(self) -> Optional[ParsingResult]:
        raise NotImplementedError(
            "SvgImporter is a facade; parse is delegated via get_doc_items"
        )

    def vectorize(
        self, parse_result: ParsingResult, spec: VectorizationSpec
    ) -> VectorizationResult:
        raise NotImplementedError(
            "SvgImporter is a facade; vectorize is delegated via get_doc_items"
        )

    def create_source_asset(self, parse_result: ParsingResult):
        raise NotImplementedError(
            "SvgImporter is a facade; create_source_asset is delegated"
        )

    def get_doc_items(
        self, vectorization_spec: Optional[VectorizationSpec] = None
    ) -> Optional[ImportResult]:
        """
        Delegates the full import process to the appropriate strategy.
        """
        spec_to_use = vectorization_spec
        if spec_to_use is None:
            spec_to_use = PassthroughSpec()

        if isinstance(spec_to_use, TraceSpec):
            logger.debug("SvgImporter: Delegating to SvgTraceImporter.")
            delegate = SvgTraceImporter(self.raw_data, self.source_file)
        else:
            if (
                isinstance(spec_to_use, PassthroughSpec)
                and not spec_to_use.active_layer_ids
            ):
                logger.debug(
                    "Empty PassthroughSpec detected in facade. "
                    "Scanning for all available layers."
                )
                manifest = self.scan()
                all_layer_ids = [layer.id for layer in manifest.layers]
                if all_layer_ids:
                    logger.debug(
                        "Populating spec with all layers: %s",
                        all_layer_ids,
                    )
                    spec_to_use = PassthroughSpec(
                        active_layer_ids=all_layer_ids,
                        create_new_layers=False,
                    )

            logger.debug("SvgImporter: Delegating to SvgVectorImporter.")
            delegate = SvgVectorImporter(self.raw_data, self.source_file)

        return delegate.get_doc_items(spec_to_use)

from typing import Optional
from pathlib import Path

from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import (
    VectorizationSpec,
    TraceSpec,
    PassthroughSpec,
)
from ..base_importer import Importer, ImporterFeature
from ..structures import (
    ParsingResult,
    VectorizationResult,
    ImportResult,
    ImportManifest,
)
from .pdf_trace import PdfTraceImporter
from .pdf_vector import PdfVectorImporter

import logging

logger = logging.getLogger(__name__)


class PdfImporter(Importer):
    """
    A Facade importer for PDF files.

    Routes the import request to either the Vector strategy (for direct
    path extraction) or the Trace strategy (for rendering and tracing),
    depending on the provided VectorizationSpec.
    """

    label = "PDF files"
    mime_types = ("application/pdf",)
    extensions = (".pdf",)
    features = {
        ImporterFeature.DIRECT_VECTOR,
        ImporterFeature.BITMAP_TRACING,
    }

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)

    def scan(self) -> ImportManifest:
        return PdfVectorImporter(self.raw_data, self.source_file).scan()

    def get_doc_items(
        self, vectorization_spec: Optional[VectorizationSpec] = None
    ) -> Optional[ImportResult]:
        spec_to_use = vectorization_spec
        if spec_to_use is None:
            spec_to_use = PassthroughSpec()

        if isinstance(spec_to_use, TraceSpec):
            logger.debug("PdfImporter: Delegating to PdfTraceImporter.")
            delegate = PdfTraceImporter(self.raw_data, self.source_file)
        else:
            logger.debug("PdfImporter: Delegating to PdfVectorImporter.")
            delegate = PdfVectorImporter(self.raw_data, self.source_file)

        import_result = delegate.get_doc_items(spec_to_use)

        if import_result:
            import_result.warnings.extend(self._warnings)
            import_result.errors.extend(self._errors)

        return import_result

    def parse(self) -> Optional[ParsingResult]:
        raise NotImplementedError(
            "PdfImporter is a facade; parse is delegated via get_doc_items"
        )

    def vectorize(
        self, parse_result: ParsingResult, spec: VectorizationSpec
    ) -> VectorizationResult:
        raise NotImplementedError(
            "PdfImporter is a facade; vectorize is delegated via get_doc_items"
        )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        raise NotImplementedError(
            "PdfImporter is a facade; create_source_asset is delegated"
        )

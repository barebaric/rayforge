import logging
from gettext import gettext as _
from pathlib import Path

from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import VectorizationSpec
from ..pdf.importer import PdfImporter
from ..structures import ImportManifest, ImportResult
from . import ghostscript

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
PDF_HEADER_WINDOW = 1024


def is_pdf(data: bytes) -> bool:
    """True if the data is a PDF stream (PDF allows leading junk)."""
    return PDF_MAGIC in data[:PDF_HEADER_WINDOW]


class _PostScriptImporter(PdfImporter):
    """
    Imports files that are either PDF streams or PostScript. PDF data is
    handed to the PDF importer unchanged; PostScript is first converted
    to PDF with Ghostscript. The source asset keeps the PDF, so projects
    render and re-import without Ghostscript.
    """

    def __init__(self, data: bytes, source_file: Path | None = None):
        super().__init__(data, source_file)
        self._pdf_ready: bool | None = None

    def _missing_ghostscript_message(self) -> str:
        return _(
            "Importing EPS and PostScript files requires Ghostscript, "
            "which was not found. Install Ghostscript and try again."
        )

    def _ensure_pdf(self) -> bool:
        if self._pdf_ready is not None:
            return self._pdf_ready
        if is_pdf(self.raw_data):
            self._pdf_ready = True
            return True
        try:
            self.raw_data = ghostscript.convert_to_pdf(self.raw_data)
            self._pdf_ready = True
        except ghostscript.GhostscriptNotFound:
            self.add_error(self._missing_ghostscript_message())
            self._pdf_ready = False
        except ghostscript.GhostscriptError as e:
            logger.warning(
                f"Ghostscript failed for {self.source_file.name}: {e}"
            )
            self.add_error(
                _("Ghostscript could not convert {filename}: {error}").format(
                    filename=self.source_file.name, error=e
                )
            )
            self._pdf_ready = False
        return self._pdf_ready

    def _failed_result(self) -> ImportResult:
        return ImportResult(
            payload=None,
            parse_result=None,
            warnings=self._warnings,
            errors=self._errors,
        )

    def scan(self) -> ImportManifest:
        if not self._ensure_pdf():
            return ImportManifest(
                title=self.source_file.name,
                warnings=self._warnings,
                errors=self._errors,
            )
        manifest = super().scan()
        manifest.title = self.source_file.name
        return manifest

    def get_doc_items(
        self, vectorization_spec: VectorizationSpec | None = None
    ) -> ImportResult | None:
        if not self._ensure_pdf():
            return self._failed_result()
        return super().get_doc_items(vectorization_spec)

    def get_doc_items_for_reimport(
        self,
        existing_source_asset: SourceAsset,
        vectorization_spec: VectorizationSpec,
    ) -> ImportResult | None:
        if not self._ensure_pdf():
            return self._failed_result()
        return super().get_doc_items_for_reimport(
            existing_source_asset, vectorization_spec
        )


class AiImporter(_PostScriptImporter):
    """
    Adobe Illustrator files. Since Illustrator 9 they embed a PDF by
    default ("Create PDF Compatible File"); older or PostScript-only
    files need Ghostscript.
    """

    label = "Adobe Illustrator files"
    mime_types = (
        "application/illustrator",
        "application/vnd.adobe.illustrator",
    )
    extensions = (".ai",)

    def _missing_ghostscript_message(self) -> str:
        return _(
            "This Illustrator file was saved without PDF compatibility. "
            "Rayforge can only read it with Ghostscript, which was not "
            "found. Install Ghostscript, or save the file again in "
            'Illustrator with "Create PDF Compatible File" enabled.'
        )


class EpsImporter(_PostScriptImporter):
    """Encapsulated PostScript and PostScript files, via Ghostscript."""

    label = "EPS and PostScript files"
    mime_types = (
        "image/x-eps",
        "application/postscript",
        "application/eps",
        "image/eps",
    )
    extensions = (".eps", ".epsf", ".ps")

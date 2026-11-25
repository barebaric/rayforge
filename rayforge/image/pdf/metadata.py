"""PDF metadata access shared by the PDF importers and renderers."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf


class PdfInfoError(Exception):
    """Raised when a PDF cannot be read for basic metadata."""


@dataclass(frozen=True)
class PdfInfo:
    """Basic metadata from a PDF document."""

    page_count: int
    width_pt: float
    height_pt: float
    title: str | None


def read_pdf_info(data: bytes) -> PdfInfo:
    """
    Reads the page count, first-page mediabox size, and document title.

    Dimensions are returned in PDF points. A document without pages is
    reported with a page count of zero and zero dimensions.

    Raises:
        PdfInfoError: If the data is not a readable, unencrypted PDF.
    """
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except (
        pymupdf.FileDataError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as e:
        raise PdfInfoError(str(e)) from e

    try:
        if doc.needs_pass:
            raise PdfInfoError("PDF is encrypted")
        if doc.page_count == 0:
            return PdfInfo(0, 0.0, 0.0, None)

        page = doc[0]
        mediabox = page.mediabox
        title = (doc.metadata or {}).get("title") or None
        return PdfInfo(
            page_count=doc.page_count,
            width_pt=float(mediabox.width),
            height_pt=float(mediabox.height),
            title=title,
        )
    except (IndexError, RuntimeError, ValueError) as e:
        raise PdfInfoError(str(e)) from e
    finally:
        doc.close()

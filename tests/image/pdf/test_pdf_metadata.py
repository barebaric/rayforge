import pymupdf
import pytest

from rayforge.image.pdf.metadata import PdfInfoError, read_pdf_info


def _make_pdf(
    width: float = 100,
    height: float = 50,
    title: str | None = None,
) -> bytes:
    doc = pymupdf.open()
    doc.new_page(width=width, height=height)
    if title is not None:
        doc.set_metadata({"title": title})
    data = doc.tobytes()
    doc.close()
    return data


def test_read_pdf_info_returns_page_size_and_title():
    info = read_pdf_info(_make_pdf(title="Calibration card"))

    assert info.page_count == 1
    assert info.width_pt == pytest.approx(100.0)
    assert info.height_pt == pytest.approx(50.0)
    assert info.title == "Calibration card"


def test_read_pdf_info_without_title_returns_none():
    info = read_pdf_info(_make_pdf())

    assert info.title is None


def test_read_pdf_info_rejects_invalid_data():
    with pytest.raises(PdfInfoError):
        read_pdf_info(b"not a pdf")


def test_read_pdf_info_rejects_encrypted_pdf():
    doc = pymupdf.open()
    doc.new_page(width=100, height=50)
    data = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,  # type: ignore
        owner_pw="owner",
        user_pw="user",
    )
    doc.close()

    with pytest.raises(PdfInfoError, match="encrypted"):
        read_pdf_info(data)

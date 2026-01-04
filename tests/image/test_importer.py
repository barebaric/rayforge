from pathlib import Path
import pytest
from rayforge.image import import_file
from rayforge.core.vectorization_spec import TraceSpec
from rayforge.core.workpiece import WorkPiece


@pytest.fixture
def tests_root() -> Path:
    """Returns the path to the 'tests' directory."""
    return Path(__file__).parent.parent


class TestImporter:
    def test_import_svg_by_path(self, tests_root: Path):
        """Tests importing an SVG file using its path."""
        svg_path = tests_root / "image/svg/nested-rect.svg"
        payload = import_file(svg_path)

        assert payload is not None
        assert payload.source is not None
        assert payload.items is not None
        assert len(payload.items) == 1
        wp = payload.items[0]
        assert isinstance(wp, WorkPiece)
        assert payload.source.source_file == svg_path

    def test_import_png_by_path_with_tracing(self, tests_root: Path):
        """Tests importing and vectorizing a PNG using its path."""
        png_path = tests_root / "image/png/color.png"
        payload = import_file(png_path, vectorization_spec=TraceSpec())

        assert payload is not None
        assert len(payload.items) == 1
        wp = payload.items[0]
        assert isinstance(wp, WorkPiece)
        assert wp.boundaries is not None, (
            "PNG import should have generated vectors"
        )
        assert len(wp.boundaries) > 0

    def test_import_by_data_with_mime(self, tests_root: Path):
        """Tests importing from bytes data when a MIME type is provided."""
        svg_path = tests_root / "image/svg/nested-rect.svg"
        svg_data = svg_path.read_bytes()

        payload = import_file(svg_data, mime_type="image/svg+xml")

        assert payload is not None
        assert len(payload.items) == 1
        assert isinstance(payload.items[0], WorkPiece)
        # When importing from data, the source file defaults to "Untitled"
        assert payload.source.source_file.name == "Untitled"

    def test_import_by_data_no_mime_fails(self, tests_root: Path):
        """
        Tests that importing from bytes data fails without a MIME type,
        as there's no extension to infer the importer from.
        """
        svg_path = tests_root / "image/svg/nested-rect.svg"
        svg_data = svg_path.read_bytes()

        payload = import_file(svg_data)
        assert payload is None

    def test_import_unknown_extension_fails(self, tmp_path: Path):
        """
        Tests that importing a file with an unknown extension fails.
        """
        unknown_file = tmp_path / "test.unknown"
        unknown_file.write_text("data")

        payload = import_file(unknown_file)
        assert payload is None

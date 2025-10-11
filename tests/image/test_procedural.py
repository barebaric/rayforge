import cairo
import json
import pytest
from pathlib import Path
from typing import Tuple, cast, Dict
from unittest.mock import Mock

from rayforge.core.import_source import ImportSource
from rayforge.core.matrix import Matrix
from rayforge.core.workpiece import WorkPiece
from rayforge.image.procedural.importer import ProceduralImporter
from rayforge.image.procedural.renderer import PROCEDURAL_RENDERER

# --- Mock Procedural Functions for Testing ---

# FIX: Use a list for color to match JSON's array type.
MOCK_PARAMS = {"width": 80.0, "height": 40.0, "color": [0.0, 1.0, 0.0]}


def mock_size_func(params: Dict) -> Tuple[float, float]:
    """A mock function that calculates size based on params."""
    return params.get("width", 10.0), params.get("height", 10.0)


def mock_draw_func(ctx: cairo.Context, width: int, height: int, params: Dict):
    """A mock function that draws a simple colored rectangle."""
    color = params.get("color", (1.0, 0.0, 0.0))  # Default to red
    ctx.set_source_rgb(*color)
    ctx.rectangle(0, 0, width, height)
    ctx.fill()


def mock_error_func(params: Dict):
    """A mock function designed to raise an exception for testing."""
    raise ValueError("This function is designed to fail.")


# --- Test Setup and Fixtures ---

# Get fully-qualified paths to the mock functions within this module
MOCK_DRAW_FUNC_PATH = f"{__name__}.mock_draw_func"
MOCK_SIZE_FUNC_PATH = f"{__name__}.mock_size_func"
MOCK_ERROR_FUNC_PATH = f"{__name__}.mock_error_func"


def _setup_workpiece_with_context(
    importer: ProceduralImporter,
) -> WorkPiece:
    """Helper to run importer and correctly link workpiece to its source."""
    payload = importer.get_doc_items()
    assert payload is not None
    source = payload.source
    wp = cast(WorkPiece, payload.items[0])

    # Mock the document context so workpiece.source resolves correctly
    mock_doc = Mock()
    mock_doc.import_sources = {source.uid: source}
    mock_doc.get_import_source_by_uid.side_effect = mock_doc.import_sources.get

    mock_parent = Mock()
    mock_parent.doc = mock_doc
    mock_parent.get_world_transform.return_value = Matrix.identity()
    wp.parent = mock_parent

    return wp


@pytest.fixture
def procedural_workpiece() -> WorkPiece:
    """A fixture that creates a valid procedural workpiece."""
    importer = ProceduralImporter(
        drawing_function_path=MOCK_DRAW_FUNC_PATH,
        size_function_path=MOCK_SIZE_FUNC_PATH,
        params=MOCK_PARAMS,
        name="Test Procedural Item",
    )
    return _setup_workpiece_with_context(importer)


class TestProceduralImporter:
    def test_importer_creates_correct_payload(self):
        """
        Tests that the importer generates a valid payload with a correctly
        configured ImportSource and WorkPiece.
        """
        importer = ProceduralImporter(
            drawing_function_path=MOCK_DRAW_FUNC_PATH,
            size_function_path=MOCK_SIZE_FUNC_PATH,
            params=MOCK_PARAMS,
            name="Test Procedural Item",
        )
        payload = importer.get_doc_items()

        assert payload is not None
        assert len(payload.items) == 1

        # 1. Test the ImportSource
        source = payload.source
        assert isinstance(source, ImportSource)
        assert source.renderer is PROCEDURAL_RENDERER
        assert source.source_file == Path("[Test Procedural Item]")

        # Verify the "recipe" data stored in the source
        recipe = json.loads(source.original_data)
        assert recipe["drawing_function_path"] == MOCK_DRAW_FUNC_PATH
        assert recipe["size_function_path"] == MOCK_SIZE_FUNC_PATH
        assert recipe["params"] == MOCK_PARAMS

        # 2. Test the WorkPiece
        wp = cast(WorkPiece, payload.items[0])
        assert isinstance(wp, WorkPiece)
        assert wp.name == "Test Procedural Item"
        assert wp.import_source_uid == source.uid

        # Verify the size was set correctly by calling the mock size func
        assert wp.size == (MOCK_PARAMS["width"], MOCK_PARAMS["height"])

    def test_importer_handles_invalid_size_function_path(self):
        """
        Tests that the importer returns None if the size function cannot be
        resolved.
        """
        importer = ProceduralImporter(
            drawing_function_path=MOCK_DRAW_FUNC_PATH,
            size_function_path="non.existent.path",
            params={},
            name="Bad Item",
        )
        payload = importer.get_doc_items()
        assert payload is None


class TestProceduralRenderer:
    def test_get_natural_size(self, procedural_workpiece: WorkPiece):
        """
        Tests that the renderer can correctly call the size function from the
        workpiece's recipe.
        """
        size = PROCEDURAL_RENDERER.get_natural_size(procedural_workpiece)
        assert size is not None
        assert size == (MOCK_PARAMS["width"], MOCK_PARAMS["height"])

    def test_render_to_pixels(self, procedural_workpiece: WorkPiece):
        """
        Tests that the renderer can correctly call the drawing function and
        produce a rendered surface.
        """
        surface = PROCEDURAL_RENDERER.render_to_pixels(
            procedural_workpiece, width=10, height=10
        )
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 10
        assert surface.get_height() == 10

        # Verify the mock draw function ran by checking a pixel color
        # The mock draws a green rectangle. Cairo surfaces are BGRA.
        buf = surface.get_data()
        assert buf[0] == 0x00  # Blue
        assert buf[1] == 0xFF  # Green
        assert buf[2] == 0x00  # Red
        assert buf[3] == 0xFF  # Alpha

    def test_renderer_handles_malformed_json(self):
        """
        Tests graceful failure when the source data is not valid JSON.
        """
        source = ImportSource(
            source_file=Path("bad"),
            original_data=b"{not_json",
            renderer=PROCEDURAL_RENDERER,
        )
        wp = WorkPiece(name="bad")
        wp.import_source_uid = source.uid
        # Mock parent context
        mock_doc = Mock(import_sources={source.uid: source})
        mock_doc.get_import_source_by_uid.side_effect = (
            mock_doc.import_sources.get
        )
        wp.parent = Mock(doc=mock_doc)

        assert PROCEDURAL_RENDERER.get_natural_size(wp) is None
        assert PROCEDURAL_RENDERER.render_to_pixels(wp, 10, 10) is None

    def test_renderer_handles_invalid_function_path(self):
        """
        Tests graceful failure when a function path in the recipe is invalid.
        """
        recipe = {
            "drawing_function_path": "invalid.path",
            "size_function_path": "invalid.path",
            "params": {},
        }
        source = ImportSource(
            source_file=Path("bad"),
            original_data=json.dumps(recipe).encode("utf-8"),
            renderer=PROCEDURAL_RENDERER,
        )
        wp = WorkPiece(name="bad")
        wp.import_source_uid = source.uid
        mock_doc = Mock(import_sources={source.uid: source})
        mock_doc.get_import_source_by_uid.side_effect = (
            mock_doc.import_sources.get
        )
        wp.parent = Mock(doc=mock_doc)

        assert PROCEDURAL_RENDERER.get_natural_size(wp) is None
        assert PROCEDURAL_RENDERER.render_to_pixels(wp, 10, 10) is None

    def test_renderer_handles_function_exception(self):
        """
        Tests graceful failure when a procedural function raises an exception.
        """
        recipe = {
            "drawing_function_path": MOCK_ERROR_FUNC_PATH,
            "size_function_path": MOCK_ERROR_FUNC_PATH,
            "params": {},
        }
        source = ImportSource(
            source_file=Path("bad"),
            original_data=json.dumps(recipe).encode("utf-8"),
            renderer=PROCEDURAL_RENDERER,
        )
        wp = WorkPiece(name="bad")
        wp.import_source_uid = source.uid
        mock_doc = Mock(import_sources={source.uid: source})
        mock_doc.get_import_source_by_uid.side_effect = (
            mock_doc.import_sources.get
        )
        wp.parent = Mock(doc=mock_doc)

        assert PROCEDURAL_RENDERER.get_natural_size(wp) is None
        assert PROCEDURAL_RENDERER.render_to_pixels(wp, 10, 10) is None

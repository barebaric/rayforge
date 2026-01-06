import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.layer import Layer
from rayforge.core.source_asset import SourceAsset
from rayforge.core.vectorization_spec import TraceSpec
from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.file_cmd import FileCmd, PreviewResult
from rayforge.image import ImportPayload
from rayforge.image.svg.renderer import SVG_RENDERER
from rayforge.shared.tasker.manager import TaskManager


@pytest.fixture
def mock_editor(context_initializer):
    """Provides a DocEditor instance with mocked dependencies."""
    task_manager = MagicMock(spec=TaskManager)
    doc = Doc()
    return DocEditor(task_manager, context_initializer, doc)


@pytest.fixture
def file_cmd(mock_editor):
    """Provides a FileCmd instance."""
    return FileCmd(mock_editor, mock_editor.task_manager)


@pytest.fixture
def sample_workpiece():
    """Provides a sample WorkPiece instance."""
    wp = WorkPiece(name="Test WorkPiece")
    wp.set_size(10.0, 20.0)
    return wp


@pytest.fixture
def sample_layer():
    """Provides a sample Layer instance."""
    return Layer(name="Test Layer")


@pytest.fixture
def sample_source_asset():
    """Provides a sample SourceAsset instance."""
    asset = SourceAsset(
        source_file=Path("test.svg"),
        original_data=b"<svg></svg>",
        renderer=SVG_RENDERER,
    )
    return asset


@pytest.fixture
def sample_payload(sample_workpiece, sample_source_asset):
    """Provides a sample ImportPayload instance."""
    return ImportPayload(
        source=sample_source_asset,
        items=[sample_workpiece],
    )


class TestScanImportFile:
    """Tests for scan_import_file method."""

    def test_scan_svg_file(self, file_cmd):
        """Test scanning an SVG file extracts layer manifest."""
        svg_bytes = b'<svg><g id="layer1"></g><g id="layer2"></g></svg>'
        result = file_cmd.scan_import_file(svg_bytes, "image/svg+xml")

        assert "layers" in result
        assert isinstance(result["layers"], list)

    def test_scan_non_svg_file(self, file_cmd):
        """Test scanning a non-SVG file returns empty dict."""
        png_bytes = b"\x89PNG\r\n\x1a\n"
        result = file_cmd.scan_import_file(png_bytes, "image/png")

        assert result == {}

    def test_scan_svg_with_exception(self, file_cmd, caplog):
        """Test that SVG scan exceptions are handled gracefully."""
        invalid_svg = b"not valid svg"
        result = file_cmd.scan_import_file(invalid_svg, "image/svg+xml")

        assert result == {"layers": []}


class TestExtractFirstWorkpiece:
    """Tests for _extract_first_workpiece method."""

    def test_extract_workpiece_from_list(self, file_cmd, sample_workpiece):
        """Test extracting WorkPiece from a list of items."""
        result = file_cmd._extract_first_workpiece([sample_workpiece])

        assert result is sample_workpiece

    def test_extract_workpiece_from_layer(self, file_cmd, sample_workpiece):
        """Test extracting WorkPiece from a Layer's children."""
        layer = Layer(name="Test Layer")
        layer.add_child(sample_workpiece)

        result = file_cmd._extract_first_workpiece([layer])

        assert result is sample_workpiece

    def test_extract_workpiece_from_nested_layer(self, file_cmd):
        """Test extracting WorkPiece from nested Layers."""
        outer_layer = Layer(name="Outer Layer")
        inner_layer = Layer(name="Inner Layer")
        wp = WorkPiece(name="Nested WorkPiece")
        inner_layer.add_child(wp)
        outer_layer.add_child(inner_layer)

        result = file_cmd._extract_first_workpiece([outer_layer])

        assert result is wp

    def test_extract_workpiece_not_found(self, file_cmd, sample_layer):
        """Test returning None when no WorkPiece is found."""
        result = file_cmd._extract_first_workpiece([sample_layer])

        assert result is None

    def test_extract_workpiece_empty_list(self, file_cmd):
        """Test returning None for empty list."""
        result = file_cmd._extract_first_workpiece([])

        assert result is None


class TestCalculateItemsBbox:
    """Tests for _calculate_items_bbox method."""

    def test_calculate_bbox_single_item(self, file_cmd, sample_workpiece):
        """Test calculating bbox for a single item."""
        sample_workpiece.set_size(10.0, 20.0)
        sample_workpiece.pos = (5.0, 10.0)

        result = file_cmd._calculate_items_bbox([sample_workpiece])

        assert result is not None
        x, y, w, h = result
        assert w == 10.0
        assert h == 20.0

    def test_calculate_bbox_multiple_items(self, file_cmd):
        """Test calculating bbox for multiple items."""
        wp1 = WorkPiece(name="Item 1")
        wp1.set_size(10.0, 10.0)
        wp1.pos = (0.0, 0.0)

        wp2 = WorkPiece(name="Item 2")
        wp2.set_size(15.0, 15.0)
        wp2.pos = (10.0, 10.0)

        result = file_cmd._calculate_items_bbox([wp1, wp2])

        assert result is not None
        x, y, w, h = result
        assert x == 0.0
        assert y == 0.0
        assert w == 25.0
        assert h == 25.0

    def test_calculate_bbox_empty_list(self, file_cmd):
        """Test returning None for empty list."""
        result = file_cmd._calculate_items_bbox([])

        assert result is None


class TestPositionNewlyImportedItems:
    """Tests for _position_newly_imported_items method."""

    def test_position_at_specific_point(self, file_cmd, sample_workpiece):
        """Test positioning items at a specific point."""
        sample_workpiece.set_size(10.0, 20.0)
        sample_workpiece.pos = (0.0, 0.0)
        items = [sample_workpiece]

        file_cmd._position_newly_imported_items(items, (50.0, 100.0))

        assert pytest.approx(45.0) == sample_workpiece.pos[0]
        assert pytest.approx(90.0) == sample_workpiece.pos[1]

    def test_position_multiple_items_same_delta(self, file_cmd):
        """Test that multiple items get the same delta."""
        wp1 = WorkPiece(name="Item 1")
        wp1.set_size(10.0, 10.0)
        wp1.pos = (0.0, 0.0)

        wp2 = WorkPiece(name="Item 2")
        wp2.set_size(15.0, 15.0)
        wp2.pos = (10.0, 10.0)

        items = [wp1, wp2]
        file_cmd._position_newly_imported_items(items, (50.0, 50.0))

        assert pytest.approx(37.5) == wp1.pos[0]
        assert pytest.approx(37.5) == wp1.pos[1]
        assert pytest.approx(47.5) == wp2.pos[0]
        assert pytest.approx(47.5) == wp2.pos[1]

    def test_position_none_uses_fit_and_center(
        self, file_cmd, sample_workpiece
    ):
        """Test that None position triggers fit and center."""
        with patch.object(
            file_cmd, "_fit_and_center_imported_items"
        ) as mock_fit_center:
            file_cmd._position_newly_imported_items([sample_workpiece], None)

            mock_fit_center.assert_called_once_with([sample_workpiece])


class TestFitAndCenterImportedItems:
    """Tests for _fit_and_center_imported_items method."""

    def test_fit_and_center_no_config(self, file_cmd, sample_workpiece):
        """Test that method returns early when no config is available."""
        with patch("rayforge.doceditor.file_cmd.get_context") as mock_ctx:
            mock_ctx.return_value.config = None

            file_cmd._fit_and_center_imported_items([sample_workpiece])

    def test_fit_and_center_no_machine(self, file_cmd, sample_workpiece):
        """Test that method returns early when no machine is configured."""
        with patch("rayforge.doceditor.file_cmd.get_context") as mock_ctx:
            mock_ctx.return_value.config.machine = None

            file_cmd._fit_and_center_imported_items([sample_workpiece])

    def test_fit_and_center_no_bbox(self, file_cmd, sample_workpiece):
        """Test that method returns early when bbox cannot be calculated."""
        with patch.object(
            file_cmd, "_calculate_items_bbox", return_value=None
        ):
            file_cmd._fit_and_center_imported_items([sample_workpiece])

    def test_fit_and_center_scale_down(self, file_cmd):
        """Test scaling down items that are too large."""
        wp = WorkPiece(name="Large Item")
        wp.set_size(300.0, 200.0)
        wp.pos = (0.0, 0.0)

        with patch("rayforge.doceditor.file_cmd.get_context") as mock_ctx:
            mock_machine = MagicMock()
            mock_machine.dimensions = (200, 150)
            mock_ctx.return_value.config.machine = mock_machine

            file_cmd._fit_and_center_imported_items([wp])

            bbox = wp.bbox
            assert bbox[2] <= 200
            assert bbox[3] <= 150

    def test_fit_and_center_center_items(self, file_cmd):
        """Test centering items in the workspace."""
        wp = WorkPiece(name="Item")
        wp.set_size(50.0, 50.0)
        wp.pos = (0.0, 0.0)

        with patch("rayforge.doceditor.file_cmd.get_context") as mock_ctx:
            mock_machine = MagicMock()
            mock_machine.dimensions = (200, 150)
            mock_ctx.return_value.config.machine = mock_machine

            file_cmd._fit_and_center_imported_items([wp])

            bbox = wp.bbox
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            assert abs(center_x - 100) < 1e-6
            assert abs(center_y - 75) < 1e-6


class TestCommitItemsToDocument:
    """Tests for _commit_items_to_document method."""

    def test_commit_workpiece_to_document(self, file_cmd, sample_workpiece):
        """Test committing a WorkPiece to the document."""
        source = SourceAsset(
            source_file=Path("test.svg"),
            original_data=b"<svg></svg>",
            renderer=SVG_RENDERER,
        )
        filename = Path("test.svg")

        file_cmd._commit_items_to_document(
            [sample_workpiece], source, filename
        )

        assert source.uid in file_cmd._editor.doc.source_assets
        assert sample_workpiece in (file_cmd._editor.doc.active_layer.children)

    def test_commit_layer_to_document(self, file_cmd, sample_layer):
        """Test committing a Layer to the document."""
        source = SourceAsset(
            source_file=Path("test.svg"),
            original_data=b"<svg></svg>",
            renderer=SVG_RENDERER,
        )
        filename = Path("test.svg")

        file_cmd._commit_items_to_document([sample_layer], source, filename)

        assert source in file_cmd._editor.doc.source_assets.values()
        assert sample_layer in file_cmd._editor.doc.children

    def test_commit_with_sketches(self, file_cmd, sample_workpiece):
        """Test committing items with sketches."""
        from rayforge.core.sketcher.sketch import Sketch

        source = SourceAsset(
            source_file=Path("test.svg"),
            original_data=b"<svg></svg>",
            renderer=SVG_RENDERER,
        )
        sketch = Sketch(name="Test Sketch")
        filename = Path("test.svg")

        file_cmd._commit_items_to_document(
            [sample_workpiece], source, filename, sketches=[sketch]
        )

        assert sketch in file_cmd._editor.doc.sketches.values()


class TestFinalizeImportOnMainThread:
    """Tests for _finalize_import_on_main_thread method."""

    def test_finalize_import(self, file_cmd, sample_payload):
        """Test finalizing import on main thread."""
        filename = Path("test.svg")

        file_cmd._finalize_import_on_main_thread(
            sample_payload, filename, None
        )

        assert sample_payload.source.uid in file_cmd._editor.doc.source_assets
        assert sample_payload.items[0] in (
            file_cmd._editor.doc.active_layer.children
        )


class TestPreviewResult:
    """Tests for PreviewResult dataclass."""

    def test_preview_result_creation(self):
        """Test creating a PreviewResult instance."""
        result = PreviewResult(
            image_bytes=b"fake png data",
            payload=None,
            aspect_ratio=1.5,
            warnings=["warning 1"],
        )

        assert result.image_bytes == b"fake png data"
        assert result.payload is None
        assert result.aspect_ratio == 1.5
        assert result.warnings == ["warning 1"]

    def test_preview_result_defaults(self):
        """Test PreviewResult default values."""
        result = PreviewResult(image_bytes=b"data", payload=None)

        assert result.aspect_ratio == 1.0
        assert result.warnings == []


class TestGeneratePreview:
    """Tests for generate_preview method."""

    @pytest.mark.asyncio
    async def test_generate_preview_success(self, file_cmd):
        """Test successful preview generation."""
        with patch.object(
            file_cmd,
            "_generate_preview_impl",
            return_value=PreviewResult(image_bytes=b"png", payload=None),
        ):
            result = await file_cmd.generate_preview(
                b"data", "test.png", "image/png", TraceSpec(), 256
            )

            assert result is not None
            assert result.image_bytes == b"png"

    @pytest.mark.asyncio
    async def test_generate_preview_failure(self, file_cmd):
        """Test preview generation failure."""
        with patch.object(
            file_cmd, "_generate_preview_impl", return_value=None
        ):
            result = await file_cmd.generate_preview(
                b"data", "test.png", "image/png", TraceSpec(), 256
            )

            assert result is None


class TestGeneratePreviewImpl:
    """Tests for _generate_preview_impl method."""

    def test_preview_impl_no_payload(self, file_cmd):
        """Test preview when no payload is generated."""
        with patch(
            "rayforge.doceditor.file_cmd.import_file_from_bytes",
            return_value=None,
        ):
            result = file_cmd._generate_preview_impl(
                b"data", "test.png", "image/png", TraceSpec(), 256
            )

            assert result is None

    def test_preview_impl_no_workpiece(self, file_cmd):
        """Test preview when no WorkPiece is found."""
        payload = ImportPayload(
            source=SourceAsset(
                source_file=Path("test.svg"),
                original_data=b"<svg></svg>",
                renderer=SVG_RENDERER,
            ),
            items=[],
        )

        with patch(
            "rayforge.doceditor.file_cmd.import_file_from_bytes",
            return_value=payload,
        ):
            result = file_cmd._generate_preview_impl(
                b"data", "test.png", "image/png", TraceSpec(), 256
            )

            assert result is None


class TestLoadFileAsync:
    """Tests for _load_file_async method."""

    @pytest.mark.asyncio
    async def test_load_file_async_success(self, file_cmd):
        """Test successful async file load."""
        payload = ImportPayload(
            source=SourceAsset(
                source_file=Path("test.svg"),
                original_data=b"<svg></svg>",
                renderer=SVG_RENDERER,
            ),
            items=[WorkPiece(name="Test")],
        )

        with patch(
            "rayforge.doceditor.file_cmd.import_file",
            return_value=payload,
        ):
            result = await file_cmd._load_file_async(
                Path("test.svg"), "image/svg+xml", None
            )

            assert result is payload

    @pytest.mark.asyncio
    async def test_load_file_async_failure(self, file_cmd):
        """Test async file load failure."""
        with patch(
            "rayforge.doceditor.file_cmd.import_file",
            return_value=None,
        ):
            result = await file_cmd._load_file_async(
                Path("test.svg"), "image/svg+xml", None
            )

            assert result is None


class TestLoadFileFromPath:
    """Tests for load_file_from_path method."""

    def test_load_file_adds_task(self, file_cmd):
        """Test that load_file_from_path adds a task to the task manager."""
        filename = Path("test.svg")

        file_cmd.load_file_from_path(filename, "image/svg+xml", None, None)

        file_cmd._task_manager.add_coroutine.assert_called_once()


class TestExportGcodeToPath:
    """Tests for export_gcode_to_path method."""

    def test_export_gcode_failure(self, file_cmd, tmp_path):
        """Test G-code export failure."""
        export_path = tmp_path / "output.gcode"

        with patch.object(
            file_cmd._editor.pipeline, "generate_job_artifact"
        ) as mock_generate:

            def failure_callback(when_done):
                when_done(None, Exception("Export failed"))

            mock_generate.side_effect = failure_callback
            file_cmd.export_gcode_to_path(export_path)

            assert not export_path.exists()

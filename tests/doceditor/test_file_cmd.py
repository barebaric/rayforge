import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.matrix import Matrix
from rayforge.core.source_asset import SourceAsset
from rayforge.core.vectorization_spec import TraceSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.file_cmd import FileCmd, PreviewResult, ImportAction
from rayforge.image import (
    ImportPayload,
    ImportResult,
    ParsingResult,
    ImporterFeature,
    ImportManifest,
    LayerInfo,
)
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


@pytest.fixture
def sample_parse_result():
    """Provides a sample ParsingResult."""
    document_bounds = (0, 0, 10, 10)
    unit_scale = 1.0
    x, y, w, h = document_bounds
    world_frame = (x * unit_scale, 0.0, w * unit_scale, h * unit_scale)
    return ParsingResult(
        document_bounds=document_bounds,
        native_unit_to_mm=unit_scale,
        is_y_down=True,
        layers=[],
        world_frame_of_reference=world_frame,
        background_world_transform=Matrix.identity(),
    )


@pytest.fixture
def sample_import_result(sample_payload, sample_parse_result):
    """Provides a sample ImportResult."""
    return ImportResult(
        payload=sample_payload, parse_result=sample_parse_result
    )


class TestScanImportFile:
    """Tests for scan_import_file method."""

    def test_scan_delegates_to_importer(self, file_cmd):
        """
        Test that scan_import_file correctly finds and calls the
        importer's scan method.
        """
        svg_bytes = b"<svg></svg>"
        file_path = Path("test.svg")
        mime_type = "image/svg+xml"

        mock_importer_instance = MagicMock()
        mock_manifest = ImportManifest(
            layers=[LayerInfo(id="layer1", name="Layer 1")]
        )
        mock_importer_instance.scan.return_value = mock_manifest

        mock_importer_class = MagicMock()
        mock_importer_class.return_value = mock_importer_instance

        with patch(
            "rayforge.doceditor.file_cmd.importer_by_mime_type",
            {mime_type: mock_importer_class},
        ):
            result = file_cmd.scan_import_file(svg_bytes, file_path, mime_type)

            mock_importer_class.assert_called_once_with(
                data=svg_bytes, source_file=file_path
            )
            mock_importer_instance.scan.assert_called_once()
            assert result is mock_manifest

    def test_scan_no_importer_found(self, file_cmd, caplog):
        """Test scanning a file with no matching importer."""
        some_bytes = b"data"
        file_path = Path("test.unknown")
        mime_type = "application/octet-stream"

        with patch("rayforge.doceditor.file_cmd.importer_by_mime_type", {}):
            with patch(
                "rayforge.doceditor.file_cmd.importer_by_extension", {}
            ):
                result = file_cmd.scan_import_file(
                    some_bytes, file_path, mime_type
                )

                assert isinstance(result, ImportManifest)
                assert result.title == "test.unknown"
                assert result.warnings == ["Unsupported file type: .unknown"]
                assert "No importer found" in caplog.text

    def test_scan_importer_raises_exception(self, file_cmd, caplog):
        """
        Test that exceptions during the importer's scan are handled
        gracefully.
        """
        svg_bytes = b"<svg></svg>"
        file_path = Path("test.svg")
        mime_type = "image/svg+xml"

        mock_importer_instance = MagicMock()
        mock_importer_instance.scan.side_effect = ValueError("Parsing failed")

        mock_importer_class = MagicMock()
        mock_importer_class.__name__ = "MockSvgImporter"
        mock_importer_class.return_value = mock_importer_instance

        with patch(
            "rayforge.doceditor.file_cmd.importer_by_mime_type",
            {mime_type: mock_importer_class},
        ):
            result = file_cmd.scan_import_file(svg_bytes, file_path, mime_type)

            assert isinstance(result, ImportManifest)
            assert result.title == "test.svg"
            assert result.warnings == [
                "An unexpected error occurred during file analysis."
            ]
            assert "Error scanning file" in caplog.text
            assert "MockSvgImporter" in caplog.text


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

        assert source.uid in file_cmd._editor.doc.assets
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

        assert source in file_cmd._editor.doc.get_all_assets()
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

        assert sketch in file_cmd._editor.doc.get_all_assets()


class TestFinalizeImportOnMainThread:
    """Tests for _finalize_import_on_main_thread method."""

    def test_finalize_import(self, file_cmd, sample_payload):
        """Test finalizing import on main thread."""
        filename = Path("test.svg")

        file_cmd._finalize_import_on_main_thread(
            sample_payload, filename, None
        )

        assert sample_payload.source.uid in file_cmd._editor.doc.assets
        assert sample_payload.items[0] in (
            file_cmd._editor.doc.active_layer.children
        )


class TestPreviewResult:
    """Tests for PreviewResult dataclass."""

    def test_preview_result_creation(self, sample_parse_result):
        """Test creating a PreviewResult instance."""
        result = PreviewResult(
            image_bytes=b"fake png data",
            payload=None,
            parse_result=sample_parse_result,
            aspect_ratio=1.5,
            warnings=["warning 1"],
        )

        assert result.image_bytes == b"fake png data"
        assert result.payload is None
        assert result.parse_result is sample_parse_result
        assert result.aspect_ratio == 1.5
        assert result.warnings == ["warning 1"]

    def test_preview_result_defaults(self):
        """Test PreviewResult default values."""
        result = PreviewResult(
            image_bytes=b"data", payload=None, parse_result=None
        )

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
            return_value=PreviewResult(
                image_bytes=b"png", payload=None, parse_result=None
            ),
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

    def test_preview_impl_no_import_result(self, file_cmd):
        """Test preview when importer returns None."""
        with patch(
            "rayforge.image.base_importer.Importer.get_doc_items",
            return_value=None,
        ):
            result = file_cmd._generate_preview_impl(
                b"data", "test.png", "image/png", TraceSpec(), 256
            )
            assert result is None

    def test_preview_impl_no_workpiece(self, file_cmd, sample_import_result):
        """Test preview when no WorkPiece is found in the payload."""
        sample_import_result.payload.items = []
        with patch(
            "rayforge.image.base_importer.Importer.get_doc_items",
            return_value=sample_import_result,
        ):
            with patch.object(
                file_cmd, "_generate_rich_preview_result"
            ) as mock_gen:
                file_cmd._generate_preview_impl(
                    b"data", "test.png", "image/png", TraceSpec(), 256
                )
                # Should still call the generator, which can handle empty items
                mock_gen.assert_called_once()


class TestLoadFileAsync:
    """Tests for _load_file_async method."""

    @pytest.mark.asyncio
    async def test_load_file_async_success(
        self, file_cmd, sample_import_result
    ):
        """Test successful async file load."""
        with patch(
            "rayforge.image.base_importer.Importer.get_doc_items",
            return_value=sample_import_result,
        ):
            result = await file_cmd._load_file_async(
                Path("tests") / "image" / "svg" / "o.svg",
                "image/svg+xml",
                None,
            )
            assert result is sample_import_result

    @pytest.mark.asyncio
    async def test_load_file_async_failure(self, file_cmd):
        """Test async file load failure."""
        with patch(
            "rayforge.image.base_importer.Importer.get_doc_items",
            return_value=None,
        ):
            result = await file_cmd._load_file_async(
                Path("tests") / "image" / "svg" / "o.svg",
                "image/svg+xml",
                None,
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


class TestGetSupportedImportFilters:
    """Tests for get_supported_import_filters method."""

    def test_get_supported_import_filters(self, file_cmd):
        """Test retrieving supported import filters."""
        filters = file_cmd.get_supported_import_filters()
        assert isinstance(filters, list)
        assert len(filters) > 0

        # Verify structure of first filter
        first = filters[0]
        assert "label" in first
        assert "extensions" in first
        assert "mime_types" in first


class TestGetImporterInfo:
    """Tests for the get_importer_info method."""

    def test_get_info_by_mime(self, file_cmd):
        """Test finding an importer and its features by MIME type."""
        mock_importer = MagicMock()
        mock_importer.features = {ImporterFeature.DIRECT_VECTOR}
        with patch(
            "rayforge.doceditor.file_cmd.importer_by_mime_type",
            {"image/vnd.dxf": mock_importer},
        ):
            cls, features = file_cmd.get_importer_info(
                Path("f.dxf"), "image/vnd.dxf"
            )
            assert cls is mock_importer
            assert features == {ImporterFeature.DIRECT_VECTOR}

    def test_get_info_by_extension(self, file_cmd):
        """Test fallback to extension matching."""
        mock_importer = MagicMock()
        mock_importer.features = {ImporterFeature.BITMAP_TRACING}
        with patch("rayforge.doceditor.file_cmd.importer_by_mime_type", {}):
            with patch(
                "rayforge.doceditor.file_cmd.importer_by_extension",
                {".png": mock_importer},
            ):
                cls, features = file_cmd.get_importer_info(Path("f.png"), None)
                assert cls is mock_importer
                assert features == {ImporterFeature.BITMAP_TRACING}

    def test_get_info_not_found(self, file_cmd):
        """Test case where no importer is found."""
        with patch("rayforge.doceditor.file_cmd.importer_by_mime_type", {}):
            with patch(
                "rayforge.doceditor.file_cmd.importer_by_extension", {}
            ):
                cls, features = file_cmd.get_importer_info(
                    Path("f.txt"), "text/plain"
                )
                assert cls is None
                assert features == set()


class TestAnalyzeImportTarget:
    """Tests for analyze_import_target method."""

    def test_analyze_svg(self, file_cmd):
        """Test that SVG files trigger interactive config."""
        path = Path("test.svg")
        mock_importer = MagicMock()
        mock_importer.features = {
            ImporterFeature.DIRECT_VECTOR,
            ImporterFeature.BITMAP_TRACING,
            ImporterFeature.LAYER_SELECTION,
        }
        with patch(
            "rayforge.doceditor.file_cmd.importer_by_mime_type",
            {"image/svg+xml": mock_importer},
        ):
            action = file_cmd.analyze_import_target(path, "image/svg+xml")
            assert action == ImportAction.INTERACTIVE_CONFIG

    def test_analyze_png(self, file_cmd):
        """Test that PNG files trigger interactive config."""
        path = Path("test.png")
        mock_importer = MagicMock()
        mock_importer.features = {ImporterFeature.BITMAP_TRACING}
        with patch(
            "rayforge.doceditor.file_cmd.importer_by_mime_type",
            {"image/png": mock_importer},
        ):
            action = file_cmd.analyze_import_target(path, "image/png")
            assert action == ImportAction.INTERACTIVE_CONFIG

    def test_analyze_dxf(self, file_cmd):
        """Test that DXF files trigger interactive config (due to layers)."""
        path = Path("test.dxf")
        mock_importer = MagicMock()
        mock_importer.features = {
            ImporterFeature.DIRECT_VECTOR,
            ImporterFeature.LAYER_SELECTION,
        }

        # Test with explicit mime
        with patch(
            "rayforge.doceditor.file_cmd.importer_by_mime_type",
            {"image/vnd.dxf": mock_importer},
        ):
            action = file_cmd.analyze_import_target(path, "image/vnd.dxf")
            assert action == ImportAction.INTERACTIVE_CONFIG

        # Test extension fallback
        with patch(
            "rayforge.doceditor.file_cmd.importer_by_extension",
            {".dxf": mock_importer},
        ):
            action = file_cmd.analyze_import_target(path, None)
            assert action == ImportAction.INTERACTIVE_CONFIG

    def test_analyze_unsupported(self, file_cmd):
        """Test that unknown files return unsupported."""
        path = Path("test.exe")
        # Ensure no importers match
        with patch("rayforge.doceditor.file_cmd.importer_by_mime_type", {}):
            with patch(
                "rayforge.doceditor.file_cmd.importer_by_extension", {}
            ):
                action = file_cmd.analyze_import_target(
                    path, "application/octet-stream"
                )
                assert action == ImportAction.UNSUPPORTED


class TestExecuteBatchImport:
    """Tests for execute_batch_import method."""

    def test_execute_batch_import(self, file_cmd):
        """Test that batch import spawns individual load tasks."""
        files = [Path("test1.png"), Path("test2.jpg")]
        spec = TraceSpec()
        pos = (10.0, 10.0)

        with patch.object(file_cmd, "load_file_from_path") as mock_load:
            file_cmd.execute_batch_import(files, spec, pos)

            assert mock_load.call_count == 2

            # Verify calls
            mock_load.assert_any_call(files[0], "image/png", spec, pos)
            mock_load.assert_any_call(files[1], "image/jpeg", spec, pos)

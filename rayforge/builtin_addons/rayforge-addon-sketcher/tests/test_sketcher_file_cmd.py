import pytest
from pathlib import Path
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.core.source_asset import SourceAsset
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.file_cmd import FileCmd
from rayforge.image.svg.renderer import SVG_RENDERER
from rayforge.shared.tasker.manager import TaskManager
from sketcher.core import Sketch


@pytest.fixture
def context_initializer():
    """Mock context initializer."""
    return MagicMock()


@pytest.fixture
def mock_editor(context_initializer):
    """Provides a DocEditor instance with mocked dependencies."""
    task_manager = MagicMock(spec=TaskManager)
    doc = Doc()
    editor = DocEditor(task_manager, context_initializer, doc)
    yield editor
    editor.cleanup()


@pytest.fixture
def file_cmd(mock_editor, context_initializer):
    """Provides a FileCmd instance for testing."""
    return FileCmd(mock_editor, context_initializer)


@pytest.fixture
def sample_workpiece():
    """Provides a simple WorkPiece instance for testing."""
    return WorkPiece("test_workpiece")


class TestCommitWithSketches:
    """Tests for committing items with sketches."""

    def test_commit_with_sketches(self, file_cmd, sample_workpiece):
        """Test committing items with sketches."""
        source = SourceAsset(
            source_file=Path("test.svg"),
            original_data=b"<svg></svg>",
            renderer=SVG_RENDERER,
        )
        sketch = Sketch(name="Test Sketch")
        filename = Path("test.svg")

        file_cmd._commit_items_to_document(
            [sample_workpiece], source, filename, assets=[sketch]
        )

        assert sketch in file_cmd._editor.doc.get_all_assets()


class TestRoundTripSketch:
    """Tests for round trip with sketch projects."""

    def test_round_trip_sketch(self, file_cmd, tmp_path):
        """Test round trip for project with sketches."""
        import_file = Path(__file__).parent / "assets" / "sketch_project.ryp"
        export_file = tmp_path / "sketch_export.ryp"

        result = file_cmd.load_project_from_path(import_file)
        assert result is True

        sketches = [
            a
            for a in file_cmd._editor.doc.get_all_assets()
            if isinstance(a, Sketch)
        ]
        assert len(sketches) == 1
        assert sketches[0].name == "Rectangle"

        workpieces = file_cmd._editor.doc.all_workpieces
        assert len(workpieces) == 1
        assert workpieces[0].geometry_provider_uid is not None

        result = file_cmd.save_project_to_path(export_file)
        assert result is True
        assert export_file.exists()

        result = file_cmd.load_project_from_path(export_file)
        assert result is True

        sketches = [
            a
            for a in file_cmd._editor.doc.get_all_assets()
            if isinstance(a, Sketch)
        ]
        assert len(sketches) == 1

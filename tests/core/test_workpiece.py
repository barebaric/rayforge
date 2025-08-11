import cairo
import pytest
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple
from rayforge.core.item import DocItem
from rayforge.core.workpiece import WorkPiece
from rayforge.importer.svg import SvgImporter
from rayforge.importer import Importer
from blinker import Signal


@pytest.fixture
def sample_svg_data() -> bytes:
    """Provides a simple SVG with defined dimensions in mm."""
    svg = """
    <svg width="100mm" height="50mm" viewBox="0 0 100 50"
         xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="100" height="50" fill="blue"/>
    </svg>
    """
    return svg.encode("utf-8")


@pytest.fixture
def workpiece_instance(sample_svg_data: bytes) -> WorkPiece:
    """Creates a default WorkPiece instance for testing."""
    return WorkPiece(Path("test_rect.svg"), sample_svg_data, SvgImporter)


class TestWorkPiece:
    def test_initialization(self, workpiece_instance, sample_svg_data):
        wp = workpiece_instance
        assert wp.source_file == Path("test_rect.svg")
        assert wp._data == sample_svg_data
        assert wp.importer_class == SvgImporter
        assert isinstance(wp.importer, SvgImporter)
        assert wp.pos == (0.0, 0.0)
        assert wp.size is None
        assert wp.angle == 0.0
        assert np.array_equal(wp.matrix, np.identity(4))
        assert isinstance(wp.changed, Signal)
        assert isinstance(wp.transform_changed, Signal)

    def test_workpiece_is_docitem(self, workpiece_instance):
        assert isinstance(workpiece_instance, DocItem)
        assert hasattr(workpiece_instance, "get_world_transform")
        assert hasattr(workpiece_instance, "get_all_workpieces")

    def test_get_all_workpieces(self, workpiece_instance):
        assert workpiece_instance.get_all_workpieces() == [workpiece_instance]

    def test_serialization_deserialization(self, workpiece_instance):
        wp = workpiece_instance
        wp.pos = 10.5, 20.2
        wp.set_size(80.0, 40.0)
        wp.angle = 90
        data_dict = wp.to_dict()

        assert isinstance(data_dict["matrix"], list)

        new_wp = WorkPiece.from_dict(data_dict)
        assert isinstance(new_wp, WorkPiece)
        assert new_wp.source_file == wp.source_file
        assert new_wp.pos == pytest.approx(wp.pos)
        assert new_wp.size == wp.size
        assert new_wp.angle == pytest.approx(wp.angle)
        assert np.allclose(new_wp.matrix, wp.matrix)
        assert new_wp.importer_class == wp.importer_class
        # The SVG's natural size should be returned, regardless of bounds.
        assert new_wp.get_default_size(
            bounds_width=1000, bounds_height=1000
        ) == (100.0, 50.0)

    def test_from_file(self, tmp_path: Path, sample_svg_data: bytes):
        p = tmp_path / "sample.svg"
        p.write_bytes(sample_svg_data)
        wp = WorkPiece.from_file(p, SvgImporter)
        assert wp.source_file == p
        assert wp._data == sample_svg_data
        assert isinstance(wp.importer, SvgImporter)

    def test_setters_and_signals(self, workpiece_instance):
        wp = workpiece_instance
        changed_events, transform_events = [], []

        # Connect signals with weak=False to prevent garbage collection of
        # lambdas
        wp.changed.connect(
            lambda sender: changed_events.append(sender), weak=False
        )
        wp.transform_changed.connect(
            lambda sender: transform_events.append(sender), weak=False
        )

        # set_pos should fire transform_changed, NOT changed.
        wp.pos = 10, 20
        assert wp.pos == (10.0, 20.0)
        assert len(changed_events) == 0
        assert len(transform_events) == 1

        # set_size should fire changed.
        wp.set_size(150, 75)
        assert wp.size == (150.0, 75.0)
        assert len(changed_events) == 1
        assert len(transform_events) == 1  # Should not have increased

        # set_angle should fire transform_changed, NOT changed.
        wp.angle = 45
        assert wp.angle == 45.0
        assert len(changed_events) == 1  # Should not have increased
        assert len(transform_events) == 2  # Should have increased

    def test_sizing_and_aspect_ratio(self, workpiece_instance):
        wp = workpiece_instance
        assert wp.get_default_aspect_ratio() == pytest.approx(2.0)
        # get_default_size should return the SVG's natural size.
        assert wp.get_default_size(bounds_width=1000, bounds_height=1000) == (
            100.0,
            50.0,
        )
        # A new workpiece has no size, so get_current_size returns None.
        assert wp.get_current_size() is None

        wp.set_size(80, 20)
        assert wp.get_current_size() == (80.0, 20.0)
        assert wp.get_current_aspect_ratio() == pytest.approx(4.0)

    def test_get_default_size_fallback(self):
        """
        Tests the fallback sizing logic when a importer has no natural size.
        """

        class MockNoSizeImporter(Importer):
            def __init__(self, data: bytes):
                super().__init__(data)

            def get_natural_size(
                self, px_factor: float = 0.0
            ) -> Tuple[Optional[float], Optional[float]]:
                return None, None

            def get_aspect_ratio(self) -> float:
                return 2.0

            def render_to_pixels(
                self, width: int, height: int
            ) -> Optional[cairo.ImageSurface]:
                return None

            def render_chunk(self, *args, **kwargs) -> Generator:
                if False:
                    yield

            def _render_to_vips_image(self, width: int, height: int):
                return None

        wp = WorkPiece(Path("nosize.dat"), b"", MockNoSizeImporter)
        # The size should be calculated based on the provided bounds and
        # aspect ratio.
        assert wp.get_default_size(
            bounds_width=400.0, bounds_height=300.0
        ) == (400.0, 200.0)

    def test_render_for_ops(self, workpiece_instance):
        """Tests that render_for_ops renders at the current size."""
        wp = workpiece_instance
        # Without a size set, render_for_ops should do nothing.
        assert (
            wp.render_for_ops(pixels_per_mm_x=10, pixels_per_mm_y=10) is None
        )

        # Set a size for the workpiece.
        wp.set_size(100, 50)
        surface = wp.render_for_ops(pixels_per_mm_x=10, pixels_per_mm_y=10)

        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 1000  # 100mm * 10px/mm
        assert surface.get_height() == 500  # 50mm * 10px/mm

    def test_render_chunk(self, workpiece_instance):
        """
        Tests that render_chunk yields chunks that correctly tile the full
        image area.
        """
        wp = workpiece_instance
        # Without a size set, render_chunk should yield nothing.
        assert (
            list(wp.render_chunk(pixels_per_mm_x=1, pixels_per_mm_y=1)) == []
        )

        # Set a size for the workpiece.
        wp.set_size(100, 50)
        chunks = list(
            wp.render_chunk(
                pixels_per_mm_x=1,
                pixels_per_mm_y=1,
                max_chunk_width=40,
                max_chunk_height=40,
            )
        )

        assert len(chunks) > 1

        max_x = 0
        max_y = 0
        for chunk_surface, (x_offset, y_offset) in chunks:
            assert isinstance(chunk_surface, cairo.ImageSurface)
            max_x = max(max_x, x_offset + chunk_surface.get_width())
            max_y = max(max_y, y_offset + chunk_surface.get_height())

        # The reconstructed size should match the total rendered size
        # (100x50 px).
        assert max_x == 100
        assert max_y == 50

    def test_dump(self, workpiece_instance, capsys):
        """
        Tests the console output of the dump method.
        """
        workpiece_instance.dump(indent=1)
        captured = capsys.readouterr()

        expected = f"   {workpiece_instance.source_file} {SvgImporter.label}\n"
        assert captured.out == expected

    def test_get_world_transform_simple_translation(self, workpiece_instance):
        wp = workpiece_instance
        wp.pos = 10, 20
        wp.angle = 0
        matrix = wp.get_world_transform()

        # Test a point at the workpiece's local origin (0,0)
        # It should end up at the workpiece's position
        p_in = np.array([0, 0, 0, 1])
        p_out = matrix @ p_in
        assert p_out[0] == pytest.approx(10)
        assert p_out[1] == pytest.approx(20)

    def test_get_world_transform_rotation(self, workpiece_instance):
        wp = workpiece_instance
        wp.pos = 0, 0
        wp.set_size(20, 10)
        wp.angle = 90  # Rotates -90 deg (CW) due to internal negation
        matrix = wp.get_world_transform()

        # The rotation is around the center (10, 5)
        # A point at the local origin (0,0) should be rotated around (10,5)
        p_in = np.array([0, 0, 0, 1])
        p_out = matrix @ p_in

        # Expected: translate to origin -> rotate -> translate back
        # Local origin (0,0) relative to center (10,5) is (-10, -5).
        # Rotated (-10,-5) by -90deg (CW) is (-5, 10).
        # Translated back by (10,5) is (-5+10, 10+5) = (5, 15).
        assert p_out[0] == pytest.approx(5)
        assert p_out[1] == pytest.approx(15)

    def test_get_world_transform_translation_and_rotation(
        self, workpiece_instance
    ):
        wp = workpiece_instance
        wp.pos = 100, 200
        wp.set_size(20, 10)
        wp.angle = 90  # Rotates -90 deg (CW) due to internal negation
        matrix = wp.get_world_transform()

        # A point at the local origin (0,0)
        p_in = np.array([0, 0, 0, 1])
        p_out = matrix @ p_in

        # Rotation part moves it to (5, 15) relative to its own pos
        # Translation part moves it by (100, 200)
        # Final position should be (100+5, 200+15) = (105, 215)
        assert p_out[0] == pytest.approx(105)
        assert p_out[1] == pytest.approx(215)

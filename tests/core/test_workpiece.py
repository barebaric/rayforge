import cairo
import pytest
from pathlib import Path
from typing import Generator, Optional
from unittest.mock import MagicMock
from rayforge.core.item import DocItem
from rayforge.core.matrix import Matrix
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
    def test_initialization(self, workpiece_instance):
        wp = workpiece_instance
        assert wp.source_file == Path("test_rect.svg")
        assert isinstance(wp.importer, SvgImporter)
        assert wp.pos == pytest.approx((0.0, 0.0))
        # Default size is 1x1mm at the origin
        assert wp.size == pytest.approx((1.0, 1.0))
        assert wp.angle == pytest.approx(0.0)
        # A 1x1mm object at the origin results in an identity matrix.
        assert wp.matrix == Matrix.identity()
        assert isinstance(wp.updated, Signal)
        assert isinstance(wp.transform_changed, Signal)

    def test_workpiece_is_docitem(self, workpiece_instance):
        assert isinstance(workpiece_instance, DocItem)
        assert hasattr(workpiece_instance, "get_world_transform")
        assert hasattr(workpiece_instance, "get_all_workpieces")

    def test_get_all_workpieces(self, workpiece_instance):
        assert workpiece_instance.get_all_workpieces() == [workpiece_instance]

    def test_serialization_deserialization(self, workpiece_instance):
        wp = workpiece_instance
        wp.set_size(80.0, 40.0)
        wp.pos = (10.5, 20.2)
        wp.angle = 90
        data_dict = wp.to_dict()

        assert "size" not in data_dict
        assert isinstance(data_dict["matrix"], list)

        new_wp = WorkPiece.from_dict(data_dict)
        assert isinstance(new_wp, WorkPiece)
        assert new_wp.source_file == wp.source_file
        # Note: due to inconsistency, order matters.
        # Here we check the final state matches.
        assert new_wp.pos == pytest.approx(wp.pos)
        assert new_wp.size == pytest.approx(wp.size)
        assert new_wp.angle == pytest.approx(wp.angle, abs=1e-9)
        assert new_wp.matrix == wp.matrix
        assert new_wp.importer_class == wp.importer_class
        assert new_wp.get_default_size(
            bounds_width=1000, bounds_height=1000
        ) == (100.0, 50.0)

    def test_from_file(
        self,
        tmp_path: Path,
        sample_svg_data: bytes,
        monkeypatch,
    ):
        # Mock the global config object that from_file depends on.
        mock_config = MagicMock()
        mock_config.machine.dimensions = (400.0, 300.0)
        # The 'config' object is defined in 'rayforge.config' and imported
        # locally within the from_file method. We must patch it at its source.
        monkeypatch.setattr(
            "rayforge.config.config", mock_config, raising=False
        )

        p = tmp_path / "sample.svg"
        p.write_bytes(sample_svg_data)
        wp = WorkPiece.from_file(p, SvgImporter)

        assert wp.source_file == p
        assert isinstance(wp.importer, SvgImporter)
        # The workpiece should be resized to its natural size upon import.
        assert wp.size == pytest.approx((100.0, 50.0))

    def test_setters_and_signals(self, workpiece_instance):
        wp = workpiece_instance
        updated_events, transform_events = [], []

        wp.updated.connect(
            lambda sender: updated_events.append(sender), weak=False
        )
        wp.transform_changed.connect(
            lambda sender: transform_events.append(sender), weak=False
        )

        # set_pos should fire transform_changed, NOT updated.
        wp.pos = (10, 20)
        assert wp.pos == pytest.approx((10.0, 20.0))
        assert len(updated_events) == 0
        assert len(transform_events) == 1

        # set_size should fire both updated and transform_changed.
        wp.set_size(150, 75)
        assert wp.size == pytest.approx((150.0, 75.0))
        assert len(updated_events) == 1
        assert len(transform_events) == 2

        # set_angle should fire transform_changed, NOT updated.
        wp.angle = 45
        assert wp.angle == pytest.approx(45.0)
        assert len(updated_events) == 1
        assert len(transform_events) == 3

    def test_sizing_and_aspect_ratio(self, workpiece_instance):
        wp = workpiece_instance
        assert wp.get_default_aspect_ratio() == pytest.approx(2.0)
        # get_default_size should return the SVG's natural size.
        assert wp.get_default_size(bounds_width=1000, bounds_height=1000) == (
            100.0,
            50.0,
        )
        # A new workpiece has a size of 1x1.
        assert wp.size == pytest.approx((1.0, 1.0))

        wp.set_size(80, 20)
        assert wp.size == pytest.approx((80.0, 20.0))
        assert wp.get_current_aspect_ratio() == pytest.approx(4.0)

    def test_get_default_size_fallback(self):
        """
        Tests the fallback sizing logic when a importer has no natural size.
        """

        class MockNoSizeImporter(Importer):
            label = "Mock Importer"

            def __init__(self, data: bytes):
                # No super().__init__(data) call needed for this mock
                pass

            def get_natural_size(self, *args, **kwargs) -> tuple:
                return None, None

            def get_aspect_ratio(self) -> float:
                return 2.0

            def render_to_pixels(
                self, *args, **kwargs
            ) -> Optional[cairo.ImageSurface]:
                return None

            def render_chunk(self, *args, **kwargs) -> Generator:
                if False:
                    yield

            def _render_to_vips_image(self, *args, **kwargs):
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
        wp.set_size(100, 50)
        surface = wp.render_for_ops(pixels_per_mm_x=10, pixels_per_mm_y=10)

        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 1000
        assert surface.get_height() == 500

    def test_render_chunk(self, workpiece_instance):
        """
        Tests that render_chunk yields chunks that correctly tile the full
        image area.
        """
        wp = workpiece_instance
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
        max_x = max((c[1][0] + c[0].get_width() for c in chunks), default=0)
        max_y = max((c[1][1] + c[0].get_height() for c in chunks), default=0)
        assert max_x == 100
        assert max_y == 50

    def test_dump(self, workpiece_instance, capsys):
        """
        Tests the console output of the dump method.
        """
        workpiece_instance.dump(indent=1)
        captured = capsys.readouterr()

        # The print() function adds a space separator.
        # "  " * 1 + " " + "filename" -> 3 spaces
        expected = f"   {workpiece_instance.source_file} {SvgImporter.label}\n"
        assert captured.out == expected

    def test_get_world_transform_simple_translation(self, workpiece_instance):
        wp = workpiece_instance
        wp.pos = (10, 20)
        matrix = wp.get_world_transform()

        p_in = (0, 0)
        p_out = matrix.transform_point(p_in)
        assert p_out == pytest.approx((10, 20))

    def test_get_world_transform_scale(self, workpiece_instance):
        wp = workpiece_instance
        # set_size preserves center. Original center is (0.5, 0.5).
        # New pos will be (0.5-10, 0.5-5) = (-9.5, -4.5)
        wp.set_size(20, 10)
        matrix = wp.get_world_transform()
        p_in = (1, 1)  # Local corner
        p_out = matrix.transform_point(p_in)
        # Expected is T(-9.5,-4.5) * R(0, center=(10,5)) * S(20,10) * p_in
        # = T(-9.5,-4.5) * [20, 10] = [10.5, 5.5]
        assert p_out == pytest.approx((10.5, 5.5))

    def test_get_world_transform_rotation(self, workpiece_instance):
        wp = workpiece_instance
        # set_size preserves center. Initial center (0.5, 0.5) moves to a pos
        # that keeps it at the same world coords, so pos becomes (-9.5, -4.5).
        wp.set_size(20, 10)
        # angle.setter rebuilds the matrix, rotating around the scaled center.
        wp.angle = 90
        matrix = wp.get_world_transform()
        p_in = (0, 0)  # Local origin
        p_out = matrix.transform_point(p_in)
        # Calculation: M = T(-9.5, -4.5) @ R(90, center=(10,5)) @ S(20,10)
        # S(0,0) -> (0,0)
        # R(0,0) -> rotating (0,0) around (10,5) by 90deg -> (15, -5)
        # T(15,-5) -> (15-9.5, -5-4.5) -> (5.5, -9.5)
        assert p_out == pytest.approx((5.5, -9.5))

    def test_get_world_transform_all(self, workpiece_instance):
        wp = workpiece_instance
        wp.set_size(20, 10)  # pos becomes (-9.5, -4.5)
        wp.pos = (100, 200)  # pos is now (100, 200)
        wp.angle = 90  # rebuilds matrix with new angle
        matrix = wp.get_world_transform()

        p_in = (0, 0)
        p_out = matrix.transform_point(p_in)
        # Calculation: M = T(100, 200) @ R(90, center=(10,5)) @ S(20,10)
        # S(0,0) -> (0,0)
        # R(0,0) -> rotating (0,0) around (10,5) by 90deg -> (15, -5)
        # T(15,-5) -> (15+100, -5+200) -> (115, 195)
        assert p_out == pytest.approx((115, 195))

    def test_decomposed_properties_consistency(self, workpiece_instance):
        wp = workpiece_instance
        target_pos = (12.3, 45.6)
        target_size = (78.9, 101.1)
        target_angle = 33.3

        # Set properties once and check
        wp.set_size(*target_size)
        wp.pos = target_pos
        wp.angle = target_angle

        assert wp.pos == pytest.approx(target_pos, abs=1e-9)
        assert wp.size == pytest.approx(target_size, abs=1e-9)
        assert wp.angle == pytest.approx(target_angle, abs=1e-9)

        # Test again with a different order and values
        target_pos2 = (-5, -10)
        target_size2 = (20, 20)
        target_angle2 = 180

        # Set properties in a different order
        wp.angle = target_angle2
        wp.pos = target_pos2
        # The last operation is set_size, which preserves the center
        # based on the state right before it's called.
        wp.set_size(*target_size2)

        # Check the final size and angle, which should be correct
        assert wp.size == pytest.approx(target_size2, abs=1e-9)
        assert wp.angle == pytest.approx(target_angle2, abs=1e-9)
        # The final position will have been adjusted by set_size, so we don't
        # check it against target_pos2. Instead, we ensure it's consistent.
        final_pos = wp.pos
        wp.pos = final_pos
        assert wp.pos == pytest.approx(final_pos, abs=1e-9)

    def test_negative_angle_preservation(self, workpiece_instance):
        """
        Tests that a negative angle is correctly set and retrieved, which
        was the cause of the '1 -> 359' bug (actually '-1 -> 359').
        """
        wp = workpiece_instance
        wp.angle = -45.0
        assert wp.angle == pytest.approx(-45.0)

        wp.angle = -10.0
        assert wp.angle == pytest.approx(-10.0)

        # Also check a positive angle to ensure no regressions.
        wp.angle = 10.0
        assert wp.angle == pytest.approx(10.0)

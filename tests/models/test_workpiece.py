import cairo
import pytest
from pathlib import Path
from typing import Generator, Optional, Tuple
from rayforge.models.workpiece import WorkPiece
from rayforge.render.svg import SVGRenderer
from rayforge.render import Renderer  # Base class for type hints
from blinker import Signal


@pytest.fixture
def sample_svg_data() -> bytes:
    """Provides a simple SVG with defined dimensions in mm."""
    svg = """
    <svg width="100mm" height="50mm" viewBox="0 0 100 50" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="100" height="50" fill="blue"/>
    </svg>
    """
    return svg.encode("utf-8")


@pytest.fixture
def workpiece_instance(sample_svg_data: bytes) -> WorkPiece:
    """Creates a default WorkPiece instance for testing."""
    return WorkPiece("test_rect.svg", sample_svg_data, SVGRenderer)


class TestWorkPiece:
    def test_initialization(self, workpiece_instance, sample_svg_data):
        wp = workpiece_instance
        assert wp.name == "test_rect.svg"
        assert wp._data == sample_svg_data
        assert wp.renderer_class == SVGRenderer
        assert isinstance(wp.renderer, SVGRenderer)
        assert wp.pos is None
        assert wp.size is None
        assert wp.angle == 0.0
        assert isinstance(wp.changed, Signal)

    def test_serialization_deserialization(self, workpiece_instance):
        wp = workpiece_instance
        wp.set_pos(10.5, 20.2)
        wp.set_size(80.0, 40.0)
        wp.set_angle(90)
        data_dict = wp.to_dict()

        # The renderer path must be resolvable in the test environment.
        # Here we point to the class in the global scope of this test file.
        data_dict["renderer"] = f"{__name__}.{SVGRenderer.__name__}"

        new_wp = WorkPiece.from_dict(data_dict)
        assert isinstance(new_wp, WorkPiece)
        assert new_wp.name == wp.name
        assert new_wp.pos == wp.pos
        assert new_wp.size == wp.size
        assert new_wp.angle == wp.angle
        assert new_wp.renderer_class == wp.renderer_class
        # The SVG's natural size should be returned, regardless of bounds.
        assert new_wp.get_default_size(
            bounds_width=1000, bounds_height=1000
        ) == (100.0, 50.0)

    def test_from_file(self, tmp_path: Path, sample_svg_data: bytes):
        p = tmp_path / "sample.svg"
        p.write_bytes(sample_svg_data)
        wp = WorkPiece.from_file(str(p), SVGRenderer)
        assert wp.name == str(p)
        assert wp._data == sample_svg_data
        assert isinstance(wp.renderer, SVGRenderer)

    def test_setters_and_signals(self, workpiece_instance):
        wp = workpiece_instance
        pos_events, size_events, angle_events, changed_events = [], [], [], []

        # Connect signals with weak=False to prevent garbage collection of lambdas
        wp.pos_changed.connect(
            lambda sender: pos_events.append(sender), weak=False
        )
        wp.size_changed.connect(
            lambda sender: size_events.append(sender), weak=False
        )
        wp.angle_changed.connect(
            lambda sender: angle_events.append(sender), weak=False
        )
        wp.changed.connect(
            lambda sender: changed_events.append(sender), weak=False
        )

        wp.set_pos(10, 20)
        assert wp.pos == (10.0, 20.0)
        assert len(pos_events) == 1
        assert len(changed_events) == 1

        wp.set_size(150, 75)
        assert wp.size == (150.0, 75.0)
        assert len(size_events) == 1
        assert len(changed_events) == 2

        wp.set_angle(45)
        assert wp.angle == 45.0
        assert len(angle_events) == 1
        assert len(changed_events) == 3

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
        Tests the fallback sizing logic when a renderer has no natural size.
        """

        class MockNoSizeRenderer(Renderer):
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

        wp = WorkPiece("nosize.dat", b"", MockNoSizeRenderer)
        # The size should be calculated based on the provided bounds and aspect ratio.
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
        Tests that render_chunk yields chunks that correctly tile the full image area.
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

        # The reconstructed size should match the total rendered size (100x50 px).
        assert max_x == 100
        assert max_y == 50

    def test_dump(self, workpiece_instance, capsys):
        """
        Tests the console output of the dump method.
        """
        workpiece_instance.dump(indent=1)
        captured = capsys.readouterr()

        expected_output = f"   {workpiece_instance.name} {SVGRenderer.label}\n"
        assert captured.out == expected_output

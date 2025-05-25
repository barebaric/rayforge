import pytest
import os
import io
import cairo
import xml.etree.ElementTree as ET
from rayforge.render.svg import SVGRenderer, parse_length, to_mm

# Fixtures
@pytest.fixture
def basic_svg():
    return b'''<svg xmlns="http://www.w3.org/2000/svg" 
                width="100mm" height="50mm" viewBox="0 0 100 50">
                <rect width="100" height="50" fill="red"/>
              </svg>'''

@pytest.fixture
def no_dimensions_svg():
    return b'''<svg xmlns="http://www.w3.org/2000/svg" 
                  viewBox="0 0 80 40">
                  <rect width="80" height="40" fill="blue"/>
               </svg>'''

@pytest.fixture
def transparent_svg():
    return b'''<svg xmlns="http://www.w3.org/2000/svg" 
                 width="200" height="200">
                 <rect x="50" y="50" width="100" height="100" fill="green"/>
               </svg>'''

# Utility functions
def create_temp_svg(tmp_path, content):
    path = tmp_path / "test.svg"
    path.write_bytes(content)
    return path

# Test cases
class TestSVGRenderer:
    def test_parse_length(self):
        assert parse_length("100mm") == (100.0, "mm")
        assert parse_length("50.5cm") == (50.5, "cm")
        assert parse_length("200") == (200.0, "px")

    def test_to_mm_conversion(self):
        assert to_mm(1, "cm") == 10.0
        assert to_mm(1, "in") == 25.4
        assert to_mm(5, "mm") == 5.0
        with pytest.raises(ValueError):
            to_mm(10, "px")

    def test_get_natural_size(self, basic_svg):
        width, height = SVGRenderer.get_natural_size(basic_svg)
        assert width == 100.0
        assert height == 50.0

    def test_get_natural_size_missing_dimensions(self, no_dimensions_svg):
        width, height = SVGRenderer.get_natural_size(no_dimensions_svg)
        assert width is None
        assert height is None

    def test_get_aspect_ratio(self, basic_svg):
        ratio = SVGRenderer.get_aspect_ratio(basic_svg)
        assert ratio == pytest.approx(2.0, abs=0.01)

    def test_render_workpiece(self, basic_svg):
        surface = SVGRenderer.render_workpiece(basic_svg, pixels_per_mm=(10, 10))
        assert isinstance(surface, cairo.ImageSurface)
        
        # Calculate expected width in pixels (100mm at 96 DPI)
        expected_width = int(100 * 10)
        assert surface.get_width() == expected_width

    def test_get_margins(self, transparent_svg):
        margins = SVGRenderer._get_margins(transparent_svg)
        assert len(margins) == 4
        # Should have 25% margins on all sides (50px/200px)
        assert margins == pytest.approx((0.25, 0.25, 0.25, 0.25), abs=0.01)

    def test_crop_to_content(self, transparent_svg):
        cropped = SVGRenderer.prepare(transparent_svg)
        root = ET.fromstring(cropped)
        viewbox = list(map(float, root.get("viewBox").split()))
        assert viewbox == pytest.approx([50.0, 50.0, 100.0, 100.0], 2)
        assert root.get("width") == "102.0px"  # Expect unit to be appended
        assert root.get("height") == "102.0px"

    def test_render_chunk_generator(self, tmp_path):
        svg = b'''<svg xmlns="http://www.w3.org/2000/svg" 
                   width="10000px" height="5000px" viewBox="0 0 10000 5000">
                    <rect width="10000" height="5000" fill="yellow"/>
                  </svg>'''
        chunk_count = 0
        for chunk, (x, y) in SVGRenderer.render_chunk(svg,
                                                      10000, 5000,
                                                      3000, 2000,
                                                      overlap_x=0,
                                                      overlap_y=0):
            assert isinstance(chunk, cairo.ImageSurface)
            assert chunk.get_width() <= 3000
            assert chunk.get_height() <= 2000
            chunk_count += 1
        
        # Verify total chunks (3 rows x 4 cols = 12 chunks)
        assert chunk_count == 12

    def test_render_chunk_generator_overlap(self, tmp_path):
        svg = b'''<svg xmlns="http://www.w3.org/2000/svg" 
                   width="1000px" height="500px" viewBox="0 0 1000 500">
                    <rect width="1000" height="500" fill="yellow"/>
                  </svg>'''
        chunks = []
        for chunk, (x, y) in SVGRenderer.render_chunk(svg,
                                                      1000, 500,
                                                      400, 300,
                                                      overlap_x=2,
                                                      overlap_y=2):
            assert isinstance(chunk, cairo.ImageSurface)
            chunks.append((x, y, chunk.get_width(), chunk.get_height()))
        
        # Verify total chunks (2 rows x 3 cols = 6 chunks)
        assert chunks == [
            (0, 0, 402, 302),   (400, 0, 402, 302),   (800, 0, 200, 302),
            (0, 300, 402, 200), (400, 300, 402, 200), (800, 300, 200, 200)
        ]

    def test_invalid_svg_handling(self):
        invalid_svg = b"<svg>invalid content"
        with pytest.raises(Exception):
            SVGRenderer.render_workpiece(invalid_svg)

    def test_edge_cases(self):
        # Empty SVG
        empty_svg = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
        assert SVGRenderer.get_natural_size(empty_svg) == (None, None)

        # SVG with percentage units
        percent_svg = b'''<svg xmlns="http://www.w3.org/2000/svg" 
                         width="100%" height="50%"></svg>'''
        assert SVGRenderer.get_natural_size(percent_svg) == (None, None)

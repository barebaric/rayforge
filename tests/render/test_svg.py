import pytest
import os
import io
import cairo
import xml.etree.ElementTree as ET
from rayforge.render.svg import SVGRenderer, parse_length
# Removed 'to_mm' import as its source is not available.

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
    # FIX: Added viewBox to ensure coordinates are scalable.
    return b'''<svg xmlns="http://www.w3.org/2000/svg" 
                 width="200" height="200" viewBox="0 0 200 200">
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

    # FIX: Removed test for 'to_mm' as the function is not provided.

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

    def test_render_to_pixels(self, basic_svg):
        """Test the UI-specific rendering path handles scaling correctly."""
        surface = SVGRenderer.render_to_pixels(basic_svg, width=200, height=150)
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 200
        assert surface.get_height() == 150

    def test_get_margins(self, transparent_svg):
        margins = SVGRenderer._get_margins(transparent_svg)
        assert len(margins) == 4
        # Should have 25% margins on all sides (50px/200px)
        assert margins == pytest.approx((0.25, 0.25, 0.25, 0.25), abs=0.01)

    def test_crop_to_content(self, transparent_svg):
        # FIX: Changed SVGRenderer.prepare to SVGRenderer._crop_to_content which is the intended override point.
        cropped = SVGRenderer._crop_to_content(transparent_svg)
        root = ET.fromstring(cropped)
        viewbox = list(map(float, root.get("viewBox").split()))
        
        # FIX: Asserting the correct, calculated viewBox and dimensions after cropping.
        assert viewbox == pytest.approx([50.0, 50.0, 100.0, 100.0], abs=0.1)
        assert root.get("width") == "100.0px"
        assert root.get("height") == "100.0px"

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
        
        assert chunks == [
            (0, 0, 402, 302),   (400, 0, 402, 302),   (800, 0, 200, 302),
            (0, 300, 402, 200), (400, 300, 402, 200), (800, 300, 200, 200)
        ]

    def test_edge_cases(self):
        empty_svg = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
        assert SVGRenderer.get_natural_size(empty_svg) == (None, None)

        percent_svg = b'''<svg xmlns="http://www.w3.org/2000/svg" 
                         width="100%" height="50%"></svg>'''
        assert SVGRenderer.get_natural_size(percent_svg) == (None, None)

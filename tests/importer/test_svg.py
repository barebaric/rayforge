import pytest
import cairo
from rayforge.importer.svg import SvgImporter, parse_length


@pytest.fixture
def basic_svg_importer():
    svg_data = b"""<svg xmlns="http://www.w3.org/2000/svg" 
                width="100mm" height="50mm" viewBox="0 0 100 50">
                <rect width="100" height="50" fill="red"/>
              </svg>"""
    return SvgImporter(svg_data)


@pytest.fixture
def transparent_svg_importer():
    svg_data = b"""<svg xmlns="http://www.w3.org/2000/svg" 
                 width="200px" height="200px" viewBox="0 0 200 200">
                 <rect x="50" y="50" width="100" height="100" fill="green"/>
               </svg>"""
    return SvgImporter(svg_data)


class TestSVGImporter:
    def test_parse_length(self):
        assert parse_length("100mm") == (100.0, "mm")
        assert parse_length("50.5cm") == (50.5, "cm")
        assert parse_length("200") == (200.0, "px")

    def test_get_natural_size(self, basic_svg_importer):
        width, height = basic_svg_importer.get_natural_size()
        assert width == 100.0
        assert height == 50.0

    def test_get_natural_size_with_margins(self, transparent_svg_importer):
        # transparent_svg is 200px wide, content is 100px wide.
        # to_mm for 'px' with px_factor=0 is undefined, so the size is based
        # on content.
        # This test relies on the internal _get_margins logic.
        width, height = transparent_svg_importer.get_natural_size(
            px_factor=1.0
        )
        assert width == pytest.approx(100.0)  # 200 * (1 - 0.25 - 0.25)
        assert height == pytest.approx(100.0)

    def test_get_aspect_ratio(self, basic_svg_importer):
        ratio = basic_svg_importer.get_aspect_ratio()
        assert ratio == pytest.approx(2.0)

    def test_render_to_pixels(self, basic_svg_importer):
        surface = basic_svg_importer.render_to_pixels(width=200, height=100)
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 200
        assert surface.get_height() == 100

    def test_get_margins(self, transparent_svg_importer):
        margins = transparent_svg_importer._get_margins()
        assert len(margins) == 4
        assert margins == pytest.approx((0.25, 0.25, 0.25, 0.25), abs=0.01)

    def test_render_chunk_generator(self):
        svg = b"""<svg xmlns="http://www.w3.org/2000/svg" 
                   width="1000px" height="500px" viewBox="0 0 1000 500">
                    <rect width="1000" height="500" fill="yellow"/>
                  </svg>"""
        importer = SvgImporter(svg)
        chunks = list(importer.render_chunk(1000, 500, 400, 300))
        assert len(chunks) == 6  # 3 cols x 2 rows

    def test_edge_cases(self):
        empty_svg = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
        importer = SvgImporter(empty_svg)
        assert importer.get_natural_size() == (None, None)

import pytest
import io
import ezdxf
import xml.etree.ElementTree as ET
from rayforge.importer.dxf import DxfImporter


# Fixtures
@pytest.fixture
def empty_dxf_importer():
    doc = ezdxf.new()
    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


@pytest.fixture
def line_dxf_importer():
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4  # Millimeters
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 50))
    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


@pytest.fixture
def circle_dxf_importer():
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4  # Millimeters
    msp = doc.modelspace()
    msp.add_circle(center=(50, 50), radius=25)
    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


@pytest.fixture
def inches_dxf_importer():
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 1  # Inches
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 1))  # 1 inch line
    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


# Utility
def get_svg_from_importer(importer):
    return importer._svg_importer.raw_data


def count_svg_elements(importer, tag_name):
    svg_bytes = get_svg_from_importer(importer)
    if (
        not svg_bytes
        or svg_bytes == b'<svg xmlns="http://www.w3.org/2000/svg"/>'
    ):
        return 0
    root = ET.fromstring(svg_bytes)
    return len(root.findall(f".//{{http://www.w3.org/2000/svg}}{tag_name}"))


# Test cases
class TestDXFImporter:
    def test_empty_dxf(self, empty_dxf_importer):
        svg_data = get_svg_from_importer(empty_dxf_importer)
        assert count_svg_elements(empty_dxf_importer, "line") == 0
        assert empty_dxf_importer.get_natural_size() == (None, None)

    def test_line_conversion(self, line_dxf_importer):
        assert count_svg_elements(line_dxf_importer, "line") == 1
        svg_data = get_svg_from_importer(line_dxf_importer)
        line = ET.fromstring(svg_data).find(
            ".//{http://www.w3.org/2000/svg}line"
        )
        assert line.get("x1") == "0.0"
        assert line.get("y1") == "0.0"
        assert line.get("x2") == "100.0"
        assert line.get("y2") == "50.0"

    def test_circle_conversion(self, circle_dxf_importer):
        svg_data = get_svg_from_importer(circle_dxf_importer)
        circle = ET.fromstring(svg_data).find(
            ".//{http://www.w3.org/2000/svg}circle"
        )
        assert circle.get("cx") == "50.0"
        assert circle.get("cy") == "50.0"
        assert circle.get("r") == "25.0"

    def test_unit_conversion(self, inches_dxf_importer):
        svg_data = get_svg_from_importer(inches_dxf_importer)
        line = ET.fromstring(svg_data).find(
            ".//{http://www.w3.org/2000/svg}line"
        )
        assert float(line.get("x2")) == pytest.approx(25.4)
        assert float(line.get("y2")) == pytest.approx(25.4)

    def test_get_natural_size(self, line_dxf_importer):
        width, height = line_dxf_importer.get_natural_size()
        assert width == pytest.approx(100.0)
        assert height == pytest.approx(50.0)

    def test_get_aspect_ratio(self, line_dxf_importer):
        ratio = line_dxf_importer.get_aspect_ratio()
        assert ratio == pytest.approx(2.0)

    def test_render_to_pixels(self, line_dxf_importer):
        surface = line_dxf_importer.render_to_pixels(width=200, height=100)
        assert surface is not None
        assert surface.get_width() == 200
        assert surface.get_height() == 100

    def test_render_chunk_generator(self, line_dxf_importer):
        chunks = list(line_dxf_importer.render_chunk(100, 50, 40, 30))
        assert len(chunks) == 6  # 3 cols x 2 rows

    def test_invalid_dxf_handling(self):
        invalid_dxf = b"invalid dxf content"
        with pytest.raises(ValueError):
            DxfImporter(invalid_dxf)

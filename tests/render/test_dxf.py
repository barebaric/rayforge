import pytest
import io
import ezdxf
import xml.etree.ElementTree as ET
from rayforge.render.dxf import DXFRenderer

# Fixtures
@pytest.fixture
def empty_dxf():
    doc = ezdxf.new()
    doc.header['$INSUNITS'] = 4  # Millimeters
    buffer = io.StringIO()
    doc.write(buffer)
    return buffer.getvalue().encode('utf-8')

@pytest.fixture
def basic_line_dxf():
    doc = ezdxf.new()
    doc.header['$INSUNITS'] = 4  # Millimeters
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 50))
    buffer = io.StringIO()
    doc.write(buffer)
    return buffer.getvalue().encode('utf-8')

@pytest.fixture
def circle_dxf():
    doc = ezdxf.new()
    doc.header['$INSUNITS'] = 4  # Millimeters
    msp = doc.modelspace()
    msp.add_circle(center=(50, 50), radius=25)
    buffer = io.StringIO()
    doc.write(buffer)
    return buffer.getvalue().encode('utf-8')

@pytest.fixture
def text_dxf():
    doc = ezdxf.new()
    doc.header['$INSUNITS'] = 4  # Millimeters
    msp = doc.modelspace()
    msp.add_text("Test", dxfattribs={'height': 5, 'insert': (10, 20)})
    buffer = io.StringIO()
    doc.write(buffer)
    return buffer.getvalue().encode('utf-8')

@pytest.fixture
def block_dxf():
    doc = ezdxf.new()
    doc.header['$INSUNITS'] = 4  # Millimeters
    msp = doc.modelspace()
    block = doc.blocks.new('TEST_BLOCK')
    block.add_line((0, 0), (10, 10))
    msp.add_blockref('TEST_BLOCK', insert=(50, 50))
    buffer = io.StringIO()
    doc.write(buffer)
    return buffer.getvalue().encode('utf-8')

@pytest.fixture
def units_inches_dxf():
    doc = ezdxf.new()
    doc.header['$INSUNITS'] = 1  # Inches (explicitly test conversion)
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 1))  # 1 inch line
    buffer = io.StringIO()
    doc.write(buffer)
    return buffer.getvalue().encode('utf-8')

# Utility functions
def count_svg_elements(svg_bytes, tag_name):
    root = ET.fromstring(svg_bytes)
    return len(root.findall(f'.//{{http://www.w3.org/2000/svg}}{tag_name}'))

# Test cases
class TestDXFRenderer:
    def test_empty_dxf(self, empty_dxf):
        svg_data = DXFRenderer.prepare(empty_dxf)
        root = ET.fromstring(svg_data)
        assert root.tag.endswith('svg')
        assert count_svg_elements(svg_data, 'line') == 0

    def test_line_conversion(self, basic_line_dxf):
        svg_data = DXFRenderer.prepare(basic_line_dxf)
        assert count_svg_elements(svg_data, 'line') == 1
        line = ET.fromstring(svg_data).find('.//{http://www.w3.org/2000/svg}line')
        assert line.get('x1') == '0.0'
        assert line.get('y1') == '0.0'
        assert line.get('x2') == '100.0'
        assert line.get('y2') == '50.0'

    def test_circle_conversion(self, circle_dxf):
        svg_data = DXFRenderer.prepare(circle_dxf)
        circle = ET.fromstring(svg_data).find('.//{http://www.w3.org/2000/svg}circle')
        assert circle.get('cx') == '50.0'
        assert circle.get('cy') == '50.0'
        assert circle.get('r') == '25.0'

    def test_text_conversion(self, text_dxf):
        svg_data = DXFRenderer.prepare(text_dxf)
        text = ET.fromstring(svg_data).find('.//{http://www.w3.org/2000/svg}text')
        assert text.get('x') == '10.0'
        assert text.get('y') == '20.0'
        assert text.get('font-size') == '5.0mm'
        assert text.text == 'Test'

    def test_block_conversion(self, block_dxf):
        svg_data = DXFRenderer.prepare(block_dxf)
        assert count_svg_elements(svg_data, 'line') == 1
        
        # Find nested group with block transform
        root = ET.fromstring(svg_data)
        groups = root.findall('.//{http://www.w3.org/2000/svg}g')
        assert len(groups) >= 2
        block_group = groups[1]  # First nested group
        assert 'translate(50.0 50.0)' in block_group.get('transform')

    def test_unit_conversion(self, units_inches_dxf):
        svg_data = DXFRenderer.prepare(units_inches_dxf)
        line = ET.fromstring(svg_data).find('.//{http://www.w3.org/2000/svg}line')
        assert line.get('x2') == '25.4'
        assert line.get('y2') == '25.4'

    def test_get_natural_size(self, basic_line_dxf):
        svg_data = DXFRenderer.prepare(basic_line_dxf)
        width, height = DXFRenderer.get_natural_size(svg_data)
        assert width == 100.0
        assert height == 50.0

    def test_get_aspect_ratio(self, basic_line_dxf):
        svg_data = DXFRenderer.prepare(basic_line_dxf)
        ratio = DXFRenderer.get_aspect_ratio(svg_data)
        assert ratio == pytest.approx(2.0, abs=0.1)

    def test_render_workpiece(self, basic_line_dxf):
        data = DXFRenderer.prepare(basic_line_dxf)
        surface = DXFRenderer.render_workpiece(data, width=500, height=250)
        assert surface.get_width() == 500
        assert surface.get_height() == 250

    def test_render_chunk_generator(self, basic_line_dxf):
        chunk_count = 0
        data = DXFRenderer.prepare(basic_line_dxf)
        for chunk, (x, y) in DXFRenderer.render_chunk(data,
                                                      1000, 500,
                                                      300, 200,
                                                      overlap_x=10,
                                                      overlap_y=10):
            assert chunk.get_width() <= 310  # 300 + overlap
            assert chunk.get_height() <= 210
            chunk_count += 1
        assert chunk_count == 12

    def test_coordinate_transform(self, basic_line_dxf):
        svg_data = DXFRenderer.prepare(basic_line_dxf)
        group = ET.fromstring(svg_data).find('.//{http://www.w3.org/2000/svg}g')
        assert 'matrix(1 0 0 -1' in group.get('transform')

    def test_invalid_dxf_handling(self):
        invalid_dxf = b"invalid dxf content"
        with pytest.raises(Exception):
            DXFRenderer.render_workpiece(invalid_dxf)

    def test_edge_cases(self):
        # DXF with no modelspace entities
        doc = ezdxf.new()
        doc.header['$INSUNITS'] = 4  # Millimeters
        buffer = io.StringIO()
        doc.write(buffer)
        dxf_data = buffer.getvalue().encode('utf-8')
        svg_data = DXFRenderer.prepare(dxf_data)
        assert DXFRenderer.get_natural_size(svg_data) == (None, None)

        # DXF with invalid units
        doc = ezdxf.new()
        doc.header['$INSUNITS'] = 99  # Invalid unit code
        buffer = io.StringIO()
        doc.write(buffer)
        dxf_data = buffer.getvalue().encode('utf-8')
        svg_data = DXFRenderer.prepare(dxf_data)
        assert DXFRenderer.get_natural_size(svg_data) == (None, None)

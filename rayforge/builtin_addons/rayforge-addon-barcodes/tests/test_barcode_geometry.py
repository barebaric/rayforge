import pytest
from barcodes.encoders import Encoded, Symbology, encode
from barcodes.geometry import BarcodeStyle, barcode_loops, orient_by_nesting
from raygeo.geo.shape.polygon import get_polygon_signed_area

SQUARE = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
INNER = [(1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0)]
RING = Encoded(
    modules=(
        (True, True, True),
        (True, False, True),
        (True, True, True),
    )
)


def _areas(loops):
    return sorted(get_polygon_signed_area(loop) for loop in loops)


def _bbox(loops):
    xs = [x for loop in loops for x, _ in loop]
    ys = [y for loop in loops for _, y in loop]
    return min(xs), min(ys), max(xs), max(ys)


def test_orient_by_nesting():
    hole, outer = orient_by_nesting([INNER[::-1], SQUARE[::-1]])
    assert get_polygon_signed_area(outer) > 0
    assert get_polygon_signed_area(hole) < 0


def test_modules_merge_into_outline_with_hole():
    loops = barcode_loops(RING, Symbology.QR, BarcodeStyle(module_mm=2.0))
    assert _areas(loops) == pytest.approx([-4.0, 36.0])
    assert _bbox(loops) == pytest.approx((0.0, 0.0, 6.0, 6.0))


def test_invert_adds_a_dark_quiet_zone():
    style = BarcodeStyle(module_mm=1.0, quiet_zone=2, invert=True)
    loops = barcode_loops(RING, Symbology.QR, style)
    assert _bbox(loops) == pytest.approx((0.0, 0.0, 7.0, 7.0))
    assert sum(get_polygon_signed_area(loop) for loop in loops) == (
        pytest.approx(49.0 - 8.0)
    )


def test_one_dimensional_bars_get_the_bar_height():
    encoded = encode(Symbology.CODE128, "A")
    style = BarcodeStyle(module_mm=0.5, bar_height_mm=12.0)
    loops = barcode_loops(encoded, Symbology.CODE128, style)
    x0, y0, x1, y1 = _bbox(loops)
    assert (y1 - y0) == pytest.approx(12.0)
    assert (x1 - x0) == pytest.approx(encoded.width * 0.5)
    dark = sum(encoded.modules[0]) * 0.5 * 12.0
    assert sum(get_polygon_signed_area(loop) for loop in loops) == (
        pytest.approx(dark)
    )


def test_text_goes_below_and_ean_guards_reach_into_it():
    encoded = encode(Symbology.EAN13, "590123412345")
    plain = barcode_loops(
        encoded, Symbology.EAN13, BarcodeStyle(module_mm=0.33)
    )
    with_text = barcode_loops(
        encoded, Symbology.EAN13, BarcodeStyle(module_mm=0.33, show_text=True)
    )
    plain_height = _bbox(plain)[3] - _bbox(plain)[1]
    text_height = _bbox(with_text)[3] - _bbox(with_text)[1]
    assert text_height > plain_height
    assert _bbox(with_text)[1] == pytest.approx(0.33, abs=0.05)
    assert len(with_text) > len(plain)
    assert all(get_polygon_signed_area(loop) != 0 for loop in with_text)


def test_qr_ignores_bar_height_and_text():
    encoded = encode(Symbology.QR, "x")
    style = BarcodeStyle(module_mm=1.0, bar_height_mm=99.0, show_text=True)
    loops = barcode_loops(encoded, Symbology.QR, style)
    x0, y0, x1, y1 = _bbox(loops)
    assert (x1 - x0, y1 - y0) == pytest.approx((21.0, 21.0))

"""
Turns encoded modules into closed outlines in millimetres (Y up).

Dark modules are merged into polygons with holes. The winding of every
contour is then normalized, so even-odd and non-zero filling agree.
"""

from dataclasses import dataclass

from raygeo.geo import Geometry
from raygeo.geo.shape.polygon import get_polygons_union
from raygeo.geo.shape.text import FontConfig, text_to_geometry

from .encoders import Encoded, Symbology

Loop = list[tuple[float, float]]

TEXT_HEIGHT_MODULES = 8.0
TEXT_GAP_MODULES = 1.0
TEXT_TOLERANCE_MM = 0.01
EAN_FIRST_DIGIT_X = -4.0
EAN_LEFT_HALF = (3, 45)
EAN_RIGHT_HALF = (50, 92)


@dataclass(frozen=True)
class BarcodeStyle:
    """
    How a symbol is drawn. ``quiet_zone`` is in modules and only drawn
    when ``invert`` is set: then the zone is the dark border around a
    light code. ``bar_height_mm`` and ``show_text`` apply to 1D codes.
    """

    module_mm: float = 0.5
    bar_height_mm: float = 15.0
    quiet_zone: int = 0
    show_text: bool = False
    invert: bool = False


def loops_to_geometry(loops: list[Loop]) -> Geometry:
    """Closed contours from a list of loops."""
    geometry = Geometry()
    for loop in loops:
        x, y = loop[0]
        geometry.move_to(x, y)
        for x, y in loop[1:]:
            geometry.line_to(x, y)
        geometry.close_path()
    return geometry


def _rect(x: float, y: float, w: float, h: float) -> Loop:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _runs(row: tuple[bool, ...]) -> list[tuple[int, int]]:
    runs, start = [], None
    for x, dark in enumerate(row + (False,)):
        if dark and start is None:
            start = x
        elif not dark and start is not None:
            runs.append((start, x - start))
            start = None
    return runs


def _text_loops(text: str, centre_x: float, top: float, height: float):
    if not text:
        return []
    geometry = text_to_geometry(text, FontConfig())
    x0, y0, x1, y1 = geometry.rect()
    if y1 <= y0:
        return []
    scale = height / (y1 - y0)
    dx = centre_x - (x0 + x1) / 2.0 * scale
    dy = top - y1 * scale
    return [
        [(x * scale + dx, y * scale + dy) for x, y in polygon]
        for polygon in geometry.to_polygons(TEXT_TOLERANCE_MM / scale)
        if len(polygon) >= 3
    ]


def _human_readable(
    encoded: Encoded, symbology: Symbology, m: float
) -> list[Loop]:
    """
    The text under a 1D code. EAN-13 puts the first digit left of the
    symbol and six digits under each half, between the guard bars.
    """
    top = (TEXT_HEIGHT_MODULES - TEXT_GAP_MODULES) * m
    height = (TEXT_HEIGHT_MODULES - 2 * TEXT_GAP_MODULES) * m
    text = encoded.text
    if symbology is not Symbology.EAN13:
        return _text_loops(text, encoded.width * m / 2.0, top, height)
    groups = (
        (text[0], EAN_FIRST_DIGIT_X),
        (text[1:7], sum(EAN_LEFT_HALF) / 2.0),
        (text[7:], sum(EAN_RIGHT_HALF) / 2.0),
    )
    loops: list[Loop] = []
    for digits, centre in groups:
        loops += _text_loops(digits, centre * m, top, height)
    return loops


def _module_rects(
    encoded: Encoded, symbology: Symbology, style: BarcodeStyle
) -> tuple[list[Loop], float]:
    """Dark module rectangles and the height of the module area."""
    m = style.module_mm
    if symbology is Symbology.QR:
        height = encoded.height * m
        rects = [
            _rect(x * m, height - (y + 1) * m, w * m, m)
            for y, row in enumerate(encoded.modules)
            for x, w in _runs(row)
        ]
        return rects, height
    text_space = 0.0
    if style.show_text:
        text_space = (TEXT_HEIGHT_MODULES + TEXT_GAP_MODULES) * m
    bars = style.bar_height_mm
    rects = []
    for x, w in _runs(encoded.modules[0]):
        guard = x in encoded.guards and text_space > 0
        extra = text_space / 2.0 if guard else 0.0
        rects.append(_rect(x * m, text_space - extra, w * m, bars + extra))
    return rects, text_space + bars


def barcode_geometry(
    encoded: Encoded, symbology: Symbology, style: BarcodeStyle
) -> Geometry:
    """The dark areas of a symbol as closed contours, in mm."""
    rects, height = _module_rects(encoded, symbology, style)
    loops: list[Loop] = [
        [(float(x), float(y)) for x, y in polygon]
        for polygon in get_polygons_union(rects)
    ]
    if symbology is not Symbology.QR and style.show_text and encoded.text:
        loops += _human_readable(encoded, symbology, style.module_mm)
    if style.invert:
        q = style.quiet_zone * style.module_mm
        width = encoded.width * style.module_mm
        loops = [[(x + q, y + q) for x, y in loop] for loop in loops]
        loops.append(_rect(0.0, 0.0, width + 2 * q, height + 2 * q))
    return loops_to_geometry(loops).normalize_winding_orders()

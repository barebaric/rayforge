"""
Turns encoded modules into closed outlines in millimetres (Y up).

Dark modules are merged into polygons with holes. Every loop is then
oriented by its nesting depth, so even-odd and non-zero filling agree.
"""

from dataclasses import dataclass

from raygeo.geo.shape.polygon import (
    get_polygon_signed_area,
    get_polygons_union,
    is_point_inside_polygon,
)
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


def _probe(loop: Loop) -> tuple[float, float]:
    (x1, y1), (x2, y2) = loop[0], loop[1]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def orient_by_nesting(loops: list[Loop]) -> list[Loop]:
    """
    Orients each loop counter-clockwise at even nesting depth and
    clockwise at odd depth, so nested loops alternate between solid and
    hole however they were drawn.
    """
    oriented = []
    for index, loop in enumerate(loops):
        probe = _probe(loop)
        depth = sum(
            1
            for other_index, other in enumerate(loops)
            if other_index != index and is_point_inside_polygon(probe, other)
        )
        counter_clockwise = get_polygon_signed_area(loop) > 0
        if counter_clockwise != (depth % 2 == 0):
            loop = loop[::-1]
        oriented.append(loop)
    return oriented


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


def barcode_loops(
    encoded: Encoded, symbology: Symbology, style: BarcodeStyle
) -> list[Loop]:
    """The dark areas of a symbol as oriented loops, in mm."""
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
    return orient_by_nesting(loops)

"""Generate the path optimization figures for the website docs.

Renders what the Optimize transformer does to a cutting job: travel
moves before and after optimization, and where the cut planner enters
a closed path with the best start point and prefer corners options.
The toolpaths are run through the real raygeo optimizer, so the
figures show what Rayforge produces. The images carry no translatable
words; titles live in the markdown that embeds them.
"""

import math
from itertools import pairwise

from PIL import Image, ImageDraw
from raygeo.ops import Ops
from raygeo.ops.transform.optimize import OptimizeSpec
from raygeo.ops.types import CommandType

from . import canvas, style

CARD_A = (340, 240)
CARD_B = (232, 200)
CARD_GAP = 30
MARGIN = 24
SCALE_A = 2.0
SCALE_B = 1.6
DESIGN_A = (150.0, 100.0)
DESIGN_B = (120.0, 100.0)

CUT_WIDTH = 2
DOT_RADIUS = 4
SEAM_GAP = 9

BEFORE_NAME = "path-optimization-before.webp"
AFTER_NAME = "path-optimization-after.webp"
START_NAME = "path-optimization-start-point.webp"


def _circle(cx: float, cy: float, r: float, segments: int = 28) -> list:
    return [
        (
            cx + r * math.cos(2 * math.pi * i / segments),
            cy + r * math.sin(2 * math.pi * i / segments),
        )
        for i in range(segments)
    ]


def _star(
    cx: float, cy: float, outer: float, inner: float, points: int = 5
) -> list:
    vertices = []
    for i in range(2 * points):
        radius = outer if i % 2 == 0 else inner
        angle = -math.pi / 2 + math.pi * i / points
        vertices.append(
            (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        )
    return vertices


def _polygon(
    cx: float, cy: float, r: float, sides: int, rotation: float = math.pi / 2
) -> list:
    return [
        (
            cx + r * math.cos(rotation + 2 * math.pi * i / sides),
            cy - r * math.sin(rotation + 2 * math.pi * i / sides),
        )
        for i in range(sides)
    ]


def _arc(
    cx: float,
    cy: float,
    r: float,
    start: float,
    end: float,
    segments: int,
) -> list:
    return [
        (
            cx + r * math.cos(start + (end - start) * i / segments),
            cy + r * math.sin(start + (end - start) * i / segments),
        )
        for i in range(1, segments + 1)
    ]


def _stadium(
    cx: float, cy: float, w: float, h: float, segments: int = 8
) -> list:
    radius = h / 2
    left = cx - w / 2 + radius
    right = cx + w / 2 - radius
    return (
        [(left, cy - radius), (right, cy - radius)]
        + _arc(right, cy, radius, -math.pi / 2, math.pi / 2, segments)
        + [(left, cy + radius)]
        + _arc(left, cy, radius, math.pi / 2, 3 * math.pi / 2, segments)
    )


def _quad_curve(p0: tuple, ctrl: tuple, p1: tuple, segments: int) -> list:
    points = []
    for i in range(segments + 1):
        t = i / segments
        u = 1.0 - t
        points.append(
            (
                u * u * p0[0] + 2 * u * t * ctrl[0] + t * t * p1[0],
                u * u * p0[1] + 2 * u * t * ctrl[1] + t * t * p1[1],
            )
        )
    return points


def _triquetra(cx: float, cy: float, r: float) -> list:
    corners = _polygon(cx, cy, r, 3)
    vertices = []
    for i in range(3):
        a = corners[i]
        b = corners[(i + 1) % 3]
        ctrl = (
            cx + ((a[0] + b[0]) / 2 - cx) * 0.35,
            cy + ((a[1] + b[1]) / 2 - cy) * 0.35,
        )
        curve = _quad_curve(a, ctrl, b, 6)
        vertices.extend(curve if i == 0 else curve[1:])
    return vertices


SHAPES_A = [
    _circle(30, 25, 14),
    _star(78, 24, 16, 7),
    _polygon(122, 26, 15, 3),
    _polygon(30, 70, 13, 4),
    _polygon(75, 70, 14, 6),
    _stadium(118, 70, 34, 14),
]
DRAWN_ORDER = [4, 0, 5, 1, 3, 2]

SEED_B = _circle(48, 84, 7, 12)
TRIQUETRA = _triquetra(62, 42, 30)
DRAWN_START_INDEX = 16


def _build_ops(loops: list) -> Ops:
    ops = Ops()
    for vertices in loops:
        ops.move_to(*vertices[0])
        for x, y in vertices[1:]:
            ops.line_to(x, y)
        if vertices[-1] != vertices[0]:
            ops.line_to(*vertices[0])
    return ops


def _optimized(loops: list, **kwargs) -> Ops:
    ops = _build_ops(loops)
    spec = OptimizeSpec(
        allow_flip=True,
        preserve_first=False,
        preserve_order=[],
        merge_scanlines=None,
        **kwargs,
    )
    ops.apply_transformers([spec])
    return ops


def _split_loops(ops: Ops) -> list:
    loops = []
    previous = None
    for i in range(ops.len()):
        command = ops.command_type(i)
        endpoint = ops.endpoint(i)
        if command == CommandType.MOVE_TO:
            loops.append({"travel_from": previous, "points": [endpoint]})
        elif loops:
            loops[-1]["points"].append(endpoint)
        previous = endpoint
    return loops


def _scaler(origin: tuple, scale: float):
    def scale_point(point: tuple) -> tuple:
        return (origin[0] + point[0] * scale, origin[1] + point[1] * scale)

    return scale_point


def _centered_origin(design: tuple, scale: float, card: tuple) -> tuple:
    return (
        MARGIN + (card[0] - design[0] * scale) / 2,
        MARGIN + (card[1] - design[1] * scale) / 2,
    )


def _draw_travel(draw: ImageDraw.ImageDraw, start: tuple, end: tuple) -> None:
    canvas.draw_dashed_line(draw, start, end, fill=style.TRAVEL, dash=5)
    direction = (end[0] - start[0], end[1] - start[1])
    canvas.draw_arrow_head(draw, end, direction, fill=style.TRAVEL)


def _draw_loop(
    draw: ImageDraw.ImageDraw,
    points: list,
    scale,
    seam: bool = False,
    dot: bool = False,
) -> None:
    px = [scale(p) for p in points]
    closed = points[0] == points[-1]
    segments = list(pairwise(px))
    if closed and seam and len(segments) > 1:
        segments[-1] = _shorten_at_end(segments[-1])
    for a, b in segments:
        draw.line([a, b], fill=style.LINE, width=CUT_WIDTH)
    if dot:
        _draw_dot(draw, px[0])


def _shorten_at_end(segment: tuple) -> tuple:
    a, b = segment
    length = math.dist(a, b)
    if length == 0:
        return segment
    shift = min(SEAM_GAP, length / 3)
    return (
        a,
        (
            b[0] + (a[0] - b[0]) * shift / length,
            b[1] + (a[1] - b[1]) * shift / length,
        ),
    )


def _draw_dot(draw: ImageDraw.ImageDraw, center: tuple) -> None:
    x, y = center
    draw.ellipse(
        [x - DOT_RADIUS, y - DOT_RADIUS, x + DOT_RADIUS, y + DOT_RADIUS],
        fill=style.START,
    )


def _card(card: tuple) -> tuple:
    img = Image.new(
        "RGB", (card[0] + 2 * MARGIN, card[1] + 2 * MARGIN), style.BG
    )
    draw = ImageDraw.Draw(img)
    canvas.draw_card(
        draw, (MARGIN, MARGIN, MARGIN + card[0], MARGIN + card[1])
    )
    return img, draw


def _draw_loops(
    draw: ImageDraw.ImageDraw, ops: Ops, scale, seam_from: int = 0
) -> None:
    loops = _split_loops(ops)
    for index, loop in enumerate(loops):
        if loop["travel_from"] is not None:
            _draw_travel(
                draw,
                scale(loop["travel_from"]),
                scale(loop["points"][0]),
            )
        if index >= seam_from:
            _draw_loop(draw, loop["points"], scale, seam=True, dot=True)
        else:
            _draw_faint_loop(draw, loop["points"], scale)


def _draw_faint_loop(draw: ImageDraw.ImageDraw, points: list, scale) -> None:
    px = [scale(p) for p in points]
    for a, b in pairwise(px):
        draw.line([a, b], fill=style.GUIDE, width=1)


def _render_order_figure(ops: Ops) -> Image.Image:
    img, draw = _card(CARD_A)
    scale = _scaler(_centered_origin(DESIGN_A, SCALE_A, CARD_A), SCALE_A)
    loops = _split_loops(ops)
    for loop in loops:
        if loop["travel_from"] is not None:
            _draw_travel(
                draw,
                scale(loop["travel_from"]),
                scale(loop["points"][0]),
            )
        _draw_loop(draw, loop["points"], scale)
    return img


def _render_start_card(ops: Ops) -> Image.Image:
    img, draw = _card(CARD_B)
    scale = _scaler(_centered_origin(DESIGN_B, SCALE_B, CARD_B), SCALE_B)
    _draw_loops(draw, ops, scale, seam_from=1)
    return img


def _rotate_loop(vertices: list, start_index: int) -> list:
    closed = vertices[0] == vertices[-1]
    body = vertices[:-1] if closed else vertices
    rotated = body[start_index:] + body[:start_index]
    return rotated + [rotated[0]] if closed else rotated


def _numbered_sheet(cards: list) -> Image.Image:
    width = 2 * MARGIN + len(cards) * CARD_B[0] + (len(cards) - 1) * CARD_GAP
    height = 2 * MARGIN + CARD_B[1] + 30
    sheet = Image.new("RGB", (width, height), style.BG)
    label_font = style.find_font("DejaVuSans.ttf", 16)
    draw = ImageDraw.Draw(sheet)
    for index, card_img in enumerate(cards):
        x = MARGIN + index * (CARD_B[0] + CARD_GAP)
        sheet.paste(card_img, (x - MARGIN, 0))
        draw.text(
            (x + CARD_B[0] / 2, MARGIN + CARD_B[1] + 6),
            str(index + 1),
            fill=style.MUTED,
            font=label_font,
            anchor="ma",
        )
    return sheet


def _start_sheet() -> Image.Image:
    drawn = _rotate_loop(TRIQUETRA, DRAWN_START_INDEX)
    loops = [SEED_B, drawn]
    cards = [
        _render_start_card(_build_ops(loops)),
        _render_start_card(_optimized(loops, best_start_point=True)),
        _render_start_card(
            _optimized(loops, best_start_point=True, prefer_corners=True)
        ),
    ]
    return _numbered_sheet(cards)


def generate() -> list:
    drawn = [SHAPES_A[i] for i in DRAWN_ORDER]
    return [
        (BEFORE_NAME, _render_order_figure(_build_ops(drawn))),
        (AFTER_NAME, _render_order_figure(_optimized(drawn))),
        (START_NAME, _start_sheet()),
    ]
